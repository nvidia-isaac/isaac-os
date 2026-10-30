/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Disable bitint support in fmt: nvcc cannot parse `_BitInt`.
#define FMT_USE_BITINT 0

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

#include "operators/agile_wbc/agile_wbc_op.h"

namespace {

__global__ void clamp_kernel(float* data, float lo, float hi, int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    data[index] = fminf(fmaxf(data[index], lo), hi);
  }
}

void require_cuda(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("{}: {}.", message, cudaGetErrorString(result));
    throw std::runtime_error(std::string{message} + ": " + cudaGetErrorString(result));
  }
}

std::size_t model_tensor_element_count(nvinfer1::IExecutionContext& context,
                                       const std::string& name) {
  const nvinfer1::Dims dimensions = context.getTensorShape(name.c_str());
  if (dimensions.nbDims < 0) {
    throw std::runtime_error("The model does not expose tensor '" + name + "'.");
  }

  std::size_t element_count = 1U;
  for (int index = 0; index < dimensions.nbDims; ++index) {
    if (dimensions.d[index] <= 0) {
      throw std::runtime_error("The model tensor '" + name + "' has an unresolved dimension.");
    }
    const std::size_t extent = static_cast<std::size_t>(dimensions.d[index]);
    if (element_count > std::numeric_limits<std::size_t>::max() / extent) {
      throw std::runtime_error("The model tensor '" + name + "' is too large.");
    }
    element_count *= extent;
  }
  return element_count;
}

}  // namespace

namespace isaac::os::ops {

void AgileWbcOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(joint_state, "in_joint_state");
  spec.input(imu, "in_imu");
  if (velocity_from_teleop_) {
    spec.input(velocity_commands, "in_velocity_commands").queue_depth(1U);
  }
  spec.output(command, "out_joint_command");
}

holoscan::Contract AgileWbcOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnAligned{.ports = holoscan::ports(joint_state, imu),
                                       .window = std::chrono::nanoseconds{0}});
  contract.input(joint_state, holoscan::Exact{});
  contract.input(imu, holoscan::Exact{});
  if (velocity_from_teleop_) {
    contract.input(velocity_commands,
                   holoscan::LatestWithin{.max_age = std::chrono::nanoseconds::max()});
    contract.on_missing(velocity_commands, holoscan::SafeDefault{});
  }
  return contract;
}

void AgileWbcOp::start() {
  const WbcParams& config = params();
  try {
    validate_wbc_params(config);
  } catch (const std::invalid_argument& error) {
    HOLOSCAN_LOG_ERROR("The WBC configuration is invalid: {}.", error.what());
    throw;
  }

  n_sdk_joints_ = static_cast<int>(config.sdk_joint_order.size());
  const int policy_joint_count = static_cast<int>(config.policy_joint_order.size());
  const int action_joint_count = static_cast<int>(config.action_joint_order.size());
  struct StartGuard {
    AgileWbcOp& op;
    bool armed{true};
    ~StartGuard() {
      if (armed) {
        op.stop();
      }
    }
  } start_guard{*this};

  require_cuda(cudaEventCreateWithFlags(&completion_event_, cudaEventDisableTiming),
               "Unable to create the CUDA event used to track policy execution");

  cudaStream_t stream{};
  require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "Unable to create the CUDA stream for policy initialization");
  struct StreamGuard {
    cudaStream_t stream;
    ~StreamGuard() { cudaStreamDestroy(stream); }
  } stream_guard{stream};

  auto trt_result = isaac::TensorRTEngine::create("AgileWbcOp", config.onnx_path);
  if (!trt_result) {
    throw trt_result.error();
  }
  tensorrt_engine_ = std::move(*trt_result);
  auto& trt_context = tensorrt_engine_->context();

  {
    auto& engine = tensorrt_engine_->engine();
    const int tensor_count = engine.getNbIOTensors();
    for (int index = 0; index < tensor_count; ++index) {
      const char* name = engine.getIOTensorName(index);
      if (engine.getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error(std::string{"The model tensor '"} + name +
                                 "' must contain float32 values.");
      }
      const bool is_action_output = name == config.action_pos_output_name ||
                                    name == config.action_kp_output_name ||
                                    name == config.action_kd_output_name;
      if (engine.getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT && !is_action_output) {
        const std::string tensor_name{name};
        const std::size_t element_count = model_tensor_element_count(trt_context, tensor_name);
        matx::make_tensor(trt_outputs_[name], {static_cast<matx::index_t>(element_count)},
                          matx::MATX_DEVICE_MEMORY, stream);
      }
    }
  }

  std::unordered_map<std::string, std::string> output_to_input;
  for (const HistoryTerm& term : config.history_terms) {
    const int64_t total = term.length > 0 ? term.length * term.dim : term.dim;
    matx::make_tensor(history_tensors_[term.input_name], {total}, matx::MATX_DEVICE_MEMORY, stream);
    require_cuda(cudaMemsetAsync(history_tensors_[term.input_name].Data(), 0,
                                 static_cast<std::size_t>(total) * sizeof(float), stream),
                 "Unable to initialize the policy history");
    output_to_input[term.output_name] = term.input_name;
  }
  require_cuda(cudaStreamSynchronize(stream), "Unable to initialize the policy history");

  std::unordered_map<std::string, int> sdk_index;
  sdk_index.reserve(n_sdk_joints_);
  for (int index = 0; index < n_sdk_joints_; ++index) {
    sdk_index[config.sdk_joint_order[index]] = index;
  }

  std::vector<int> sdk_to_policy(policy_joint_count);
  for (int index = 0; index < policy_joint_count; ++index) {
    sdk_to_policy[index] = sdk_index.at(config.policy_joint_order[index]);
  }
  matx::make_tensor(idx_sdk_to_policy_, {policy_joint_count}, matx::MATX_DEVICE_MEMORY, stream);
  require_cuda(cudaMemcpyAsync(idx_sdk_to_policy_.Data(), sdk_to_policy.data(),
                               static_cast<std::size_t>(policy_joint_count) * sizeof(int),
                               cudaMemcpyHostToDevice, stream),
               "Unable to upload the policy joint order");

  {
    std::unordered_map<int, int> sdk_to_action;
    for (int index = 0; index < action_joint_count; ++index) {
      sdk_to_action[sdk_index.at(config.action_joint_order[index])] = index;
    }
    std::vector<int> sdk_source_index(n_sdk_joints_);
    for (int index = 0; index < n_sdk_joints_; ++index) {
      const auto action = sdk_to_action.find(index);
      sdk_source_index[index] =
          action != sdk_to_action.end() ? action->second : action_joint_count + index;
    }
    matx::make_tensor(idx_sdk_src_, {n_sdk_joints_}, matx::MATX_DEVICE_MEMORY, stream);
    require_cuda(cudaMemcpyAsync(idx_sdk_src_.Data(), sdk_source_index.data(),
                                 static_cast<std::size_t>(n_sdk_joints_) * sizeof(int),
                                 cudaMemcpyHostToDevice, stream),
                 "Unable to upload the command joint order");
  }

  matx::make_tensor(t_quat_w_, {4}, matx::MATX_DEVICE_MEMORY, stream);
  matx::make_tensor(t_ang_vel_b_, {3}, matx::MATX_DEVICE_MEMORY, stream);
  matx::make_tensor(t_vel_cmds_, {3}, matx::MATX_DEVICE_MEMORY, stream);
  matx::make_tensor(t_joint_pos_raw_, {n_sdk_joints_}, matx::MATX_DEVICE_MEMORY, stream);
  matx::make_tensor(t_joint_vel_raw_, {n_sdk_joints_}, matx::MATX_DEVICE_MEMORY, stream);
  matx::make_tensor(t_joint_pos_, {policy_joint_count}, matx::MATX_DEVICE_MEMORY, stream);
  matx::make_tensor(t_joint_vel_, {policy_joint_count}, matx::MATX_DEVICE_MEMORY, stream);

  const std::array<float, 3U> velocity_command{
      static_cast<float>(config.velocity_command_lin_x),
      static_cast<float>(config.velocity_command_lin_y),
      static_cast<float>(config.velocity_command_ang_z),
  };
  require_cuda(
      cudaMemcpyAsync(t_vel_cmds_.Data(), velocity_command.data(),
                      velocity_command.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
      "Unable to upload the velocity command");

  auto* last_actions_tensor = &history_tensors_.at(config.last_actions_input_name);

  {
    std::vector<float> default_position(n_sdk_joints_, 0.0F);
    std::vector<float> default_kp(n_sdk_joints_, 0.0F);
    std::vector<float> default_kd(n_sdk_joints_, 0.0F);
    for (int index = 0; index < n_sdk_joints_; ++index) {
      const std::string& joint = config.sdk_joint_order[index];
      if (const auto value = config.default_joint_positions.find(joint);
          value != config.default_joint_positions.end()) {
        default_position[index] = static_cast<float>(value->second);
      }
      if (const auto value = config.default_kp.find(joint); value != config.default_kp.end()) {
        default_kp[index] = static_cast<float>(value->second);
      }
      if (const auto value = config.default_kd.find(joint); value != config.default_kd.end()) {
        default_kd[index] = static_cast<float>(value->second);
      }
    }

    const matx::index_t combined_count =
        static_cast<matx::index_t>(action_joint_count + n_sdk_joints_);
    const matx::index_t sdk_count = static_cast<matx::index_t>(n_sdk_joints_);
    matx::make_tensor(t_src_pos_, {combined_count}, matx::MATX_DEVICE_MEMORY, stream);
    matx::make_tensor(t_src_kp_, {combined_count}, matx::MATX_DEVICE_MEMORY, stream);
    matx::make_tensor(t_src_kd_, {combined_count}, matx::MATX_DEVICE_MEMORY, stream);
    matx::make_tensor(t_out_pos_, {sdk_count}, matx::MATX_DEVICE_MEMORY, stream);
    matx::make_tensor(t_out_kp_, {sdk_count}, matx::MATX_DEVICE_MEMORY, stream);
    matx::make_tensor(t_out_kd_, {sdk_count}, matx::MATX_DEVICE_MEMORY, stream);
    const std::size_t sdk_bytes = static_cast<std::size_t>(n_sdk_joints_) * sizeof(float);
    require_cuda(cudaMemcpyAsync(t_src_pos_.Data() + action_joint_count, default_position.data(),
                                 sdk_bytes, cudaMemcpyHostToDevice, stream),
                 "Unable to upload the default joint positions");
    require_cuda(cudaMemcpyAsync(t_src_kp_.Data() + action_joint_count, default_kp.data(),
                                 sdk_bytes, cudaMemcpyHostToDevice, stream),
                 "Unable to upload the default proportional gains");
    require_cuda(cudaMemcpyAsync(t_src_kd_.Data() + action_joint_count, default_kd.data(),
                                 sdk_bytes, cudaMemcpyHostToDevice, stream),
                 "Unable to upload the default derivative gains");
  }

  std::unordered_map<std::string, std::pair<float*, std::size_t>> history_sources;
  for (auto& [name, tensor] : trt_outputs_) {
    history_sources[name] = {tensor.Data(), static_cast<std::size_t>(tensor.TotalSize())};
  }
  history_sources[config.action_pos_output_name] = {t_src_pos_.Data(),
                                                    static_cast<std::size_t>(action_joint_count)};
  history_sources[config.action_kp_output_name] = {t_src_kp_.Data(),
                                                   static_cast<std::size_t>(action_joint_count)};
  history_sources[config.action_kd_output_name] = {t_src_kd_.Data(),
                                                   static_cast<std::size_t>(action_joint_count)};
  for (const auto& [output_name, input_name] : output_to_input) {
    const auto source = history_sources.find(output_name);
    if (source == history_sources.end()) {
      throw std::runtime_error("The history output '" + output_name +
                               "' is not an output of the ONNX model.");
    }
    const std::size_t destination_size =
        static_cast<std::size_t>(history_tensors_.at(input_name).TotalSize());
    if (source->second.second != destination_size) {
      throw std::runtime_error("The model output '" + output_name +
                               "' does not match the configured history size.");
    }
  }

  auto set_address = [&](const std::string& name, void* address, std::size_t provided) {
    const std::size_t required = model_tensor_element_count(trt_context, name);
    if (provided != required) {
      throw std::runtime_error("The model tensor '" + name + "' requires " +
                               std::to_string(required) + " values, but the WBC buffer provides " +
                               std::to_string(provided) + ".");
    }
    if (!trt_context.setTensorAddress(name.c_str(), address)) {
      throw std::runtime_error("Unable to bind model tensor '" + name +
                               "'. Check that the configuration matches the ONNX model.");
    }
  };
  set_address(config.quat_input_name, t_quat_w_.Data(),
              static_cast<std::size_t>(t_quat_w_.TotalSize()));
  set_address(config.ang_vel_input_name, t_ang_vel_b_.Data(),
              static_cast<std::size_t>(t_ang_vel_b_.TotalSize()));
  set_address(config.velocity_commands_name, t_vel_cmds_.Data(),
              static_cast<std::size_t>(t_vel_cmds_.TotalSize()));
  set_address(config.joint_pos_input_name, t_joint_pos_.Data(),
              static_cast<std::size_t>(t_joint_pos_.TotalSize()));
  set_address(config.joint_vel_input_name, t_joint_vel_.Data(),
              static_cast<std::size_t>(t_joint_vel_.TotalSize()));
  for (auto& [name, tensor] : history_tensors_) {
    set_address(name, tensor.Data(), static_cast<std::size_t>(tensor.TotalSize()));
  }
  for (auto& [name, tensor] : trt_outputs_) {
    set_address(name, tensor.Data(), static_cast<std::size_t>(tensor.TotalSize()));
  }
  set_address(config.action_pos_output_name, t_src_pos_.Data(), action_joint_count);
  set_address(config.action_kp_output_name, t_src_kp_.Data(), action_joint_count);
  set_address(config.action_kd_output_name, t_src_kd_.Data(), action_joint_count);

  require_cuda(cudaStreamSynchronize(stream), "Unable to finish policy initialization");

  const matx::index_t sdk_count = static_cast<matx::index_t>(n_sdk_joints_);
  matx::make_tensor(t_cmd_pos_host_, {sdk_count}, matx::MATX_HOST_MEMORY);
  matx::make_tensor(t_cmd_kp_host_, {sdk_count}, matx::MATX_HOST_MEMORY);
  matx::make_tensor(t_cmd_kd_host_, {sdk_count}, matx::MATX_HOST_MEMORY);

  require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Unable to begin GPU graph preparation for the policy");
  struct CaptureGuard {
    cudaStream_t stream;
    bool armed{true};
    ~CaptureGuard() {
      if (armed) {
        cudaGraph_t failed_graph{};
        (void)cudaStreamEndCapture(stream, &failed_graph);
        if (failed_graph != nullptr) {
          (void)cudaGraphDestroy(failed_graph);
        }
      }
    }
  } capture_guard{stream};

  matx::cudaExecutor executor{stream};
  (t_joint_pos_ = matx::select(t_joint_pos_raw_, idx_sdk_to_policy_)).run(executor);
  (t_joint_vel_ = matx::select(t_joint_vel_raw_, idx_sdk_to_policy_) *
                  static_cast<float>(config.velocity_scale))
      .run(executor);

  {
    const int element_count = static_cast<int>(last_actions_tensor->TotalSize());
    constexpr int block_size = 128;
    const int grid_size = (element_count + block_size - 1) / block_size;
    clamp_kernel<<<grid_size, block_size, 0, stream>>>(
        last_actions_tensor->Data(), -static_cast<float>(config.action_clamp),
        static_cast<float>(config.action_clamp), element_count);
  }

  if (!trt_context.enqueueV3(stream)) {
    throw std::runtime_error("The policy could not run while preparing the GPU graph.");
  }
  for (const auto& [output_name, input_name] : output_to_input) {
    const auto& [source, element_count] = history_sources.at(output_name);
    require_cuda(cudaMemcpyAsync(history_tensors_.at(input_name).Data(), source,
                                 element_count * sizeof(float), cudaMemcpyDeviceToDevice, stream),
                 "Unable to update the policy history");
  }

  (t_out_pos_ = matx::select(t_src_pos_, idx_sdk_src_)).run(executor);
  (t_out_kp_ = matx::select(t_src_kp_, idx_sdk_src_)).run(executor);
  (t_out_kd_ = matx::select(t_src_kd_, idx_sdk_src_)).run(executor);

  const std::size_t sdk_bytes = static_cast<std::size_t>(n_sdk_joints_) * sizeof(float);
  require_cuda(cudaMemcpyAsync(t_cmd_pos_host_.Data(), t_out_pos_.Data(), sdk_bytes,
                               cudaMemcpyDeviceToHost, stream),
               "Unable to download the joint positions");
  require_cuda(cudaMemcpyAsync(t_cmd_kp_host_.Data(), t_out_kp_.Data(), sdk_bytes,
                               cudaMemcpyDeviceToHost, stream),
               "Unable to download the proportional gains");
  require_cuda(cudaMemcpyAsync(t_cmd_kd_host_.Data(), t_out_kd_.Data(), sdk_bytes,
                               cudaMemcpyDeviceToHost, stream),
               "Unable to download the derivative gains");

  const cudaError_t capture_result = cudaStreamEndCapture(stream, &graph_);
  capture_guard.armed = false;
  require_cuda(capture_result, "Unable to finish GPU graph preparation for the policy");
  require_cuda(cudaGraphInstantiate(&graph_exec_, graph_, 0),
               "Unable to create the prepared GPU graph for the policy");
  start_guard.armed = false;
}

void AgileWbcOp::stop() {
  if (completion_event_ != nullptr) {
    cudaEventDestroy(completion_event_);
    completion_event_ = nullptr;
  }
  if (graph_exec_ != nullptr) {
    cudaGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  if (graph_ != nullptr) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }
  tensorrt_engine_.reset();
  trt_outputs_.clear();
  history_tensors_.clear();
}

holoscan::expected<void, holoscan::Error> AgileWbcOp::compute(holoscan::ExecutionContext& context) {
  auto received_joint_state = joint_state.receive();
  if (!received_joint_state) {
    HOLOSCAN_LOG_ERROR("Unable to receive the WBC joint state: {} (error code {}).",
                       received_joint_state.error().message,
                       static_cast<int>(received_joint_state.error().code));
    return holoscan::make_unexpected(std::move(received_joint_state).error());
  }
  auto received_imu = imu.receive();
  if (!received_imu) {
    HOLOSCAN_LOG_ERROR("Unable to receive the WBC IMU sample: {} (error code {}).",
                       received_imu.error().message, static_cast<int>(received_imu.error().code));
    return holoscan::make_unexpected(std::move(received_imu).error());
  }
  const isaac::JointStateCpuT& joint_state_input = received_joint_state->data;
  const holoscan::schema::ImuT& imu_input = received_imu->data;
  const std::size_t sdk_joint_count = static_cast<std::size_t>(n_sdk_joints_);
  if (joint_state_input.position.size() != sdk_joint_count ||
      joint_state_input.velocity.size() != sdk_joint_count) {
    HOLOSCAN_LOG_ERROR("The WBC joint state has {} positions and {} velocities; {} are required.",
                       joint_state_input.position.size(), joint_state_input.velocity.size(),
                       sdk_joint_count);
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument,
        "The WBC joint state must contain one position and velocity per configured SDK joint."});
  }
  const WbcParams& config = context.params<WbcParams>();
  cudaStream_t stream = context.cuda_stream().get();

  if (velocity_from_teleop_ && !velocity_commands.is_missing()) {
    auto received_velocity = velocity_commands.receive();
    if (!received_velocity) {
      HOLOSCAN_LOG_ERROR("Unable to receive the WBC velocity command: {} (error code {}).",
                         received_velocity.error().message,
                         static_cast<int>(received_velocity.error().code));
      return holoscan::make_unexpected(std::move(received_velocity).error());
    }
    const isaac::TwistCpuT& twist = received_velocity->data;
    const std::array<float, 3U> velocity_command{
        static_cast<float>(twist.linear.x()),
        static_cast<float>(twist.linear.y()),
        static_cast<float>(twist.angular.z()),
    };
    require_cuda(
        cudaMemcpyAsync(t_vel_cmds_.Data(), velocity_command.data(),
                        velocity_command.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
        "Unable to upload the commanded base velocity");
  }

  h_ang_vel_scaled_[0] = imu_input.angular_velocity.x() * static_cast<float>(config.ang_vel_scale);
  h_ang_vel_scaled_[1] = imu_input.angular_velocity.y() * static_cast<float>(config.ang_vel_scale);
  h_ang_vel_scaled_[2] = imu_input.angular_velocity.z() * static_cast<float>(config.ang_vel_scale);
  const std::array<float, 4U> orientation{
      imu_input.orientation.w(),
      imu_input.orientation.x(),
      imu_input.orientation.y(),
      imu_input.orientation.z(),
  };

  require_cuda(cudaMemcpyAsync(t_quat_w_.Data(), orientation.data(),
                               orientation.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
               "Unable to upload the root orientation");
  require_cuda(cudaMemcpyAsync(t_ang_vel_b_.Data(), h_ang_vel_scaled_, sizeof(h_ang_vel_scaled_),
                               cudaMemcpyHostToDevice, stream),
               "Unable to upload the root angular velocity");
  require_cuda(cudaMemcpyAsync(t_joint_pos_raw_.Data(), joint_state_input.position.data(),
                               joint_state_input.position.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream),
               "Unable to upload the joint positions");
  require_cuda(cudaMemcpyAsync(t_joint_vel_raw_.Data(), joint_state_input.velocity.data(),
                               joint_state_input.velocity.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream),
               "Unable to upload the joint velocities");
  require_cuda(cudaGraphLaunch(graph_exec_, stream),
               "Unable to start the prepared GPU graph for the policy");
  require_cuda(cudaEventRecord(completion_event_, stream), "Unable to record policy completion");

  // Avoid a blocking host wakeup in the 1 kHz low-latency loop.
  cudaError_t completion;
  while ((completion = cudaEventQuery(completion_event_)) == cudaErrorNotReady) {
  }
  require_cuda(completion, "The prepared GPU graph failed while running the policy");

  auto output_loan = command.allocate();
  if (!output_loan) {
    HOLOSCAN_LOG_ERROR("Unable to allocate the WBC joint command: {} (error code {}).",
                       output_loan.error().message, static_cast<int>(output_loan.error().code));
    return holoscan::make_unexpected(std::move(output_loan).error());
  }
  isaac::JointCommandCpuT& output = output_loan->value();
  output.position.resize(sdk_joint_count);
  output.velocity.resize(sdk_joint_count);
  output.effort.resize(sdk_joint_count);
  output.kp.resize(sdk_joint_count);
  output.kd.resize(sdk_joint_count);
  std::ranges::fill(output.velocity, 0.0F);
  std::ranges::fill(output.effort, 0.0F);
  const std::size_t sdk_bytes = static_cast<std::size_t>(n_sdk_joints_) * sizeof(float);
  std::memcpy(output.position.data(), t_cmd_pos_host_.Data(), sdk_bytes);
  std::memcpy(output.kp.data(), t_cmd_kp_host_.Data(), sdk_bytes);
  std::memcpy(output.kd.data(), t_cmd_kd_host_.Data(), sdk_bytes);

  holoscan::EmitOptions options{.flags = received_joint_state->metadata.flags};
  const holoscan::TimePoint capture_time = received_joint_state->metadata.capture_time();
  if (capture_time.valid()) {
    options.capture_time = capture_time;
  }
  if (received_joint_state->metadata.frame_id != 0U) {
    options.frame_id = received_joint_state->metadata.frame_id;
  }
  if (received_joint_state->metadata.trace_id != 0U) {
    options.trace = holoscan::TraceContext::root(received_joint_state->metadata.trace_id);
  }
  auto emitted = command.emit(std::move(*output_loan), options);
  if (!emitted) {
    HOLOSCAN_LOG_ERROR("Unable to publish the WBC joint command: {} (error code {}).",
                       emitted.error().message, static_cast<int>(emitted.error().code));
  }
  return emitted;
}

}  // namespace isaac::os::ops
