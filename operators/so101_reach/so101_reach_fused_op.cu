/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <holoscan/logger/logger.hpp>

#include "operators/so101_reach/so101_forward_kinematics.h"
#include "operators/so101_reach/so101_reach_fused_op.h"

namespace {

constexpr std::size_t kObservationWidth = 20U;
constexpr std::size_t kActionWidth = 4U;
constexpr std::int64_t kMaximumGridSizePerAxis = 100;
constexpr auto kDistanceLogPeriod = std::chrono::seconds(2);
constexpr auto kStallWarningPeriod = std::chrono::seconds(1);

float linspace(double minimum, double maximum, std::int64_t index, std::int64_t count) {
  if (count == 1) return static_cast<float>((minimum + maximum) * 0.5);
  return static_cast<float>(minimum + static_cast<double>(index) * (maximum - minimum) /
                                          static_cast<double>(count - 1));
}

float distance(const std::array<float, 3U>& first, const std::array<float, 3U>& second) {
  const float dx = first[0U] - second[0U];
  const float dy = first[1U] - second[1U];
  const float dz = first[2U] - second[2U];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void require_cuda(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string{message} + ": " + cudaGetErrorString(result));
  }
}

struct CudaBufferDeleter {
  void operator()(float* pointer) const noexcept {
    if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
  }
};

using CudaBuffer = std::unique_ptr<float, CudaBufferDeleter>;

CudaBuffer allocate_cuda_buffer(std::size_t element_count, const char* message) {
  float* pointer = nullptr;
  require_cuda(cudaMalloc(&pointer, element_count * sizeof(float)), message);
  return CudaBuffer{pointer};
}

void require_tensor(const nvinfer1::ICudaEngine& engine, const char* name,
                    nvinfer1::TensorIOMode mode, int expected_width) {
  if (engine.getTensorIOMode(name) != mode ||
      engine.getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
    throw std::runtime_error(std::string{"The SO-101 policy tensor '"} + name +
                             "' must have the required float32 direction.");
  }
  const nvinfer1::Dims shape = engine.getTensorShape(name);
  if (shape.nbDims != 2 || shape.d[0] != 1 || shape.d[1] != expected_width) {
    throw std::runtime_error(std::string{"The SO-101 policy tensor '"} + name +
                             "' must have shape [1," + std::to_string(expected_width) + "].");
  }
}

}  // namespace

namespace isaac::os::ops {

bool validate_so101_reach_params(const So101ReachParams& params) {
  const auto finite = [](const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  const auto valid_range = [&finite](const std::vector<double>& range) {
    return range.size() == 2U && finite(range) && range[0U] <= range[1U];
  };
  const bool valid_grid_size =
      params.goal_grid_nx > 0 && params.goal_grid_nx <= kMaximumGridSizePerAxis &&
      params.goal_grid_ny > 0 && params.goal_grid_ny <= kMaximumGridSizePerAxis &&
      params.goal_grid_nz > 0 && params.goal_grid_nz <= kMaximumGridSizePerAxis;
  return !params.onnx_path.empty() && params.default_joint_position.size() == 6U &&
         finite(params.default_joint_position) && valid_range(params.goal_pos_x_range) &&
         valid_range(params.goal_pos_y_range) && valid_range(params.goal_pos_z_range) &&
         valid_grid_size && std::isfinite(params.goal_hold_time) && params.goal_hold_time > 0.0 &&
         std::isfinite(params.goal_settle_time) && params.goal_settle_time >= 0.0 &&
         std::isfinite(params.reach_threshold) && params.reach_threshold > 0.0 &&
         std::isfinite(params.action_clip) && params.action_clip > 0.0 &&
         std::isfinite(params.inference_period) && params.inference_period > 0.0;
}

std::vector<std::array<float, 3U>> build_so101_reach_goal_grid(const So101ReachParams& params) {
  if (!validate_so101_reach_params(params)) {
    throw std::invalid_argument("The SO-101 reach goal grid configuration is invalid.");
  }
  std::vector<std::array<float, 3U>> goals;
  goals.reserve(
      static_cast<std::size_t>(params.goal_grid_nx * params.goal_grid_ny * params.goal_grid_nz));
  for (std::int64_t z = 0; z < params.goal_grid_nz; ++z) {
    for (std::int64_t y = 0; y < params.goal_grid_ny; ++y) {
      for (std::int64_t x = 0; x < params.goal_grid_nx; ++x) {
        goals.push_back({
            linspace(params.goal_pos_x_range[0U], params.goal_pos_x_range[1U], x,
                     params.goal_grid_nx),
            linspace(params.goal_pos_y_range[0U], params.goal_pos_y_range[1U], y,
                     params.goal_grid_ny),
            linspace(params.goal_pos_z_range[0U], params.goal_pos_z_range[1U], z,
                     params.goal_grid_nz),
        });
      }
    }
  }
  return goals;
}

std::array<float, kObservationWidth> build_so101_reach_observation(
    const std::array<float, 6U>& joint_position, const So101ReachParams& params,
    const std::array<float, 3U>& goal_position,
    const std::array<float, kActionWidth>& last_action) {
  std::array<float, kObservationWidth> observation{};
  for (std::size_t index = 0; index < kActionWidth; ++index) {
    observation[index] =
        joint_position[index] - static_cast<float>(params.default_joint_position[index]);
  }
  // The two non-actuated joints were held at their defaults during training. Their relative
  // position slots are contractually zero even if the hardware reports a different position.
  observation[4U] = 0.0F;
  observation[5U] = 0.0F;
  const std::array<float, 3U> fingertip_position = so101_fingertip_position(joint_position);
  observation[6U] = fingertip_position[0U];
  observation[7U] = fingertip_position[1U];
  observation[8U] = fingertip_position[2U];
  observation[9U] = goal_position[0U];
  observation[10U] = goal_position[1U];
  observation[11U] = goal_position[2U];
  observation[12U] = 1.0F;
  observation[13U] = 0.0F;
  observation[14U] = 0.0F;
  observation[15U] = 0.0F;
  std::copy(last_action.begin(), last_action.end(), observation.begin() + 16U);
  return observation;
}

std::array<float, kActionWidth> sanitize_so101_reach_action(
    const std::array<float, kActionWidth>& action, float action_clip) {
  std::array<float, kActionWidth> sanitized{};
  if (!std::all_of(action.begin(), action.end(),
                   [](float value) { return std::isfinite(value); })) {
    return sanitized;
  }
  std::transform(action.begin(), action.end(), sanitized.begin(), [action_clip](float value) {
    return std::clamp(value, -action_clip, action_clip);
  });
  return sanitized;
}

void So101ReachFusedOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(state, "in_state").queue_depth(1U);
  spec.output(delta, "out_delta");
}

holoscan::Contract So101ReachFusedOp::contract() const {
  const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(params().inference_period));
  holoscan::Contract contract;
  contract.trigger(holoscan::OnClock{.period = period});
  contract.input(state, holoscan::LatestWithin{.max_age = period});
  contract.on_missing(state, holoscan::Skip{});
  return contract;
}

void So101ReachFusedOp::start() {
  const So101ReachParams& config = params();
  if (!validate_so101_reach_params(config)) {
    HOLOSCAN_LOG_ERROR(
        "The SO-101 reach policy requires a valid ONNX path, joint defaults, goal grid, timing, "
        "reach threshold, action clip, and inference period.");
    throw std::runtime_error(
        "The SO-101 reach policy requires a valid ONNX path, joint defaults, goal grid, timing, "
        "reach threshold, action clip, and inference period.");
  }
  auto result = isaac::TensorRTEngine::create("So101ReachFusedOp", config.onnx_path);
  if (!result) {
    throw result.error();
  }
  tensorrt_engine_ = std::move(*result);
  auto& engine = tensorrt_engine_->engine();
  require_tensor(engine, "obs", nvinfer1::TensorIOMode::kINPUT,
                 static_cast<int>(kObservationWidth));
  require_tensor(engine, "actions", nvinfer1::TensorIOMode::kOUTPUT,
                 static_cast<int>(kActionWidth));
  int device_id = 0;
  require_cuda(cudaGetDevice(&device_id), "Unable to select the SO-101 policy CUDA device");
  cuda_stream_pool_ = std::make_unique<holoscan::CudaStreamPool>(holoscan::CudaStreamPool::Config{
      .device_id = device_id,
      .reserved_size = 1U,
      .max_size = 1U,
      .nvtx_identifier = "so101_reach",
  });
  auto initialized = cuda_stream_pool_->initialize_pool();
  if (!initialized) {
    throw std::move(initialized).error();
  }
  auto stream = cuda_stream_pool_->acquire();
  if (!stream) {
    throw std::move(stream).error();
  }
  cuda_stream_ = std::move(*stream);
  CudaBuffer observation_device =
      allocate_cuda_buffer(kObservationWidth, "Unable to allocate the SO-101 observation");
  CudaBuffer actions_device =
      allocate_cuda_buffer(kActionWidth, "Unable to allocate the SO-101 actions");
  if (!tensorrt_engine_->context().setTensorAddress("obs", observation_device.get()) ||
      !tensorrt_engine_->context().setTensorAddress("actions", actions_device.get())) {
    throw std::runtime_error("Unable to bind the fixed SO-101 policy tensors.");
  }

  goals_ = build_so101_reach_goal_grid(config);
  active_goal_index_ = 0U;
  goals_reached_ = 0;
  goal_settling_ = false;
  observation_host_.fill(0.0F);
  actions_host_.fill(0.0F);
  last_action_.fill(0.0F);
  goal_start_time_ = Clock::now();
  last_distance_log_ = {};
  inferences_ = 0U;
  HOLOSCAN_LOG_INFO(
      "Prepared a {}x{}x{} goal grid with {} goals; hold time {:.1f}s, settle time {:.1f}s, "
      "and tolerance {:.3f}m.",
      config.goal_grid_nx, config.goal_grid_ny, config.goal_grid_nz, goals_.size(),
      config.goal_hold_time, config.goal_settle_time, config.reach_threshold);
  const auto& goal = goals_.front();
  HOLOSCAN_LOG_INFO("Starting goal 1/{} at [{:.4f}, {:.4f}, {:.4f}] (initial).", goals_.size(),
                    goal[0U], goal[1U], goal[2U]);
  observation_device_ = observation_device.release();
  actions_device_ = actions_device.release();
}

void So101ReachFusedOp::stop() {
  if (inferences_ == 0U) {
    HOLOSCAN_LOG_WARN("The reach policy ran no inferences.");
  } else {
    const double active_seconds =
        std::chrono::duration<double>(last_inference_time_ - first_inference_time_).count();
    const auto scheduled =
        static_cast<std::uint64_t>(std::llround(active_seconds / params().inference_period)) + 1U;
    HOLOSCAN_LOG_INFO(
        "Ran {} reach policy inferences in {:.1f}s and skipped {} without a fresh joint state.",
        inferences_, active_seconds, scheduled > inferences_ ? scheduled - inferences_ : 0U);
    const auto idle = Clock::now() - last_inference_time_;
    if (idle >= kStallWarningPeriod) {
      HOLOSCAN_LOG_WARN(
          "The reach policy received no fresh joint state in the last {:.1f}s before stopping.",
          std::chrono::duration<double>(idle).count());
    }
  }
  if (observation_device_ != nullptr) {
    cudaFree(observation_device_);
    observation_device_ = nullptr;
  }
  if (actions_device_ != nullptr) {
    cudaFree(actions_device_);
    actions_device_ = nullptr;
  }
  tensorrt_engine_.reset();
  cuda_stream_ = {};
  cuda_stream_pool_.reset();
}

void So101ReachFusedOp::advance_goal(std::string_view reason) {
  ++active_goal_index_;
  if (active_goal_index_ == goals_.size()) {
    const float percentage =
        100.0F * static_cast<float>(goals_reached_) / static_cast<float>(goals_.size());
    HOLOSCAN_LOG_INFO("Completed all {} goals ({}/{} achieved, {:.1f}%).", goals_.size(),
                      goals_reached_, goals_.size(), percentage);
    active_goal_index_ = 0U;
    goals_reached_ = 0;
  }
  goal_settling_ = false;
  goal_start_time_ = Clock::now();
  const auto& goal = goals_[active_goal_index_];
  HOLOSCAN_LOG_INFO("Starting goal {}/{} at [{:.4f}, {:.4f}, {:.4f}] ({}).",
                    active_goal_index_ + 1U, goals_.size(), goal[0U], goal[1U], goal[2U], reason);
}

const std::array<float, 3U>& So101ReachFusedOp::update_goal(
    const std::array<float, 3U>& fingertip_position) {
  const So101ReachParams& config = params();
  const Clock::time_point now = Clock::now();
  const auto& goal = goals_[active_goal_index_];
  const float goal_distance = distance(fingertip_position, goal);

  if (goal_settling_) {
    if (goal_distance >= static_cast<float>(config.reach_threshold)) {
      goal_settling_ = false;
    } else {
      const double settled_for = std::chrono::duration<double>(now - settle_start_time_).count();
      if (settled_for >= config.goal_settle_time) {
        ++goals_reached_;
        HOLOSCAN_LOG_INFO("Reached goal {}/{} (distance {:.4f}m, {}/{} completed).",
                          active_goal_index_ + 1U, goals_.size(), goal_distance, goals_reached_,
                          active_goal_index_ + 1U);
        advance_goal("reached");
      }
      return goals_[active_goal_index_];
    }
  }

  const double elapsed = std::chrono::duration<double>(now - goal_start_time_).count();
  if (elapsed >= config.goal_hold_time) {
    HOLOSCAN_LOG_WARN(
        "Goal {}/{} timed out at [{:.4f}, {:.4f}, {:.4f}] after {:.1f}s; fingertip position "
        "was [{:.4f}, {:.4f}, {:.4f}] and the final distance was {:.4f}m.",
        active_goal_index_ + 1U, goals_.size(), goal[0U], goal[1U], goal[2U], config.goal_hold_time,
        fingertip_position[0U], fingertip_position[1U], fingertip_position[2U], goal_distance);
    advance_goal("timeout");
  } else if (goal_distance < static_cast<float>(config.reach_threshold)) {
    if (config.goal_settle_time > 0.0) {
      goal_settling_ = true;
      settle_start_time_ = now;
      HOLOSCAN_LOG_INFO("Goal {}/{} is within tolerance; holding for {:.1f}s.",
                        active_goal_index_ + 1U, goals_.size(), config.goal_settle_time);
    } else {
      ++goals_reached_;
      HOLOSCAN_LOG_INFO("Reached goal {}/{} (distance {:.4f}m, {}/{} completed).",
                        active_goal_index_ + 1U, goals_.size(), goal_distance, goals_reached_,
                        active_goal_index_ + 1U);
      advance_goal("reached");
    }
  } else if (now - last_distance_log_ >= kDistanceLogPeriod) {
    last_distance_log_ = now;
    HOLOSCAN_LOG_INFO("Distance to goal {}/{}: {:.4f}m.", active_goal_index_ + 1U, goals_.size(),
                      goal_distance);
  }
  return goals_[active_goal_index_];
}

holoscan::expected<void, holoscan::Error> So101ReachFusedOp::compute(
    holoscan::ExecutionContext& context) {
  auto received = state.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  const Clock::time_point inference_time = Clock::now();
  if (inferences_++ == 0U) first_inference_time_ = inference_time;
  last_inference_time_ = inference_time;
  const isaac::StateSampleCpuT& sample = received->data;
  if (sample.position.size() != 6U ||
      !std::all_of(sample.position.begin(), sample.position.end(),
                   [](float position) { return std::isfinite(position); })) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "The SO-101 state sample must contain six finite positions."});
  }

  const So101ReachParams& config = context.params<So101ReachParams>();
  std::array<float, 6U> joint_position{};
  for (std::size_t index = 0U; index < 6U; ++index) {
    joint_position[index] = sample.position[index];
  }
  const std::array<float, 3U> fingertip_position = so101_fingertip_position(joint_position);
  const std::array<float, 3U>& goal_position = update_goal(fingertip_position);
  // The 20-D trained layout is joint_pos_rel(6), ee_position(3), goal_pose(7), last_action(4).
  observation_host_ =
      build_so101_reach_observation(joint_position, config, goal_position, last_action_);

  const cudaStream_t stream = cuda_stream_.native_handle();
  require_cuda(
      cudaMemcpyAsync(observation_device_, observation_host_.data(),
                      observation_host_.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
      "Unable to upload the SO-101 observation");
  if (!tensorrt_engine_->context().enqueueV3(stream)) {
    throw std::runtime_error("The SO-101 TensorRT policy failed to enqueue.");
  }
  require_cuda(
      cudaMemcpyAsync(actions_host_.data(), actions_device_, actions_host_.size() * sizeof(float),
                      cudaMemcpyDeviceToHost, stream),
      "Unable to download the SO-101 actions");
  require_cuda(cudaStreamSynchronize(stream), "The SO-101 TensorRT policy did not complete");

  auto output = delta.allocate();
  if (!output) {
    return holoscan::make_unexpected(std::move(output).error());
  }
  isaac::PolicyDeltaCpuT& message = output->value();
  last_action_ = sanitize_so101_reach_action(actions_host_, static_cast<float>(config.action_clip));
  message.action.assign(last_action_.begin(), last_action_.end());
  message.source_sequence = sample.source_sequence;
  message.fault_epoch = sample.fault_epoch;
  holoscan::EmitOptions options{.capture_time = received->metadata.capture_time()};
  if (received->metadata.trace_id != 0U) {
    options.trace = holoscan::TraceContext::root(received->metadata.trace_id);
  }
  return delta.emit(std::move(*output), options);
}

}  // namespace isaac::os::ops
