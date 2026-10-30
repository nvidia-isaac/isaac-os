/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Disable bitint support in fmt: nvcc cannot parse `_BitInt`. Must precede the header (-> holoscan
// -> fmt) include, same as gr00t_wbc_op.cu.
#define FMT_USE_BITINT 0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda.h>
#include <holoscan/core/predicate.hpp>
#include <holoscan/logger/logger.hpp>

#include "operators/gr00t_n1_7/gr00t_n1_7_fused_op.h"

namespace isaac::os::ops {

namespace {

const std::string kOperatorName{"Gr00tN17FusedOp"};
const holoscan::TensorInputSpec kDeviceImage{
    .representation =
        {
            .memory_kind = holoscan::MemoryKind::kCudaDevice,
            .dtype = DLDataType{kDLUInt, 8U, 1U},
            .rank = 3U,
        },
};

// Destroy a one-time initialization or graph-capture stream on every exit path.
struct StreamGuard {
  cudaStream_t s;
  ~StreamGuard() {
    if (s) cudaStreamDestroy(s);
  }
};

void require_cuda(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string{message} + ": " + cudaGetErrorString(result));
  }
}

void require_cuda_driver(CUresult result, const char* message) {
  if (result != CUDA_SUCCESS) {
    const char* detail = nullptr;
    cuGetErrorString(result, &detail);
    throw std::runtime_error(std::string{message} + ": " +
                             (detail ? detail : "Unknown CUDA driver error"));
  }
}

cudaStream_t make_capture_stream(cudaStream_t reference) {
  CUcontext cuda_context = nullptr;
  CUgreenCtx green_context = nullptr;
  require_cuda_driver(cuStreamGetCtx_v2(reference, &cuda_context, &green_context),
                      "Unable to identify the inference stream's CUDA context");
  int priority = 0;
  require_cuda(cudaStreamGetPriority(reference, &priority),
               "Unable to read the inference stream's priority");

  StreamGuard capture{nullptr};
  if (green_context) {
    require_cuda_driver(
        cuGreenCtxStreamCreate(&capture.s, green_context, CU_STREAM_NON_BLOCKING, priority),
        "Unable to create a private capture stream in the inference Green Context");
  } else {
    require_cuda_driver(cuCtxPushCurrent(cuda_context),
                        "Unable to activate the inference CUDA context");
    const auto created = cuStreamCreateWithPriority(&capture.s, CU_STREAM_NON_BLOCKING, priority);
    CUcontext popped = nullptr;
    const auto restored = cuCtxPopCurrent(&popped);
    require_cuda_driver(created, "Unable to create a private inference capture stream");
    require_cuda_driver(restored, "Unable to restore the previous CUDA context");
  }
  return std::exchange(capture.s, nullptr);
}

// TensorRT dtype -> element size in bytes; buffers are sized from TRT binding shapes.
const std::unordered_map<nvinfer1::DataType, size_t> kTrtElemBytes = {
    {nvinfer1::DataType::kFLOAT, 4}, {nvinfer1::DataType::kHALF, 2},
    {nvinfer1::DataType::kINT8, 1},  {nvinfer1::DataType::kINT32, 4},
    {nvinfer1::DataType::kINT64, 8},
};
size_t trt_elem_bytes(nvinfer1::DataType dtype) {
  auto it = kTrtElemBytes.find(dtype);
  if (it == kTrtElemBytes.end()) {
    throw std::runtime_error(
        "A model binding uses an unsupported data type (supported: float32, float16, int8, int32, "
        "int64).");
  }
  return it->second;
}

// crop + drop-alpha (keep first 3 channels) + cast-to-float32. src [srcH, srcW, srcC] row-major HWC
// -> dst [cropH, cropW, 3] f32, matching the torch preprocessor's
// slice(0,0,cropH).slice(1,0,cropW).slice(2,0,3).to(float32).
// Templated on the source element type; the fused op instantiates it for uint8 only.
template <typename T>
__global__ void image_prep_kernel(const T* __restrict__ src, float* __restrict__ dst, int src_w,
                                  int src_c, int crop_h, int crop_w) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= crop_w || y >= crop_h) return;
#pragma unroll
  for (int c = 0; c < 3; ++c)
    dst[(y * crop_w + x) * 3 + c] = static_cast<float>(src[(y * src_w + x) * src_c + c]);
}

// Collapse leading unit dims to reach HWC. Returns false if the tensor is not rank-3 after
// collapsing.
bool extract_hwc(const holoscan::Tensor& img, int64_t& h, int64_t& w, int64_t& c) {
  std::vector<int64_t> s = img.shape();
  while (s.size() > 3 && s.front() == 1) s.erase(s.begin());
  if (s.size() != 3) return false;
  h = s[0];
  w = s[1];
  c = s[2];
  return true;
}

bool valid_inference_period(double seconds) {
  const double max_period_s = static_cast<double>(std::chrono::nanoseconds::max().count()) * 1e-9;
  return std::isfinite(seconds) && seconds >= 1e-9 && seconds < max_period_s;
}

}  // namespace

bool select_gr00t_n17_frame(std::int64_t capture_time_ns, std::int64_t period_ns,
                            std::optional<std::int64_t>& last_period, bool defer_initial_interval) {
  const auto capture_period = capture_time_ns / period_ns;
  // Compare interval indices to skip missed periods without shifting the cadence.
  if (!last_period) {
    last_period = capture_period;
    return !defer_initial_interval;
  }
  if (capture_period <= *last_period) return false;
  last_period = capture_period;
  return true;
}

holoscan::expected<void, holoscan::Error> validate_gr00t_n17_params(const Gr00tN17Params& config) {
  const auto invalid = [](std::string message) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, std::move(message)});
  };
  if (config.inference.models.empty()) {
    return invalid("At least one GR00T N1.7 model must be configured.");
  }
  if (config.sdk_joint_order.empty() ||
      config.sdk_joint_order.size() != static_cast<std::size_t>(config.inference.num_joints)) {
    return invalid("The SDK joint order must match the configured joint count.");
  }
  if (config.joint_order.empty() ||
      config.joint_order.size() != static_cast<std::size_t>(config.num_motors)) {
    return invalid("The output joint order must match the configured motor count.");
  }
  if (config.crop_height <= 0 || config.crop_width <= 0 ||
      config.inference.max_action_horizon <= 0 || config.inference.max_state_dim <= 0 ||
      config.inference.chunk_size <= 0 || config.prediction_horizon <= 0) {
    return invalid("GR00T N1.7 dimensions must be positive.");
  }
  if (!valid_inference_period(config.inference_period)) {
    return invalid("The inference period must fit in a positive nanosecond duration.");
  }
  const std::size_t noise_count = static_cast<std::size_t>(config.inference.max_action_horizon) *
                                  static_cast<std::size_t>(config.inference.max_state_dim);
  if (noise_count % 4U != 0U) {
    return invalid("The GR00T N1.7 noise element count must be divisible by four.");
  }
  for (const Gr00tN17Model& model : config.inference.models) {
    if (model.path.empty() || model.inputs.empty() || model.outputs.empty()) {
      return invalid("Each GR00T N1.7 model needs a path plus input and output bindings.");
    }
  }

  const auto contains_joint = [](const std::vector<std::string>& order, const std::string& joint) {
    return std::ranges::find(order, joint) != order.end();
  };
  for (const auto& [group, joints] : config.state_output_spec) {
    if (group.empty() || joints.empty()) {
      return invalid("Each GR00T N1.7 state group must contain at least one joint.");
    }
    for (const std::string& joint : joints) {
      if (!contains_joint(config.sdk_joint_order, joint)) {
        return invalid("A GR00T N1.7 state group contains an unknown SDK joint.");
      }
    }
  }
  for (const auto& [group, joints] : config.input_spec) {
    if (group.empty() || joints.empty()) {
      return invalid("Each GR00T N1.7 action group must contain at least one joint.");
    }
    for (const std::string& joint : joints) {
      if (!contains_joint(config.joint_order, joint)) {
        return invalid("A GR00T N1.7 action group contains an unknown output joint.");
      }
    }
    for (const auto* gains : {&config.kp, &config.kd}) {
      const auto values = gains->find(group);
      if (values != gains->end() && values->second.size() != 1U &&
          values->second.size() != joints.size()) {
        return invalid("GR00T N1.7 gains must contain one value or one value per group joint.");
      }
      if (values != gains->end() && !std::ranges::all_of(values->second, [](double value) {
            return std::isfinite(value) &&
                   value >= static_cast<double>(std::numeric_limits<float>::lowest()) &&
                   value <= static_cast<double>(std::numeric_limits<float>::max());
          })) {
        return invalid("GR00T N1.7 gains must be finite values representable as float32.");
      }
    }
  }
  return {};
}

holoscan::expected<void, holoscan::Error> validate_gr00t_n17_inputs(
    const Gr00tN17Params& config, const isaac::JointStateCpuT& state,
    const holoscan::Tensor& input_image) {
  const auto invalid = [](std::string message) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, std::move(message)});
  };
  if (state.position.size() < config.sdk_joint_order.size()) {
    return invalid("The GR00T N1.7 joint state does not contain every configured SDK joint.");
  }
  const DLDataType dtype = input_image.dtype();
  if (dtype.code != kDLUInt || dtype.bits != 8U || dtype.lanes != 1U ||
      !input_image.is_contiguous() || input_image.data() == nullptr) {
    return invalid("The GR00T N1.7 image must be a packed uint8 Tensor.");
  }
  std::int64_t height{};
  std::int64_t width{};
  std::int64_t channels{};
  if (!extract_hwc(input_image, height, width, channels) || channels < 3 ||
      height < config.crop_height || width < config.crop_width) {
    return invalid("The GR00T N1.7 image does not satisfy the configured HWC crop.");
  }
  return {};
}

holoscan::expected<isaac::JointCommandChunkCpuT, holoscan::Error> make_joint_command_chunk_cpu(
    std::span<const float> position, std::span<const float> effort, std::span<const float> kp,
    std::span<const float> kd, std::uint32_t joint_count, std::uint32_t action_count,
    std::int64_t image_capture_time_ns) {
  if (joint_count == 0U || action_count == 0U ||
      joint_count > std::numeric_limits<std::size_t>::max() / action_count) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
  }
  const std::size_t value_count = static_cast<std::size_t>(joint_count) * action_count;
  if (position.size() != value_count || effort.size() != value_count || kp.size() != joint_count ||
      kd.size() != joint_count) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
  }
  isaac::JointCommandChunkCpuT result;
  result.position.assign(position.begin(), position.end());
  result.effort.assign(effort.begin(), effort.end());
  result.kp.assign(kp.begin(), kp.end());
  result.kd.assign(kd.begin(), kd.end());
  result.joint_count = joint_count;
  result.action_count = action_count;
  result.image_capture_time_ns = image_capture_time_ns;
  return result;
}

void Gr00tN17FusedOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(joint_state, "in_joint_state").queue_depth(1U);
  spec.input(image, "in_image").queue_depth(1U).expects_tensor(kDeviceImage);
  spec.output(action_chunk, "out_action_chunk");
}

holoscan::Contract Gr00tN17FusedOp::contract() const {
  if (!valid_inference_period(params().inference_period)) {
    throw std::runtime_error("The inference period must fit in a positive nanosecond duration.");
  }
  holoscan::Contract contract;
  if (params().align_observations) {
    contract.trigger(holoscan::OnAligned{.ports = holoscan::ports(joint_state, image),
                                         .window = std::chrono::nanoseconds{0}});
    contract.input(joint_state, holoscan::Exact{});
    contract.input(image, holoscan::Exact{});
    return contract;
  }
  contract.trigger(holoscan::OnPredicate{
      .evaluate_on = holoscan::OnInputUpdate{image},
      .predicate = holoscan::predicate<std::optional<std::int64_t>>(
          "inference-cadence",
          holoscan::bind_predicate(*this, &Gr00tN17FusedOp::select_inference_frame)),
      .firing = holoscan::PredicateFiring::kOnEveryTrueEvaluation,
  });
  contract.input(image, holoscan::Exact{});
  contract.input(joint_state, holoscan::LatestWithin{.max_age = std::chrono::nanoseconds::max()});
  contract.on_missing(joint_state, holoscan::ReuseLast{.max_age = std::chrono::nanoseconds::max()});
  return contract;
}

bool Gr00tN17FusedOp::select_inference_frame(
    const holoscan::PredicateEvaluationContext&,
    const holoscan::PredicateInputView<holoscan::Tensor>& frame,
    std::optional<std::int64_t>& last_period) const noexcept {
  const auto captured = frame.sample().metadata.capture_time();
  const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::duration<double>(params().inference_period))
                             .count();
  return captured.valid() && captured.timestamp_ns >= 0 &&
         select_gr00t_n17_frame(captured.timestamp_ns, period_ns, last_period, false);
}

//  TensorRT plumbing. The CUDA-graph staging path is always on (the fused op is graph-only).

void Gr00tN17FusedOp::load_engines() {
  model_entries_.clear();
  for (std::size_t index = 0; index < params().inference.models.size(); ++index) {
    const Gr00tN17Model& configured = params().inference.models[index];
    ModelEntry model;
    model.path = configured.path;

    const std::string log_prefix =
        "GR00T N1.7 fused '" + kOperatorName + "', model at index " + std::to_string(index);
    auto result = isaac::TensorRTEngine::create(log_prefix, model.path);
    if (!result) throw result.error();
    model.engine = std::move(*result);
    model.input_symbols.insert(configured.inputs.begin(), configured.inputs.end());
    model.output_symbols.insert(configured.outputs.begin(), configured.outputs.end());
    model_entries_.push_back(std::move(model));
  }
}

void Gr00tN17FusedOp::resolve_and_alloc_intermediates() {
  // Any symbol that is neither an operator input ref (input.*) nor an operator output ref
  // (output.*) is an intermediate or noise buffer that this op cudaMalloc's directly.
  const auto needs_alloc = [](const std::string& sym) {
    if (sym.starts_with("input.")) return false;
    if (sym.starts_with("output.")) return false;
    return true;
  };

  const auto alloc_for_output_binding = [&](size_t engine_idx, const std::string& binding_name,
                                            const std::string& symbol) {
    auto& trt_engine = model_entries_[engine_idx].engine->engine();
    const std::string& path = model_entries_[engine_idx].path;

    bool found = false;
    for (int32_t t = 0; t < trt_engine.getNbIOTensors(); ++t) {
      if (std::string(trt_engine.getIOTensorName(t)) == binding_name) {
        found = true;
        if (trt_engine.getTensorIOMode(binding_name.c_str()) != nvinfer1::TensorIOMode::kOUTPUT) {
          throw std::runtime_error("Model file '" + path + "' does not provide '" + binding_name +
                                   "' as an output. Fix the \"outputs\" map for this model.");
        }
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("Model file '" + path + "' does not contain a tensor named '" +
                               binding_name + "'. Fix the \"outputs\" map for this model.");
    }

    const auto dims = trt_engine.getTensorShape(binding_name.c_str());
    size_t n_elems = 1;
    for (int d = 0; d < dims.nbDims; ++d) {
      if (dims.d[d] < 0) {
        throw std::runtime_error("Model file '" + path + "' uses a variable-sized output '" +
                                 binding_name + "'; CUDA-graph replay requires static shapes.");
      }
      n_elems *= static_cast<size_t>(dims.d[d]);
    }

    IntermediateBuffer buf;
    buf.bytes = n_elems * trt_elem_bytes(trt_engine.getTensorDataType(binding_name.c_str()));
    if (cudaMalloc(&buf.ptr, buf.bytes) != cudaSuccess) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                               "' could not reserve GPU memory for intermediate data '" + symbol +
                               "'.");
    }
    intermediate_buffers_[symbol] = buf;
  };

  // Phase 1: allocate intermediates written by OUTPUT bindings (stage.*, _unused_*).
  for (size_t i = 0; i < model_entries_.size(); ++i) {
    for (const auto& [binding_name, symbol] : model_entries_[i].output_symbols) {
      if (!needs_alloc(symbol)) continue;
      if (intermediate_buffers_.count(symbol)) continue;
      alloc_for_output_binding(i, binding_name, symbol);
    }
  }

  // Phase 2: validate INPUT bindings - every intermediate must be produced by an earlier model, or
  // be noise.* (allocated explicitly in initialize()).
  for (size_t i = 0; i < model_entries_.size(); ++i) {
    for (const auto& [binding_name, symbol] : model_entries_[i].input_symbols) {
      if (!needs_alloc(symbol)) continue;
      if (symbol.starts_with("noise.")) continue;
      if (!intermediate_buffers_.count(symbol)) {
        throw std::runtime_error("Model file '" + model_entries_[i].path + "' needs input '" +
                                 binding_name +
                                 "', but no earlier model or application input provides it.");
      }
    }
  }

  // Phase 3: stable staging buffers for the captured graph. Operator inputs (input.*) and outputs
  // (output.*) arrive/leave on per-tick messages; the de-torched pre/post write/read these fixed
  // buffers, and the graph binds them by address. Always allocated (the fused op is graph-only).
  for (size_t i = 0; i < model_entries_.size(); ++i) {
    for (const auto& [binding_name, symbol] : model_entries_[i].input_symbols) {
      if (symbol.starts_with("input.") && !input_staging_.count(symbol)) {
        alloc_staging(i, binding_name, symbol, input_staging_);
      }
    }
    for (const auto& [binding_name, symbol] : model_entries_[i].output_symbols) {
      if (symbol.starts_with("output.") && !output_staging_.count(symbol)) {
        alloc_staging(i, binding_name, symbol, output_staging_);
      }
    }
  }
}

void Gr00tN17FusedOp::alloc_staging(size_t engine_idx, const std::string& binding_name,
                                    const std::string& symbol,
                                    std::unordered_map<std::string, IntermediateBuffer>& target) {
  auto& trt_engine = model_entries_[engine_idx].engine->engine();
  const std::string& path = model_entries_[engine_idx].path;
  const auto dims = trt_engine.getTensorShape(binding_name.c_str());
  size_t n_elems = 1;
  for (int d = 0; d < dims.nbDims; ++d) {
    if (dims.d[d] < 0) {
      throw std::runtime_error("Model file '" + path + "' uses a variable-sized binding '" +
                               binding_name + "'; CUDA-graph replay requires static shapes.");
    }
    n_elems *= static_cast<size_t>(dims.d[d]);
  }
  IntermediateBuffer buf;
  buf.bytes = n_elems * trt_elem_bytes(trt_engine.getTensorDataType(binding_name.c_str()));
  if (cudaMalloc(&buf.ptr, buf.bytes) != cudaSuccess) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' could not reserve GPU memory for model data '" + symbol + "'.");
  }
  target[symbol] = buf;
}

void* Gr00tN17FusedOp::resolve_symbol_fixed(const std::string& symbol) const {
  if (symbol.starts_with("input.")) {
    auto it = input_staging_.find(symbol);
    if (it == input_staging_.end()) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                               "' has no prepared input buffer for '" + symbol + "'.");
    }
    return it->second.ptr;
  }
  if (symbol.starts_with("output.")) {
    auto it = output_staging_.find(symbol);
    if (it == output_staging_.end()) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                               "' has no prepared output buffer for '" + symbol + "'.");
    }
    return it->second.ptr;
  }
  auto it = intermediate_buffers_.find(symbol);
  if (it == intermediate_buffers_.end()) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' has no prepared buffer for intermediate data '" + symbol + "'.");
  }
  return it->second.ptr;
}

void Gr00tN17FusedOp::run_inference_chain_fixed(cudaStream_t stream) {
  for (const auto& model : model_entries_) {
    auto& ctx = model.engine->context();
    for (const auto& [binding, sym] : model.input_symbols) {
      if (!ctx.setTensorAddress(binding.c_str(), resolve_symbol_fixed(sym))) {
        throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                                 "' could not prepare model input '" + binding + "' in file '" +
                                 model.path + "'.");
      }
    }
    for (const auto& [binding, sym] : model.output_symbols) {
      if (!ctx.setTensorAddress(binding.c_str(), resolve_symbol_fixed(sym))) {
        throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                                 "' could not prepare model output '" + binding + "' in file '" +
                                 model.path + "'.");
      }
    }
    if (!ctx.enqueueV3(stream)) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName + "' could not run model file '" +
                               model.path + "'. Check preceding TensorRT messages for details.");
    }
  }
}

void Gr00tN17FusedOp::sample_noise(void* dst, cudaStream_t stream) {
  // No per-tick reseed: the generator is seeded once in initialize(); each curandGenerateNormal
  // advances its sequence, producing fresh Gaussian noise without re-running curandGenerateSeeds
  // (which device-syncs). Called OUTSIDE the graph (curand is not capturable), stream-ordered
  // before the launch so action_head reads it from the fixed noise.initial buffer.
  if (curandSetStream(curand_gen_, stream) != CURAND_STATUS_SUCCESS) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' could not prepare GPU work for inference noise.");
  }
  const size_t n_floats = static_cast<size_t>(params().inference.max_action_horizon) *
                          static_cast<size_t>(params().inference.max_state_dim);
  if (curandGenerateNormal(curand_gen_, static_cast<float*>(dst), n_floats, /*mean=*/0.0f,
                           /*stddev=*/1.0f) != CURAND_STATUS_SUCCESS) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' could not generate inference noise.");
  }
}

void Gr00tN17FusedOp::start() {
  last_aligned_period_.reset();
  if (auto valid = validate_gr00t_n17_params(params()); !valid) {
    throw std::runtime_error(valid.error().message);
  }

  if (cudaEventCreateWithFlags(&completion_event_, cudaEventDisableTiming) != cudaSuccess) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' could not create the GPU completion event.");
  }

  load_engines();
  resolve_and_alloc_intermediates();

  // Noise buffer plus seed-once cuRAND warmup.
  if (!intermediate_buffers_.count("noise.initial")) {
    const size_t noise_bytes = static_cast<size_t>(params().inference.max_action_horizon) *
                               static_cast<size_t>(params().inference.max_state_dim) *
                               sizeof(float);
    IntermediateBuffer buf;
    buf.bytes = noise_bytes;
    if (cudaMalloc(&buf.ptr, noise_bytes) != cudaSuccess) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                               "' could not reserve GPU memory for inference noise.");
    }
    intermediate_buffers_["noise.initial"] = buf;
  }
  if (curandCreateGenerator(&curand_gen_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' could not initialize the inference random-number generator.");
  }
  // Seed EXACTLY ONCE. Reseeding per tick would re-run curandGenerateSeeds, a device-wide sync.
  uint64_t seed = params().inference.noise_seed;
  if (seed == 0) {
    seed = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
  }
  if (curandSetPseudoRandomGeneratorSeed(curand_gen_, seed) != CURAND_STATUS_SUCCESS) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' could not set the inference random seed.");
  }
  const size_t noise_floats = static_cast<size_t>(params().inference.max_action_horizon) *
                              static_cast<size_t>(params().inference.max_state_dim);
  if (noise_floats % 4 != 0) {
    throw std::runtime_error(
        "GR00T N1.7 '" + kOperatorName +
        "' has an unsupported inference-noise size. In the YAML configuration, make the product "
        "of \"max_action_horizon\" and \"max_state_dim\" divisible by four.");
  }
  // Warm up the generator now so its one-time curandGenerateSeeds (and device-wide sync) runs here,
  // off the runtime hot path. The noise buffer is scratch; it is overwritten on the first tick.
  if (curandGenerateNormal(curand_gen_,
                           static_cast<float*>(intermediate_buffers_.at("noise.initial").ptr),
                           noise_floats, 0.0f, 1.0f) != CURAND_STATUS_SUCCESS) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' could not prepare the inference random-number generator.");
  }

  // A private non-blocking stream carries initialization only. Runtime work uses the stream from
  // ExecutionContext.
  cudaStream_t init_stream{};
  if (cudaStreamCreateWithFlags(&init_stream, cudaStreamNonBlocking) != cudaSuccess) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' could not create a GPU work stream.");
  }
  StreamGuard init_guard{init_stream};

  n_sdk_joints_ = static_cast<int>(params().sdk_joint_order.size());
  if (n_sdk_joints_ <= 0) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' has no robot joints configured. Add joint names to the YAML "
                             "\"sdk_joint_order\" list.");
  }
  matx::make_tensor(t_joint_pos_raw_, {static_cast<matx::index_t>(n_sdk_joints_)},
                    matx::MATX_DEVICE_MEMORY, init_stream);

  // Validate the crop here; the raw-image scratch is sized from the FIRST frame's true resolution
  // and lazily allocated on the first tick (before any capture), so a camera larger than the crop
  // is handled - not just a crop-sized feed.
  const int crop_h = params().crop_height;
  const int crop_w = params().crop_width;
  if (crop_h <= 0 || crop_w <= 0) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' has an invalid camera crop. Set positive \"crop_height\" and "
                             "\"crop_width\" values in the YAML configuration.");
  }

  const auto nmot = static_cast<matx::index_t>(params().num_motors);
  const auto nhor = static_cast<matx::index_t>(params().prediction_horizon);
  if (nmot <= 0 || nhor <= 0) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' has an invalid action layout. Set positive \"num_motors\" and "
                             "\"prediction_horizon\" values in the YAML configuration.");
  }
  matx::make_tensor(t_pos_chunk_, {nmot, nhor}, matx::MATX_DEVICE_MEMORY, init_stream);
  matx::make_tensor(t_eff_chunk_, {nmot, nhor}, matx::MATX_DEVICE_MEMORY, init_stream);
  // Zero the chunks ONCE: each tick GR00T scatters the SAME disjoint rows (arms/hands/waist); the
  // rows it never writes (e.g. legs, filled downstream by the WBC combiner) stay 0 for the process
  // lifetime, keeping the graph body to scatter + D2H (no per-tick re-zero).
  require_cuda(cudaMemsetAsync(t_pos_chunk_.Data(), 0, sizeof(float) * nmot * nhor, init_stream),
               "Unable to initialize the GR00T position output");
  require_cuda(cudaMemsetAsync(t_eff_chunk_.Data(), 0, sizeof(float) * nmot * nhor, init_stream),
               "Unable to initialize the GR00T effort output");

  // Pinned (MATX_HOST_MEMORY) D2H targets: the graph's final nodes copy the chunks directly here.
  matx::make_tensor(t_pos_host_, {nmot, nhor}, matx::MATX_HOST_MEMORY);
  matx::make_tensor(t_eff_host_, {nmot, nhor}, matx::MATX_HOST_MEMORY);

  prepare_pre_and_post(init_stream);
  require_cuda(cudaStreamSynchronize(init_stream), "Unable to finish GR00T initialization");

  HOLOSCAN_LOG_INFO(
      "GR00T N1.7 '{}': ready with {} models, {} input joints, {} output motors, and a {}-step "
      "prediction horizon.",
      kOperatorName, model_entries_.size(), n_sdk_joints_, params().num_motors,
      params().prediction_horizon);
}

void Gr00tN17FusedOp::prepare_pre_and_post(cudaStream_t stream) {
  // SDK joint name -> position index.
  const auto& sdk = params().sdk_joint_order;
  std::unordered_map<std::string, int> sdk_index;
  sdk_index.reserve(sdk.size());
  for (int i = 0; i < static_cast<int>(sdk.size()); ++i) sdk_index[sdk[i]] = i;

  //  (a) Gather index tensors: per state_output_spec group, the SDK indices into the joint
  // position vector. Each gather writes the matching input.state.<group> staging buffer (built in
  // resolve_and_alloc_intermediates()). Cache the staging destination pointer so the captured graph
  // body does no map lookups.
  const std::string video_symbol = "input.video." + params().view_name;
  {
    auto it = input_staging_.find(video_symbol);
    if (it == input_staging_.end()) {
      throw std::runtime_error(
          "GR00T N1.7 '" + kOperatorName + "' has no prepared camera input for view '" +
          params().view_name +
          "'. In the YAML model list, make the camera input reference that view.");
    }
    const size_t want =
        static_cast<size_t>(params().crop_height) * params().crop_width * 3 * sizeof(float);
    if (it->second.bytes != want) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName + "' camera input for view '" +
                               params().view_name + "' has " + std::to_string(it->second.bytes) +
                               " bytes, but the configured crop requires " + std::to_string(want) +
                               ". Ensure the YAML crop dimensions match the ONNX model.");
    }
    video_staging_ptr_ = static_cast<float*>(it->second.ptr);
  }

  gathers_.clear();
  for (const auto& [group, joint_names] : params().state_output_spec) {
    std::vector<int> idx;
    idx.reserve(joint_names.size());
    for (const auto& joint_name : joint_names) {
      auto it = sdk_index.find(joint_name);
      if (it == sdk_index.end()) {
        throw std::runtime_error("GR00T N1.7 '" + kOperatorName + "' state group '" + group +
                                 "' names joint '" + joint_name +
                                 "', which is absent from the YAML \"sdk_joint_order\" list.");
      }
      idx.push_back(it->second);
    }
    const std::string sym = "input.state." + group;
    auto sit = input_staging_.find(sym);
    if (sit == input_staging_.end()) {
      throw std::runtime_error(
          "GR00T N1.7 '" + kOperatorName + "' has no prepared state input for group '" + group +
          "'. Ensure the YAML \"state_output_spec\" entry matches the ONNX model input.");
    }
    if (sit->second.bytes != idx.size() * sizeof(float)) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName + "' state group '" + group +
                               "' has " + std::to_string(idx.size()) +
                               " joints, but its ONNX model input "
                               "expects " +
                               std::to_string(sit->second.bytes / sizeof(float)) +
                               ". Update the YAML joint list to match the ONNX model.");
    }
    GatherEntry ge;
    matx::make_tensor(ge.idx, {static_cast<matx::index_t>(idx.size())}, matx::MATX_DEVICE_MEMORY,
                      stream);
    cudaMemcpyAsync(ge.idx.Data(), idx.data(), idx.size() * sizeof(int), cudaMemcpyHostToDevice,
                    stream);
    ge.dst = static_cast<float*>(sit->second.ptr);
    ge.n = static_cast<matx::index_t>(idx.size());
    gathers_.push_back(std::move(ge));
  }

  //  (b) action_indices_ + scatter index tensors: per input_spec group, the SDK rows into
  // joint_order.
  const auto& joint_order = params().joint_order;
  action_indices_.clear();
  for (const auto& [group, joint_names] : params().input_spec) {
    std::vector<int> idx;
    idx.reserve(joint_names.size());
    for (const auto& joint_name : joint_names) {
      auto it = std::find(joint_order.begin(), joint_order.end(), joint_name);
      if (it == joint_order.end()) {
        throw std::runtime_error("GR00T N1.7 '" + kOperatorName + "' action group '" + group +
                                 "' names joint '" + joint_name +
                                 "', which is absent from the YAML \"joint_order\" list.");
      }
      idx.push_back(static_cast<int>(it - joint_order.begin()));
    }
    action_indices_[group] = std::move(idx);
  }

  // Fill kp and kd host gains.
  kp_.assign(params().num_motors, 0.0f);
  kd_.assign(params().num_motors, 0.0f);
  auto fill_gains = [this](const std::map<std::string, std::vector<double>>& spec,
                           const std::unordered_map<std::string, std::vector<int>>& indices_by_key,
                           std::vector<float>& dst, const char* label) {
    for (const auto& [key, values] : spec) {
      auto it = indices_by_key.find(key);
      if (it == indices_by_key.end()) continue;
      const auto& group_indices = it->second;
      if (values.size() == group_indices.size()) {
        for (size_t i = 0; i < values.size(); ++i) {
          dst[group_indices[i]] = static_cast<float>(values[i]);
        }
      } else if (values.size() == 1) {
        for (int idx : group_indices) dst[idx] = static_cast<float>(values[0]);
      } else {
        throw std::runtime_error(
            "GR00T N1.7 '" + kOperatorName + "': " + label + " gains for action group '" + key +
            "' have " + std::to_string(values.size()) + " entries, but the group has " +
            std::to_string(group_indices.size()) +
            " joints. Supply one gain or one gain per joint in the YAML configuration.");
      }
    }
  };
  fill_gains(params().kp, action_indices_, kp_, "kp");
  fill_gains(params().kd, action_indices_, kd_, "kd");
  // Build the scatter work list from the output staging buffers: an "output.<group>" whose group is
  // a known action group scatters into the position chunk; an "output.effort_<group>" scatters into
  // the effort chunk (strip "effort_" and reuse the same group indices, like the postprocessor).
  scatters_.clear();
  for (const auto& [symbol, buf] : output_staging_) {
    const std::string key = symbol.substr(std::strlen("output."));  // strip "output."
    std::string group;
    bool is_effort;
    if (action_indices_.count(key)) {
      group = key;
      is_effort = false;
    } else if (key.rfind("effort_", 0) == 0 &&
               action_indices_.count(key.substr(std::strlen("effort_")))) {
      group = key.substr(std::strlen("effort_"));
      is_effort = true;
    } else {
      continue;  // unmapped output (e.g. navigate/base-height are routed to _unused_, not output.*)
    }
    const auto& rows = action_indices_.at(group);
    const size_t want =
        static_cast<size_t>(params().prediction_horizon) * rows.size() * sizeof(float);
    if (buf.bytes != want) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName + "' action output for group '" +
                               group + "' has " + std::to_string(buf.bytes) +
                               " bytes, but the configured "
                               "prediction horizon and group size require " +
                               std::to_string(want) +
                               ". Ensure the YAML action configuration matches the ONNX model.");
    }
    ScatterEntry se;
    matx::make_tensor(se.idx, {static_cast<matx::index_t>(rows.size())}, matx::MATX_DEVICE_MEMORY,
                      stream);
    cudaMemcpyAsync(se.idx.Data(), rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice,
                    stream);
    se.src = static_cast<float*>(buf.ptr);
    se.is_effort = is_effort;
    se.horizon = static_cast<matx::index_t>(params().prediction_horizon);
    se.group_size = static_cast<matx::index_t>(rows.size());
    scatters_.push_back(std::move(se));
  }
}

void Gr00tN17FusedOp::run_fused_chain_fixed(cudaStream_t stream) {
  matx::cudaExecutor exec{stream};

  //  De-torched preprocess
  // Video: crop + drop-alpha + cast-to-float32. image_scratch_ (uint8) -> input.video.<view>
  // staging.
  {
    const int crop_h = params().crop_height;
    const int crop_w = params().crop_width;
    const dim3 blk(16, 16);
    const dim3 grd((crop_w + 15) / 16, (crop_h + 15) / 16);
    image_prep_kernel<uint8_t><<<grd, blk, 0, stream>>>(static_cast<const uint8_t*>(image_scratch_),
                                                        video_staging_ptr_, img_src_w_, img_src_c_,
                                                        crop_h, crop_w);
  }
  // State: per-group gather positions[idx] -> input.state.<group> staging.
  for (const auto& g : gathers_) {
    auto dst = matx::make_tensor(g.dst, {g.n});  // non-owning view over the staging buffer
    (dst = matx::select(t_joint_pos_raw_, g.idx)).run(exec);
  }

  //  5-engine TRT chain (fixed staging/intermediate addresses)
  run_inference_chain_fixed(stream);

  //  De-torched postprocess
  // Scatter each decode_action group into its [num_motors, horizon] chunk (transposed remap):
  // chunk[idx[j], h] = group[h, j].
  for (const auto& sc : scatters_) {
    auto group = matx::make_tensor(sc.src, {sc.horizon, sc.group_size});  // [horizon, J] view
    auto& chunk = sc.is_effort ? t_eff_chunk_ : t_pos_chunk_;
    (matx::remap<0>(chunk, sc.idx) = group.Permute({1, 0})).run(exec);
  }
  // D2H the position + effort chunks into the pinned host buffers (fixed addresses).
  const size_t chunk_bytes =
      static_cast<size_t>(params().num_motors) * params().prediction_horizon * sizeof(float);
  cudaMemcpyAsync(t_pos_host_.Data(), t_pos_chunk_.Data(), chunk_bytes, cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(t_eff_host_.Data(), t_eff_chunk_.Data(), chunk_bytes, cudaMemcpyDeviceToHost,
                  stream);
}

holoscan::expected<void, holoscan::Error> Gr00tN17FusedOp::compute(
    holoscan::ExecutionContext& context) {
  if (joint_state.is_missing() || image.is_missing()) {
    return {};
  }
  auto received_state = joint_state.receive();
  if (!received_state) {
    return holoscan::make_unexpected(std::move(received_state).error());
  }
  auto received_image = image.receive();
  if (!received_image) {
    return holoscan::make_unexpected(std::move(received_image).error());
  }
  const isaac::JointStateCpuT& state = received_state->data;
  const holoscan::Tensor& input_image = received_image->data;
  const auto image_capture_time = received_image->metadata.capture_time();
  if (!image_capture_time.valid() || image_capture_time.timestamp_ns < 0) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument,
        "The camera frame has no valid capture time; inference cannot select its interval."});
  }
  if (params().align_observations) {
    const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::duration<double>(params().inference_period))
                               .count();
    if (!select_gr00t_n17_frame(image_capture_time.timestamp_ns, period_ns, last_aligned_period_,
                                true)) {
      return {};
    }
  }
  if (auto valid = validate_gr00t_n17_inputs(params(), state, input_image); !valid) {
    return holoscan::make_unexpected(std::move(valid).error());
  }

  int64_t img_h = 0;
  int64_t img_w = 0;
  int64_t img_c = 0;
  (void)extract_hwc(input_image, img_h, img_w, img_c);
  const size_t img_bytes = static_cast<size_t>(img_h) * img_w * img_c;
  // src_w / src_c are baked into the captured graph's image_prep launch, so they must be stable;
  // the scratch is sized from this first frame's true resolution and allocated once, before any
  // capture.
  if (!img_dims_known_) {
    img_src_w_ = static_cast<int>(img_w);
    img_src_c_ = static_cast<int>(img_c);
    image_scratch_bytes_ = img_bytes;
    if (cudaMalloc(&image_scratch_, image_scratch_bytes_) != cudaSuccess) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                               "' could not reserve GPU memory for the camera frame.");
    }
    img_dims_known_ = true;
  } else if (img_src_w_ != static_cast<int>(img_w) || img_src_c_ != static_cast<int>(img_c) ||
             img_bytes != image_scratch_bytes_) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' received a camera frame whose dimensions changed during the run. "
                             "Keep the camera format constant.");
  }

  cudaStream_t stream = context.cuda_stream().get();

  //  OUTSIDE the graph (stream-ordered before the launch)
  // Stage the raw joint position and camera frame into the fixed scratch buffers, and sample fresh
  // noise. cudaMemcpyDefault lets the message data live on host OR device (mock vs real feeds).
  if (cudaMemcpyAsync(t_joint_pos_raw_.Data(), state.position.data(),
                      static_cast<size_t>(n_sdk_joints_) * sizeof(float), cudaMemcpyDefault,
                      stream) != cudaSuccess) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' could not copy the joint positions to the GPU.");
  }
  if (cudaMemcpyAsync(image_scratch_, input_image.data(), img_bytes, cudaMemcpyDefault, stream) !=
      cudaSuccess) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' could not copy the camera frame to the GPU.");
  }
  sample_noise(intermediate_buffers_.at("noise.initial").ptr, stream);

  //  Warmup -> capture -> replay (GR00T needs one eager warmup to flush TRT's non-capturable
  // first-enqueue phase).
  if (!graph_warmup_done_) {
    run_fused_chain_fixed(stream);
    graph_warmup_done_ = true;
  } else if (!graph_captured_) {
    // Drain the pre-capture staging (H2D + noise) so the stream is idle before capture begins - the
    // fused chain alone gets recorded.
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                               "' could not prepare GPU work for acceleration.");
    }
    // Capture on a private stream in the same CUDA context to exclude tensor-retirement events.
    // Replay uses the SDK stream for input/output ordering.
    StreamGuard capture{make_capture_stream(stream)};
    if (cudaStreamBeginCapture(capture.s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                               "' could not start GPU acceleration setup.");
    }
    try {
      run_fused_chain_fixed(capture.s);
    } catch (...) {
      // Close the open capture before propagating so the stream isn't left in capture mode.
      cudaGraph_t partial = nullptr;
      cudaStreamEndCapture(capture.s, &partial);
      if (partial != nullptr) {
        cudaGraphDestroy(partial);
      }
      throw;
    }
    if (cudaStreamEndCapture(capture.s, &graph_) != cudaSuccess) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                               "' could not finish GPU acceleration setup.");
    }
    if (cudaGraphInstantiate(&graph_exec_, graph_, 0) != cudaSuccess) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                               "' could not prepare accelerated inference.");
    }
    graph_captured_ = true;
    HOLOSCAN_LOG_INFO("GR00T N1.7 '{}': accelerated inference is ready.", kOperatorName);
    if (cudaGraphLaunch(graph_exec_, stream) != cudaSuccess) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                               "' could not run accelerated inference.");
    }
  } else {
    if (cudaGraphLaunch(graph_exec_, stream) != cudaSuccess) {
      throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                               "' could not run accelerated inference.");
    }
  }

  // Block until the whole chain (including the pinned D2H) completes; GR00T's long compute favors a
  // blocking wait over a busy-spin (no CPU core burned spinning).
  if (cudaEventRecord(completion_event_, stream) != cudaSuccess) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' could not track inference completion.");
  }
  if (cudaEventSynchronize(completion_event_) != cudaSuccess) {
    throw std::runtime_error("GR00T N1.7 '" + kOperatorName +
                             "' did not complete inference successfully.");
  }

  const std::size_t value_count = static_cast<std::size_t>(params().num_motors) *
                                  static_cast<std::size_t>(params().prediction_horizon);
  // Normal sample metadata remains tied to joint-state benchmark lineage. Carry image time in the
  // message so action selection can account for inference latency independently.
  auto chunk = make_joint_command_chunk_cpu(
      std::span<const float>{t_pos_host_.Data(), value_count},
      std::span<const float>{t_eff_host_.Data(), value_count}, kp_, kd_,
      static_cast<std::uint32_t>(params().num_motors),
      static_cast<std::uint32_t>(params().prediction_horizon), image_capture_time.timestamp_ns);
  if (!chunk) {
    return holoscan::make_unexpected(std::move(chunk).error());
  }

  holoscan::EmitOptions options{.flags = received_state->metadata.flags};
  const holoscan::TimePoint capture_time = received_state->metadata.capture_time();
  if (capture_time.valid()) {
    options.capture_time = capture_time;
  } else if (auto now = context.clock().now()) {
    options.capture_time = *now;
  }
  return action_chunk.emit(std::move(*chunk), options);
}

void Gr00tN17FusedOp::free_all_device_buffers() {
  for (auto* map : {&intermediate_buffers_, &input_staging_, &output_staging_}) {
    for (auto& [sym, buf] : *map) {
      if (buf.ptr) {
        cudaFree(buf.ptr);
        buf.ptr = nullptr;
      }
    }
    map->clear();
  }
}

void Gr00tN17FusedOp::stop() {
  if (completion_event_) {
    cudaEventDestroy(completion_event_);
    completion_event_ = nullptr;
  }
  if (graph_exec_) {
    cudaGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  if (graph_) {
    cudaGraphDestroy(graph_);
    graph_ = nullptr;
  }
  if (curand_gen_) {
    curandDestroyGenerator(curand_gen_);
    curand_gen_ = nullptr;
  }
  // Tear down the TRT contexts before freeing device buffers so contexts release while the device
  // is still accessible. matx tensors free themselves via RAII after this body.
  model_entries_.clear();
  free_all_device_buffers();
  if (image_scratch_) {
    cudaFree(image_scratch_);
    image_scratch_ = nullptr;
  }
  image_scratch_bytes_ = 0U;
  img_dims_known_ = false;
  graph_warmup_done_ = false;
  graph_captured_ = false;
}

Gr00tN17FusedOp::~Gr00tN17FusedOp() { stop(); }

}  // namespace isaac::os::ops
