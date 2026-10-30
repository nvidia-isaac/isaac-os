/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_GR00T_N1_7_GR00T_N1_7_FUSED_OP_H
#define ISAAC_OS_OPERATORS_GR00T_N1_7_GR00T_N1_7_FUSED_OP_H

// MatX 0.9.4 assumes these tuple declarations are transitively visible.
// clang-format off
#include <thrust/tuple.h>
#include <matx.h>
// clang-format on

#include <curand.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "common/tensorrt_engine/tensorrt_engine.h"
#include "joint_command_chunk_cpu_schema_traits.h"
#include "joint_state_cpu_schema_traits.h"

namespace isaac::os::ops {

struct Gr00tN17Model {
  std::string path;
  std::map<std::string, std::string> inputs;
  std::map<std::string, std::string> outputs;
};

struct Gr00tN17InferenceParams {
  int num_joints{};
  int chunk_size{16};
  int max_action_horizon{40};
  int max_state_dim{132};
  std::uint64_t noise_seed{};
  std::vector<Gr00tN17Model> models;
};

HOLOSCAN_PARAM_SCHEMA(
    Gr00tN17InferenceParams, holoscan::param_schema("com.nvidia.isaac.Gr00tN17InferenceParams", 1),
    holoscan::field<&Gr00tN17InferenceParams::num_joints>("num_joints"),
    holoscan::field<&Gr00tN17InferenceParams::chunk_size>("chunk_size"),
    holoscan::field<&Gr00tN17InferenceParams::max_action_horizon>("max_action_horizon"),
    holoscan::field<&Gr00tN17InferenceParams::max_state_dim>("max_state_dim"),
    holoscan::field<&Gr00tN17InferenceParams::noise_seed>("noise_seed"),
    holoscan::field<&Gr00tN17InferenceParams::models>("models"));

struct Gr00tN17Params {
  std::vector<std::string> sdk_joint_order;
  std::map<std::string, std::vector<std::string>> state_output_spec;
  std::string view_name;
  int crop_height{};
  int crop_width{};
  Gr00tN17InferenceParams inference;
  int num_motors{43};
  int prediction_horizon{16};
  std::vector<std::string> joint_order;
  std::map<std::string, std::vector<std::string>> input_spec;
  std::map<std::string, std::vector<double>> kp;
  std::map<std::string, std::vector<double>> kd;
  // Select at most one observation per fixed capture-time interval.
  double inference_period{0.25};
  // For sensors sharing capture timestamps: infer from exact frame/state pairs at this cadence.
  bool align_observations{false};
};

HOLOSCAN_PARAM_SCHEMA(
    Gr00tN17Params, holoscan::param_schema("com.nvidia.isaac.Gr00tN17Params", 2),
    holoscan::field<&Gr00tN17Params::sdk_joint_order>("sdk_joint_order"),
    holoscan::field<&Gr00tN17Params::state_output_spec>("state_output_spec"),
    holoscan::field<&Gr00tN17Params::view_name>("view_name"),
    holoscan::field<&Gr00tN17Params::crop_height>("crop_height"),
    holoscan::field<&Gr00tN17Params::crop_width>("crop_width"),
    holoscan::field<&Gr00tN17Params::inference>("inference").nested<Gr00tN17InferenceParams>(),
    holoscan::field<&Gr00tN17Params::num_motors>("num_motors"),
    holoscan::field<&Gr00tN17Params::prediction_horizon>("prediction_horizon"),
    holoscan::field<&Gr00tN17Params::joint_order>("joint_order"),
    holoscan::field<&Gr00tN17Params::input_spec>("input_spec"),
    holoscan::field<&Gr00tN17Params::kp>("kp"), holoscan::field<&Gr00tN17Params::kd>("kd"),
    holoscan::field<&Gr00tN17Params::inference_period>("inference_period"),
    holoscan::field<&Gr00tN17Params::align_observations>("align_observations"));

/// Select the first available frame in each absolute capture-time interval, skipping missed ones.
/// Requires a nonnegative capture time and a positive period, both in nanoseconds.
/// Aligned simulation observations defer the initial interval; real/mock accept it immediately.
[[nodiscard]] bool select_gr00t_n17_frame(std::int64_t capture_time_ns, std::int64_t period_ns,
                                          std::optional<std::int64_t>& last_period,
                                          bool defer_initial_interval);

[[nodiscard]] holoscan::expected<void, holoscan::Error> validate_gr00t_n17_params(
    const Gr00tN17Params& params);

[[nodiscard]] holoscan::expected<void, holoscan::Error> validate_gr00t_n17_inputs(
    const Gr00tN17Params& params, const isaac::JointStateCpuT& state,
    const holoscan::Tensor& image);

[[nodiscard]] holoscan::expected<isaac::JointCommandChunkCpuT, holoscan::Error>
make_joint_command_chunk_cpu(std::span<const float> position, std::span<const float> effort,
                             std::span<const float> kp, std::span<const float> kd,
                             std::uint32_t joint_count, std::uint32_t action_count,
                             std::int64_t image_capture_time_ns);

/// Fuses GR00T N1.7 preprocessing, five TensorRT engines, and postprocessing into one CUDA graph.
class Gr00tN17FusedOp final : public holoscan::Operator<Gr00tN17Params> {
 public:
  ~Gr00tN17FusedOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::JointStateCpuT> joint_state;
  holoscan::Input<holoscan::Tensor> image;
  holoscan::Output<isaac::JointCommandChunkCpuT> action_chunk;

 private:
  bool select_inference_frame(const holoscan::PredicateEvaluationContext&,
                              const holoscan::PredicateInputView<holoscan::Tensor>& frame,
                              std::optional<std::int64_t>& last_period) const noexcept;
  std::optional<std::int64_t> last_aligned_period_;
  struct IntermediateBuffer {
    void* ptr{};
    std::size_t bytes{};
  };

  struct ModelEntry {
    std::string path;
    std::unique_ptr<isaac::TensorRTEngine> engine;
    std::unordered_map<std::string, std::string> input_symbols;
    std::unordered_map<std::string, std::string> output_symbols;
  };

  struct GatherEntry {
    matx::tensor_t<int, 1> idx;
    float* dst{};
    matx::index_t n{};
  };

  struct ScatterEntry {
    matx::tensor_t<int, 1> idx;
    float* src{};
    bool is_effort{};
    matx::index_t horizon{};
    matx::index_t group_size{};
  };

  void load_engines();
  void resolve_and_alloc_intermediates();
  void alloc_staging(std::size_t engine_index, const std::string& binding_name,
                     const std::string& symbol,
                     std::unordered_map<std::string, IntermediateBuffer>& target);
  [[nodiscard]] void* resolve_symbol_fixed(const std::string& symbol) const;
  void run_inference_chain_fixed(cudaStream_t stream);
  void sample_noise(void* destination, cudaStream_t stream);
  void prepare_pre_and_post(cudaStream_t stream);
  void run_fused_chain_fixed(cudaStream_t stream);
  void free_all_device_buffers();

  std::vector<ModelEntry> model_entries_;
  std::unordered_map<std::string, IntermediateBuffer> intermediate_buffers_;
  std::unordered_map<std::string, IntermediateBuffer> input_staging_;
  std::unordered_map<std::string, IntermediateBuffer> output_staging_;

  curandGenerator_t curand_gen_{};
  cudaGraph_t graph_{};
  cudaGraphExec_t graph_exec_{};
  cudaEvent_t completion_event_{};
  bool graph_warmup_done_{};
  bool graph_captured_{};

  int n_sdk_joints_{};
  matx::tensor_t<float, 1> t_joint_pos_raw_;
  void* image_scratch_{};
  std::size_t image_scratch_bytes_{};
  float* video_staging_ptr_{};
  int img_src_w_{};
  int img_src_c_{};
  bool img_dims_known_{};
  std::vector<GatherEntry> gathers_;

  std::unordered_map<std::string, std::vector<int>> action_indices_;
  std::vector<ScatterEntry> scatters_;
  std::vector<float> kp_;
  std::vector<float> kd_;
  matx::tensor_t<float, 2> t_pos_chunk_;
  matx::tensor_t<float, 2> t_eff_chunk_;
  matx::tensor_t<float, 2> t_pos_host_;
  matx::tensor_t<float, 2> t_eff_host_;
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_GR00T_N1_7_GR00T_N1_7_FUSED_OP_H
