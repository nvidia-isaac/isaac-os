/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_SO101_REACH_SO101_REACH_FUSED_OP_H
#define ISAAC_OS_OPERATORS_SO101_REACH_SO101_REACH_FUSED_OP_H

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/cuda_execution/cuda_stream_pool.hpp>

#include "common/tensorrt_engine/tensorrt_engine.h"
#include "policy_delta_cpu_schema_traits.h"
#include "state_sample_cpu_schema_traits.h"

namespace isaac::os::ops {

struct So101ReachParams {
  std::string onnx_path;
  std::vector<double> default_joint_position{0.0, 0.0, 0.0, 1.5700000524520874, 0.0, 0.0};
  std::vector<double> goal_pos_x_range{0.2, 0.2};
  std::vector<double> goal_pos_y_range{-0.112, 0.112};
  std::vector<double> goal_pos_z_range{0.136, 0.264};
  std::int64_t goal_grid_nx{1};
  std::int64_t goal_grid_ny{3};
  std::int64_t goal_grid_nz{3};
  double goal_hold_time{10.0};
  double goal_settle_time{1.0};
  double reach_threshold{0.05};
  double action_clip{1.0};
  double inference_period{0.02};
};

/// Returns whether the fixed SO-101 reach policy contract is configured safely.
[[nodiscard]] bool validate_so101_reach_params(const So101ReachParams& params);

/// Builds the uniformly spaced reach goals in Z/Y/X iteration order.
[[nodiscard]] std::vector<std::array<float, 3U>> build_so101_reach_goal_grid(
    const So101ReachParams& params);

/// Builds the fixed 20-value observation used to train the SO-101 reach policy.
[[nodiscard]] std::array<float, 20U> build_so101_reach_observation(
    const std::array<float, 6U>& joint_position, const So101ReachParams& params,
    const std::array<float, 3U>& goal_position, const std::array<float, 4U>& last_action);

/// Clips a finite SO-101 policy action, or returns a zero action if any value is invalid.
[[nodiscard]] std::array<float, 4U> sanitize_so101_reach_action(const std::array<float, 4U>& action,
                                                                float action_clip);

HOLOSCAN_PARAM_SCHEMA(
    So101ReachParams, holoscan::param_schema("com.nvidia.isaac.So101ReachParams", 1),
    holoscan::field<&So101ReachParams::onnx_path>("onnx_path"),
    holoscan::field<&So101ReachParams::default_joint_position>("default_joint_position"),
    holoscan::field<&So101ReachParams::goal_pos_x_range>("goal_pos_x_range"),
    holoscan::field<&So101ReachParams::goal_pos_y_range>("goal_pos_y_range"),
    holoscan::field<&So101ReachParams::goal_pos_z_range>("goal_pos_z_range"),
    holoscan::field<&So101ReachParams::goal_grid_nx>("goal_grid_nx"),
    holoscan::field<&So101ReachParams::goal_grid_ny>("goal_grid_ny"),
    holoscan::field<&So101ReachParams::goal_grid_nz>("goal_grid_nz"),
    holoscan::field<&So101ReachParams::goal_hold_time>("goal_hold_time"),
    holoscan::field<&So101ReachParams::goal_settle_time>("goal_settle_time"),
    holoscan::field<&So101ReachParams::reach_threshold>("reach_threshold"),
    holoscan::field<&So101ReachParams::action_clip>("action_clip"),
    holoscan::field<&So101ReachParams::inference_period>("inference_period"));

/// Runs the fixed SO-101 20-D reach observation contract through TensorRT at 50 Hz.
class So101ReachFusedOp final : public holoscan::Operator<So101ReachParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::StateSampleCpuT> state;
  holoscan::Output<isaac::PolicyDeltaCpuT> delta;

 private:
  using Clock = std::chrono::steady_clock;

  void advance_goal(std::string_view reason);
  [[nodiscard]] const std::array<float, 3U>& update_goal(
      const std::array<float, 3U>& fingertip_position);

  std::unique_ptr<isaac::TensorRTEngine> tensorrt_engine_;
  std::unique_ptr<holoscan::CudaStreamPool> cuda_stream_pool_;
  holoscan::CudaStreamPool::StreamLease cuda_stream_;
  float* observation_device_{};
  float* actions_device_{};
  std::array<float, 20> observation_host_{};
  std::array<float, 4> actions_host_{};
  std::array<float, 4> last_action_{};
  std::vector<std::array<float, 3U>> goals_;
  std::size_t active_goal_index_{};
  int goals_reached_{};
  bool goal_settling_{};
  Clock::time_point goal_start_time_{};
  Clock::time_point settle_start_time_{};
  Clock::time_point last_distance_log_{};
  std::uint64_t inferences_{};
  Clock::time_point first_inference_time_{};
  Clock::time_point last_inference_time_{};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_SO101_REACH_SO101_REACH_FUSED_OP_H
