/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_SO101_SO101_DELTA_MOTION_CONTROL_OP_H
#define ISAAC_OS_OPERATORS_SO101_SO101_DELTA_MOTION_CONTROL_OP_H

#include <array>
#include <cstdint>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "joint_command_cpu_schema_traits.h"
#include "policy_delta_cpu_schema_traits.h"
#include "state_sample_cpu_schema_traits.h"

namespace isaac::os::ops {

struct So101DeltaMotionParams {
  double action_max_step_size{0.1};
  double action_scale{0.2};
  std::vector<double> fixed_joint_positions{0.0, 0.0};
  std::vector<double> joint_pos_limits_lower;
  std::vector<double> joint_pos_limits_upper;
  double motion_max_step_size{0.05};
  double target_expiry{0.04};
};

HOLOSCAN_PARAM_SCHEMA(
    So101DeltaMotionParams, holoscan::param_schema("com.nvidia.isaac.So101DeltaMotionParams", 1),
    holoscan::field<&So101DeltaMotionParams::action_max_step_size>("action_max_step_size"),
    holoscan::field<&So101DeltaMotionParams::action_scale>("action_scale"),
    holoscan::field<&So101DeltaMotionParams::fixed_joint_positions>("fixed_joint_positions"),
    holoscan::field<&So101DeltaMotionParams::joint_pos_limits_lower>("joint_pos_limits_lower"),
    holoscan::field<&So101DeltaMotionParams::joint_pos_limits_upper>("joint_pos_limits_upper"),
    holoscan::field<&So101DeltaMotionParams::motion_max_step_size>("motion_max_step_size"),
    holoscan::field<&So101DeltaMotionParams::target_expiry>("target_expiry"));

[[nodiscard]] bool validate_so101_delta_motion_params(const So101DeltaMotionParams& params);

/// Converts 50 Hz policy deltas into 100 Hz, measured-state-bounded joint commands.
class So101DeltaMotionControlOp final : public holoscan::Operator<So101DeltaMotionParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::PolicyDeltaCpuT> delta;
  holoscan::Input<isaac::StateSampleCpuT> state;
  holoscan::Output<isaac::JointCommandCpuT> command;

 private:
  std::array<float, 6U> target_{};
  std::int64_t target_time_ns_{};
  std::uint64_t target_fault_epoch_{};
  std::uint64_t target_sequence_{};
  bool has_accepted_delta_{};
  bool has_target_{};
  std::uint64_t states_{};
  std::uint64_t commands_{};
  std::uint64_t expired_targets_{};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_SO101_SO101_DELTA_MOTION_CONTROL_OP_H
