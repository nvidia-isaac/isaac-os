/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_SIMPLE_MOTION_PLANNER_SIMPLE_MOTION_PLANNER_H
#define ISAAC_OS_OPERATORS_SIMPLE_MOTION_PLANNER_SIMPLE_MOTION_PLANNER_H

#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "joint_command_cpu_schema_traits.h"
#include "state_sample_cpu_schema_traits.h"

namespace isaac::os::ops {

/// Bounds `SimpleMotionPlannerOp` applies to each joint command.
struct SimpleMotionPlannerParams {
  /// Largest allowed deviation from the measured position on any one tick, in radians.
  double max_step_size{0.20};
  /// Joint count. State samples of any other size are ignored rather than partially commanded.
  int num_motors{6};
};

HOLOSCAN_PARAM_SCHEMA(SimpleMotionPlannerParams,
                      holoscan::param_schema("com.nvidia.isaac.SimpleMotionPlannerParams", 1),
                      holoscan::field<&SimpleMotionPlannerParams::max_step_size>("max_step_size"),
                      holoscan::field<&SimpleMotionPlannerParams::num_motors>("num_motors"));

/// Reports whether the motor count is positive and the step size is finite and greater than zero.
[[nodiscard]] bool validate_simple_motion_planner_params(const SimpleMotionPlannerParams& params);

/// Clamps each joint command to the measured position plus or minus max_step_size.
///
/// The planner emits one command per measured state sample. Between target updates it
/// re-clamps the most recent target against fresh state, so the command rate follows the
/// joint-state rate rather than the target rate.
class SimpleMotionPlannerOp final : public holoscan::Operator<SimpleMotionPlannerParams> {
 public:
  /// Declares the target and joint-state inputs and the clamped joint-command output.
  void setup(holoscan::OperatorSpec& spec) override;
  /// Triggers on each joint-state sample and accepts a target up to 40 ms old, substituting a
  /// safe default when none has arrived. This is what lets commands continue at the state rate
  /// while the target is stale.
  [[nodiscard]] holoscan::Contract contract() const override;
  /// Throws `std::runtime_error` if the parameters are invalid, and drops any retained target.
  void start() override;
  /// Clamps the retained target to the measured position plus or minus `max_step_size` and
  /// emits one command per state sample. State samples with the wrong joint count or a
  /// non-finite position are dropped, and nothing is emitted until a valid target arrives.
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  /// Desired joint positions. Retained across ticks and re-clamped against fresh state, so the
  /// target may update more slowly than the command rate.
  holoscan::Input<isaac::JointCommandCpuT> target;
  /// Measured joint state, which both paces the operator and bounds the command.
  holoscan::Input<isaac::StateSampleCpuT> state;
  /// Joint command, clamped to within one step of the measured position.
  holoscan::Output<isaac::JointCommandCpuT> command;

 private:
  std::vector<float> target_position_;
  bool has_target_{};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_SIMPLE_MOTION_PLANNER_SIMPLE_MOTION_PLANNER_H
