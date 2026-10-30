/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_JOINT_COMMAND_COMBINER_JOINT_COMMAND_COMBINER_OP_H
#define ISAAC_OS_OPERATORS_JOINT_COMMAND_COMBINER_JOINT_COMMAND_COMBINER_OP_H

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "joint_command_cpu_schema_traits.h"

namespace isaac::os::ops {

struct JointCommandCombinerParams {
  std::vector<std::string> joint_order;
  std::vector<std::string> joint_names_from_primary;
  std::vector<std::string> joint_names_from_overlay;
  double overlay_startup_delay_s{5.0};
};

HOLOSCAN_PARAM_SCHEMA(JointCommandCombinerParams,
                      holoscan::param_schema("com.nvidia.isaac.JointCommandCombinerParams", 1),
                      holoscan::field<&JointCommandCombinerParams::joint_order>("joint_order"),
                      holoscan::field<&JointCommandCombinerParams::joint_names_from_primary>(
                          "joint_names_from_primary"),
                      holoscan::field<&JointCommandCombinerParams::joint_names_from_overlay>(
                          "joint_names_from_overlay"),
                      holoscan::field<&JointCommandCombinerParams::overlay_startup_delay_s>(
                          "overlay_startup_delay_s"));

/// Replaces selected joints in each primary command with values from the latest overlay command.
/// Marks the primary-only fallback degraded until the configured overlay is applied.
class JointCommandCombinerOp final : public holoscan::Operator<JointCommandCombinerParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::JointCommandCpuT> primary_command;
  holoscan::Input<isaac::JointCommandCpuT> overlay_command;
  holoscan::Output<isaac::JointCommandCpuT> command;

 private:
  [[nodiscard]] bool command_size_is_valid(const isaac::JointCommandCpuT& command) const noexcept;

  std::vector<bool> take_from_overlay_;
  isaac::JointCommandCpuT cached_overlay_command_;
  holoscan::SampleFlags cached_overlay_flags_{holoscan::SampleFlags::kNone};
  bool overlay_seen_{};
  std::chrono::steady_clock::time_point first_tick_time_{};
  bool first_tick_seen_{};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_JOINT_COMMAND_COMBINER_JOINT_COMMAND_COMBINER_OP_H
