/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_GR00T_ACTION_HANDLER_GR00T_ACTION_HANDLER_OP_H
#define ISAAC_OS_OPERATORS_GR00T_ACTION_HANDLER_GR00T_ACTION_HANDLER_OP_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "common/joint_command_chunk_cpu/joint_command_chunk_cpu.h"
#include "joint_command_chunk_cpu_schema_traits.h"
#include "joint_command_cpu_schema_traits.h"

namespace isaac::os::ops {

struct Gr00tActionHandlerParams {
  double action_interval_s{0.0333333};
  double handover_blend_steps{0.0};
  bool feedforward_torque{true};
};

HOLOSCAN_PARAM_SCHEMA(
    Gr00tActionHandlerParams,
    holoscan::param_schema("com.nvidia.isaac.Gr00tActionHandlerParams", 1),
    holoscan::field<&Gr00tActionHandlerParams::action_interval_s>("action_interval_s"),
    holoscan::field<&Gr00tActionHandlerParams::handover_blend_steps>("handover_blend_steps"),
    holoscan::field<&Gr00tActionHandlerParams::feedforward_torque>("feedforward_torque"));

/// Selects the GR00T action due on each control command and forwards that command unchanged.
class Gr00tActionHandlerOp final : public holoscan::Operator<Gr00tActionHandlerParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::JointCommandCpuT> control_command;
  holoscan::Input<isaac::JointCommandChunkCpuT> action_chunk;
  holoscan::Output<isaac::JointCommandCpuT> action_command;
  holoscan::Output<isaac::JointCommandCpuT> forwarded_control_command;

 private:
  isaac::JointCommandChunkCpuT latest_chunk_;
  std::int64_t action_interval_ns_{};
  std::int64_t latest_image_time_ns_{};
  std::int64_t last_command_time_ns_{};
  std::vector<float> last_position_;
  std::vector<float> handover_offset_;
  double handover_start_step_{};
  bool started_{};
  bool warned_missing_image_time_{};
  bool warned_future_image_time_{};
  bool warned_expired_{};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_GR00T_ACTION_HANDLER_GR00T_ACTION_HANDLER_OP_H
