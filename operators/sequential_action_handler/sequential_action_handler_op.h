/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_SEQUENTIAL_ACTION_HANDLER_SEQUENTIAL_ACTION_HANDLER_OP_H
#define ISAAC_OS_OPERATORS_SEQUENTIAL_ACTION_HANDLER_SEQUENTIAL_ACTION_HANDLER_OP_H

#include <cstddef>
#include <cstdint>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "common/joint_command_chunk_cpu/joint_command_chunk_cpu.h"
#include "joint_command_chunk_cpu_schema_traits.h"
#include "joint_command_cpu_schema_traits.h"

namespace isaac::os::ops {

struct SequentialActionHandlerParams {
  double dt{0.1};
};

HOLOSCAN_PARAM_SCHEMA(SequentialActionHandlerParams,
                      holoscan::param_schema("com.nvidia.isaac.SequentialActionHandlerParams", 1),
                      holoscan::field<&SequentialActionHandlerParams::dt>("dt"));

/// Emits one joint command at a time from the latest action chunk.
class SequentialActionHandlerOp final : public holoscan::Operator<SequentialActionHandlerParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::JointCommandChunkCpuT> action_chunk;
  holoscan::Output<isaac::JointCommandCpuT> command;

 private:
  isaac::JointCommandChunkCpuT latest_chunk_;
  std::int64_t period_ns_{};
  std::int64_t latest_chunk_time_ns_{};
  bool started_{};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_SEQUENTIAL_ACTION_HANDLER_SEQUENTIAL_ACTION_HANDLER_OP_H
