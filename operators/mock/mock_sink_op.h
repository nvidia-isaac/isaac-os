/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_MOCK_MOCK_SINK_OP_H
#define ISAAC_OS_OPERATORS_MOCK_MOCK_SINK_OP_H

#include <cstddef>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "joint_command_cpu_schema_traits.h"

namespace isaac::os::ops {

class MockSinkOp final : public holoscan::Operator<> {
 public:
  explicit MockSinkOp(std::size_t joint_count) : joint_count_(joint_count) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::JointCommandCpuT> command;

 private:
  std::size_t joint_count_;
  bool first_command_{true};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_MOCK_MOCK_SINK_OP_H
