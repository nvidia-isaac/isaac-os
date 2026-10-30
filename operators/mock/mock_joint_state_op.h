/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_MOCK_MOCK_JOINT_STATE_OP_H
#define ISAAC_OS_OPERATORS_MOCK_MOCK_JOINT_STATE_OP_H

#include <cstddef>
#include <optional>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "joint_state_cpu_schema_traits.h"

namespace isaac::os::ops {

class MockJointStateOp final : public holoscan::Operator<> {
 public:
  explicit MockJointStateOp(std::size_t joint_count,
                            std::optional<std::size_t> compute_call_limit = std::nullopt)
      : joint_count_(joint_count), compute_call_limit_(compute_call_limit) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Output<isaac::JointStateCpuT> joint_state;
  holoscan::Output<holoscan::schema::ImuT> imu;

 private:
  std::size_t joint_count_;
  std::optional<std::size_t> compute_call_limit_;
  isaac::JointStateCpuT joint_state_message_;
  holoscan::schema::ImuT imu_message_;
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_MOCK_MOCK_JOINT_STATE_OP_H
