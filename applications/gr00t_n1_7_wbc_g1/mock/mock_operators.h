/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_APPLICATIONS_GR00T_N1_7_WBC_G1_MOCK_MOCK_OPERATORS_H
#define ISAAC_OS_APPLICATIONS_GR00T_N1_7_WBC_G1_MOCK_MOCK_OPERATORS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "joint_command_cpu_schema_traits.h"
#include "joint_state_cpu_schema_traits.h"

namespace isaac::os {

class MockRobotStateOp final : public holoscan::Operator<> {
 public:
  explicit MockRobotStateOp(std::size_t joint_count) : joint_count_(joint_count) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Output<isaac::JointStateCpuT> wbc_joint_state;
  holoscan::Output<isaac::JointStateCpuT> gr00t_joint_state;
  holoscan::Output<holoscan::schema::ImuT> imu;

 private:
  std::size_t joint_count_;
  isaac::JointStateCpuT joint_state_message_;
  holoscan::schema::ImuT imu_message_;
};

struct MockImageParams {
  double frame_rate{30.0};
};

HOLOSCAN_PARAM_SCHEMA(MockImageParams,
                      holoscan::param_schema("com.nvidia.isaac.MockImageParams", 1),
                      holoscan::field<&MockImageParams::frame_rate>("frame_rate"));

class MockImageOp final : public holoscan::Operator<MockImageParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::Tensor> image;

 private:
  std::uint32_t sequence_{};
};

class MockCommandSinkOp final : public holoscan::Operator<> {
 public:
  MockCommandSinkOp(std::size_t expected_count, const std::vector<std::string>& joint_order,
                    const std::vector<std::string>& overlay_joints);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::JointCommandCpuT> command;
  holoscan::Input<isaac::JointCommandCpuT> action_command;
  holoscan::Input<isaac::JointCommandCpuT> primary_command;

 private:
  std::size_t expected_count_;
  std::size_t joint_count_;
  std::size_t received_count_{};
  bool completion_requested_{};
  bool composition_validated_{};
  std::vector<bool> overlay_indices_;
};

}  // namespace isaac::os

#endif  // ISAAC_OS_APPLICATIONS_GR00T_N1_7_WBC_G1_MOCK_MOCK_OPERATORS_H
