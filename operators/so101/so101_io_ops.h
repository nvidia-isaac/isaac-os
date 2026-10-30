/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_SO101_SO101_IO_OPS_H
#define ISAAC_OS_OPERATORS_SO101_SO101_IO_OPS_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "common/so101_driver/feetech_serial_driver.h"
#include "joint_command_cpu_schema_traits.h"
#include "state_sample_cpu_schema_traits.h"

namespace isaac::os::ops {

struct So101StateSourceParams {
  double period{0.01};
};

HOLOSCAN_PARAM_SCHEMA(So101StateSourceParams,
                      holoscan::param_schema("com.nvidia.isaac.So101StateSourceParams", 1),
                      holoscan::field<&So101StateSourceParams::period>("period"));

class So101StateSourceOp final : public holoscan::Operator<So101StateSourceParams> {
 public:
  explicit So101StateSourceOp(std::shared_ptr<so101::FeetechSerialDriver> driver);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Output<isaac::StateSampleCpuT> state;

 private:
  std::shared_ptr<so101::FeetechSerialDriver> driver_;
  std::optional<std::chrono::steady_clock::time_point> read_failure_since_;
  bool completion_requested_{};
  std::uint64_t fault_epoch_{};
  std::uint64_t sequence_{};
  std::vector<float> positions_;
  std::vector<float> velocities_;
};

class So101CommandSinkOp final : public holoscan::Operator<> {
 public:
  explicit So101CommandSinkOp(std::shared_ptr<so101::FeetechSerialDriver> driver,
                              std::size_t expected_count = 0U);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::JointCommandCpuT> command;

 private:
  std::shared_ptr<so101::FeetechSerialDriver> driver_;
  std::size_t expected_count_{};
  std::size_t received_count_{};
  bool completion_requested_{};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_SO101_SO101_IO_OPS_H
