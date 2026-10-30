/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_APPLICATIONS_SO101_REACH_MOCK_MOCK_OPERATORS_H
#define ISAAC_OS_APPLICATIONS_SO101_REACH_MOCK_MOCK_OPERATORS_H

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "joint_command_cpu_schema_traits.h"
#include "operators/so101/so101_io_ops.h"
#include "state_sample_cpu_schema_traits.h"

namespace isaac::os {

class MockStateSourceOp final : public holoscan::Operator<ops::So101StateSourceParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Output<isaac::StateSampleCpuT> state;

 private:
  std::uint64_t sequence_{};
};

class MockCommandSinkOp final : public holoscan::Operator<> {
 public:
  static void reset(std::size_t expected_count);
  [[nodiscard]] static bool wait_for_expected_count(std::chrono::duration<double> timeout);
  [[nodiscard]] static bool valid();

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::JointCommandCpuT> command;

 private:
  static inline std::mutex mutex_;
  static inline std::condition_variable changed_;
  static inline std::size_t expected_count_{};
  static inline std::size_t received_count_{};
  static inline bool valid_{};
};

}  // namespace isaac::os

#endif  // ISAAC_OS_APPLICATIONS_SO101_REACH_MOCK_MOCK_OPERATORS_H
