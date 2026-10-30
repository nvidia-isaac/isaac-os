/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_APPLICATIONS_SO101_SINUS_MOTION_MOCK_MOCK_OPERATORS_H
#define ISAAC_OS_APPLICATIONS_SO101_SINUS_MOTION_MOCK_MOCK_OPERATORS_H

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

/// Stands in for the SO-101 serial state source so the graph can run without hardware, emitting
/// zeroed joint state on a fixed clock instead of reading a servo bus.
class MockStateSourceOp final : public holoscan::Operator<ops::So101StateSourceParams> {
 public:
  /// Declares the joint-state output.
  void setup(holoscan::OperatorSpec& spec) override;
  /// Ticks on a fixed clock at the configured period, taking the place of serial state arrival.
  [[nodiscard]] holoscan::Contract contract() const override;
  /// Throws `std::runtime_error` if the configured period is not positive and finite.
  void start() override;
  /// Emits a zeroed six-joint sample carrying a monotonic sequence number, the current capture
  /// time, and a fresh trace root that downstream operators propagate.
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  /// Synthetic joint state replacing the hardware feedback path.
  holoscan::Output<isaac::StateSampleCpuT> state;

 private:
  std::uint64_t sequence_{};
};

/// Terminates the mock graph, checking every command it receives and recording source-to-sink
/// latency for the benchmark. State is static so the test harness can inspect results after the
/// session ends.
class MockCommandSinkOp final : public holoscan::Operator<> {
 public:
  /// Clears the counters and arms the sink for `expected_count` commands. Call before running.
  static void reset(std::size_t expected_count);
  /// Blocks until the expected number of commands arrive, returning false if `timeout` elapses.
  [[nodiscard]] static bool wait_for_expected_count(std::chrono::duration<double> timeout);
  /// Reports whether every command seen so far carried six finite entries in each field.
  [[nodiscard]] static bool valid();

  /// Declares the joint-command input.
  void setup(holoscan::OperatorSpec& spec) override;
  /// Triggers on each command, so the sink observes every message the planner emits.
  [[nodiscard]] holoscan::Contract contract() const override;
  /// Validates the command, records source-to-sink latency when benchmarking is enabled, and
  /// releases any thread waiting in `wait_for_expected_count`.
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  /// Joint command arriving from the motion planner.
  holoscan::Input<isaac::JointCommandCpuT> command;

 private:
  static inline std::mutex mutex_;
  static inline std::condition_variable changed_;
  static inline std::size_t expected_count_{};
  static inline std::size_t received_count_{};
  static inline bool valid_{};
};

}  // namespace isaac::os

#endif  // ISAAC_OS_APPLICATIONS_SO101_SINUS_MOTION_MOCK_MOCK_OPERATORS_H
