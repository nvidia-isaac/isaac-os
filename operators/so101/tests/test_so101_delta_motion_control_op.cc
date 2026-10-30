/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "operators/so101/so101_delta_motion_control_op.h"

namespace isaac::os::ops {
namespace {

using namespace std::chrono_literals;

class Source final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(state, "out_state");
    spec.output(delta, "out_delta");
  }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 1ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    const auto now = context.clock().now();
    if (!now) {
      return holoscan::make_unexpected(std::move(now).error());
    }
    auto state_output = state.allocate();
    if (!state_output) {
      return holoscan::make_unexpected(std::move(state_output).error());
    }
    isaac::StateSampleCpuT& sample = state_output->value();
    sample.position = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    sample.source_sequence = ++state_sequence_;
    const holoscan::EmitOptions options{
        .capture_time = *now,
        .trace = holoscan::TraceContext::root(state_sequence_),
    };
    auto emitted = state.emit(std::move(*state_output), options);
    if (!emitted) {
      return emitted;
    }

    auto delta_output = delta.allocate();
    if (!delta_output) {
      return holoscan::make_unexpected(std::move(delta_output).error());
    }
    isaac::PolicyDeltaCpuT& action = delta_output->value();
    action.action = {1.0F, 1.0F, 1.0F, 1.0F};
    action.source_sequence = 1U;
    return delta.emit(std::move(*delta_output), options);
  }

  holoscan::Output<isaac::StateSampleCpuT> state;
  holoscan::Output<isaac::PolicyDeltaCpuT> delta;

 private:
  std::uint64_t state_sequence_{};
};

struct CommandCapture {
  static void reset() {
    const std::lock_guard lock(mutex);
    count = 0U;
  }

  static void record(const isaac::JointCommandCpuT& command) {
    {
      const std::lock_guard lock(mutex);
      if (count < commands.size()) {
        commands[count++] = command;
      }
    }
    ready.notify_all();
  }

  static bool wait_for_two() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] { return count == commands.size(); });
  }

  static std::array<isaac::JointCommandCpuT, 2U> take() {
    const std::lock_guard lock(mutex);
    return commands;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::array<isaac::JointCommandCpuT, 2U> commands;
  static inline std::size_t count{};
};

class Sink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(command, "in_command"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{command});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = command.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    CommandCapture::record(received->data);
    return {};
  }

  holoscan::Input<isaac::JointCommandCpuT> command;
};

TEST(So101DeltaMotionControlOp, ContinuesLastTargetAfterDuplicateDelta) {
  CommandCapture::reset();
  holoscan::Graph graph{"so101-motion-control-test"};
  const auto source = graph.op<Source>("source");
  auto control = graph.op<So101DeltaMotionControlOp>("control");
  const auto sink = graph.op<Sink>("sink");
  control.set_params(So101DeltaMotionParams{
      .fixed_joint_positions = {0.0, 0.0},
      .joint_pos_limits_lower = {-1.0, -1.0, -1.0, -1.0},
      .joint_pos_limits_upper = {1.0, 1.0, 1.0, 1.0},
      .target_expiry = 1.0,
  });
  const holoscan::ConnectionOptions latest{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  graph.add_flow(source->state, control->state, latest);
  graph.add_flow(source->delta, control->delta, latest);
  graph.add_flow(control->command, sink->command);
  graph.partition("source").add(source);
  graph.partition("control").add(control);
  graph.partition("sink").add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool received_commands = CommandCapture::wait_for_two();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(received_commands) << "Received " << CommandCapture::count << " commands.";
  ASSERT_EQ(session.termination(), holoscan::RunTermination::kStopped);
  const auto commands = CommandCapture::take();
  const std::vector<float> expected_position{0.05F, 0.05F, 0.05F, 0.05F, 0.0F, 0.0F};
  const std::vector<float> expected_zeroes(6U, 0.0F);
  for (const isaac::JointCommandCpuT& command : commands) {
    EXPECT_EQ(command.position, expected_position);
    EXPECT_EQ(command.velocity, expected_zeroes);
    EXPECT_EQ(command.effort, expected_zeroes);
    EXPECT_EQ(command.kp, expected_zeroes);
    EXPECT_EQ(command.kd, expected_zeroes);
  }
}

}  // namespace
}  // namespace isaac::os::ops
