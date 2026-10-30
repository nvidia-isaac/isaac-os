/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <utility>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "operators/simple_motion_planner/simple_motion_planner.h"

namespace isaac::os::ops {
namespace {

using namespace std::chrono_literals;

/// Publishes state on every tick and a target only on the first tick, so the clamp has to
/// keep commanding from the retained target.
class Source final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(state, "out_state");
    spec.output(target, "out_target");
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
    sample.position.assign(6U, 0.0F);
    sample.velocity.assign(6U, 0.0F);
    sample.source_sequence = ++sequence_;
    const holoscan::EmitOptions options{
        .capture_time = *now,
        .trace = holoscan::TraceContext::root(sequence_),
    };
    auto emitted = state.emit(std::move(*state_output), options);
    if (!emitted) {
      return emitted;
    }
    if (sequence_ > 1U) {
      return {};
    }
    auto target_output = target.allocate();
    if (!target_output) {
      return holoscan::make_unexpected(std::move(target_output).error());
    }
    isaac::JointCommandCpuT& command = target_output->value();
    command.position.assign(6U, 1.0F);
    command.velocity.assign(6U, 0.0F);
    command.effort.assign(6U, 0.0F);
    command.kp.assign(6U, 0.0F);
    command.kd.assign(6U, 0.0F);
    return target.emit(std::move(*target_output), options);
  }

  holoscan::Output<isaac::StateSampleCpuT> state;
  holoscan::Output<isaac::JointCommandCpuT> target;

 private:
  std::uint64_t sequence_{};
};

/// Collects emitted commands off the graph threads so a test can block until a given number
/// arrive and then inspect them.
struct CommandCapture {
  static void reset() {
    const std::lock_guard lock(mutex);
    count = 0U;
  }

  static void record(const isaac::JointCommandCpuT& command) {
    {
      const std::lock_guard lock(mutex);
      if (count == 0U) {
        first = command;
      }
      ++count;
    }
    ready.notify_all();
  }

  static bool wait_for_count(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [expected] { return count >= expected; });
  }

  static std::size_t received() {
    const std::lock_guard lock(mutex);
    return count;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline isaac::JointCommandCpuT first;
  static inline std::size_t count{};
};

/// Terminates the graph, handing every command it receives to `CommandCapture`.
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

/// Runs the source-planner-sink graph until `expected_commands` arrive, and reports whether
/// they all arrived before the timeout.
bool run_planner_graph(std::size_t expected_commands) {
  CommandCapture::reset();
  holoscan::Graph graph{"simple-motion-planner"};
  const auto source = graph.op<Source>("source");
  auto planner = graph.op<SimpleMotionPlannerOp>("motion_planner");
  const auto sink = graph.op<Sink>("sink");
  planner.set_params(SimpleMotionPlannerParams{.max_step_size = 0.20});
  const holoscan::ConnectionOptions latest{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  graph.add_flow(source->state, planner->state, latest);
  graph.add_flow(source->target, planner->target, latest);
  graph.add_flow(planner->command, sink->command);
  graph.partition("source").add(source);
  graph.partition("motion_planner").add(planner);
  graph.partition("sink").add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    return false;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool completed = CommandCapture::wait_for_count(expected_commands);
  session.request_stop();
  session.wait();
  return completed;
}

TEST(SimpleMotionPlannerOp, LimitsEachJointToMeasuredPlusMaxStep) {
  ASSERT_TRUE(run_planner_graph(1U));

  const std::vector<float> expected(6U, 0.20F);
  EXPECT_EQ(CommandCapture::first.position, expected);
}

TEST(SimpleMotionPlannerOp, KeepsCommandingAfterTheTargetStopsUpdating) {
  ASSERT_TRUE(run_planner_graph(60U));

  EXPECT_GE(CommandCapture::received(), 60U);
}

}  // namespace
}  // namespace isaac::os::ops
