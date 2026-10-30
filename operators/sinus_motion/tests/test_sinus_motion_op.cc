/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
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

#include "operators/sinus_motion/sinus_motion_op.h"

namespace isaac::os::ops {
namespace {

using namespace std::chrono_literals;

/// Publishes a zeroed six-joint state sample every millisecond to pace the operator under test.
class Source final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(state, "out_state"); }

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
    auto output = state.allocate();
    if (!output) {
      return holoscan::make_unexpected(std::move(output).error());
    }
    isaac::StateSampleCpuT& sample = output->value();
    sample.position = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    sample.velocity = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    sample.source_sequence = ++sequence_;
    return state.emit(std::move(*output), holoscan::EmitOptions{
                                              .capture_time = *now,
                                              .trace = holoscan::TraceContext::root(sequence_),
                                          });
  }

  holoscan::Output<isaac::StateSampleCpuT> state;

 private:
  std::uint64_t sequence_{};
};

/// Collects emitted commands off the graph threads so a test can block until the first one
/// arrives and then inspect it.
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

  static bool wait_for_one() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] { return count > 0U; });
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

TEST(SinusMotionOp, ZeroAmplitudeEmitsDefaultPositions) {
  CommandCapture::reset();
  holoscan::Graph graph{"so101-sinus-zero-amplitude"};
  const auto source = graph.op<Source>("source");
  auto sinus = graph.op<SinusMotionOp>("sinus");
  const auto sink = graph.op<Sink>("sink");
  sinus.set_params(SinusMotionParams{
      .default_pos = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
      .sin_amplitude = 0.0,
  });
  const holoscan::ConnectionOptions latest{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  graph.add_flow(source->state, sinus->state, latest);
  graph.add_flow(sinus->command, sink->command);
  graph.partition("source").add(source);
  graph.partition("sinus").add(sinus);
  graph.partition("sink").add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool received = CommandCapture::wait_for_one();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(received);
  const std::vector<float> expected{0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};
  EXPECT_EQ(CommandCapture::first.position, expected);
}

TEST(SinusMotionParams, RejectsNegativeAmplitude) {
  EXPECT_FALSE(validate_sinus_motion_params(SinusMotionParams{.sin_amplitude = -0.1}));
}

}  // namespace
}  // namespace isaac::os::ops
