/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "operators/gr00t_action_handler/gr00t_action_handler_op.h"

namespace isaac::os::ops {
namespace {

using namespace std::chrono_literals;

constexpr auto kControlPeriod = 20ms;
constexpr std::uint64_t kFrameId = 7U;
constexpr holoscan::TraceId kTraceId = 11U;

isaac::JointCommandCpuT make_command(float base, std::size_t joint_count = 2U) {
  isaac::JointCommandCpuT command;
  command.position.resize(joint_count);
  command.velocity.resize(joint_count);
  command.effort.resize(joint_count);
  command.kp.resize(joint_count);
  command.kd.resize(joint_count);
  for (std::size_t joint = 0; joint < joint_count; ++joint) {
    const float offset = static_cast<float>(joint);
    command.position[joint] = base + offset;
    command.velocity[joint] = base + 10.0F + offset;
    command.effort[joint] = base + 20.0F + offset;
    command.kp[joint] = base + 30.0F + offset;
    command.kd[joint] = base + 40.0F + offset;
  }
  return command;
}

isaac::JointCommandChunkCpuT make_chunk(float base, std::uint32_t joint_count = 2U,
                                        std::uint32_t action_count = 5U, float step_delta = 10.0F) {
  isaac::JointCommandChunkCpuT chunk;
  chunk.joint_count = joint_count;
  chunk.action_count = action_count;
  chunk.position.resize(static_cast<std::size_t>(joint_count) * action_count);
  chunk.effort.resize(chunk.position.size());
  chunk.kp.resize(joint_count);
  chunk.kd.resize(joint_count);
  for (std::size_t joint = 0; joint < joint_count; ++joint) {
    for (std::size_t step = 0; step < action_count; ++step) {
      const std::size_t index = joint * action_count + step;
      chunk.position[index] =
          base + static_cast<float>(joint) * 100.0F + static_cast<float>(step) * step_delta;
      chunk.effort[index] = chunk.position[index] + 20.0F;
    }
    chunk.kp[joint] = 40.0F + static_cast<float>(joint);
    chunk.kd[joint] = 1.0F + static_cast<float>(joint) * 0.5F;
  }
  return chunk;
}

struct Tick {
  isaac::JointCommandCpuT control_command;
  std::optional<isaac::JointCommandChunkCpuT> action_chunk;
  std::chrono::nanoseconds image_age{};
  bool stamp_image_time{true};
  holoscan::SampleFlags flags{holoscan::SampleFlags::kNone};
};

class CommandSource final : public holoscan::Operator<> {
 public:
  explicit CommandSource(std::vector<Tick> ticks) : ticks_(std::move(ticks)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(control_command, "out_control_command");
    spec.output(action_chunk, "out_action_chunk");
  }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kControlPeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    const Tick& configured = ticks_[std::min(index_, ticks_.size() - 1U)];
    auto now = context.clock().now();
    if (!now) {
      return holoscan::make_unexpected(std::move(now).error());
    }
    if (configured.action_chunk) {
      isaac::JointCommandChunkCpuT chunk = *configured.action_chunk;
      if (configured.stamp_image_time) {
        chunk.image_capture_time_ns = now->timestamp_ns - configured.image_age.count();
      }
      auto emitted =
          action_chunk.emit(std::move(chunk), holoscan::EmitOptions{.capture_time = *now});
      if (!emitted) {
        return emitted;
      }
    }
    ++index_;
    return control_command.emit(
        configured.control_command,
        holoscan::EmitOptions{.capture_time = *now,
                              .frame_id = kFrameId,
                              .trace = holoscan::TraceContext::root(kTraceId),
                              .flags = configured.flags});
  }

  holoscan::Output<isaac::JointCommandCpuT> control_command;
  holoscan::Output<isaac::JointCommandChunkCpuT> action_chunk;

 private:
  std::vector<Tick> ticks_;
  std::size_t index_{};
};

struct CapturedTick {
  isaac::JointCommandCpuT control_command;
  holoscan::SampleMetadata control_metadata;
  std::optional<isaac::JointCommandCpuT> action_command;
  std::optional<holoscan::SampleMetadata> action_metadata;
};

struct CommandCapture {
  static void reset() {
    const std::lock_guard lock(mutex);
    ticks.clear();
  }

  static void record(CapturedTick tick) {
    {
      const std::lock_guard lock(mutex);
      ticks.push_back(std::move(tick));
    }
    ready.notify_all();
  }

  static std::vector<CapturedTick> wait_for(std::size_t count) {
    std::unique_lock lock(mutex);
    if (!ready.wait_for(lock, 5s, [count] { return ticks.size() >= count; })) {
      return {};
    }
    return {ticks.begin(), ticks.begin() + static_cast<std::ptrdiff_t>(count)};
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<CapturedTick> ticks;
};

class CommandSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(control_command, "in_control_command");
    spec.input(action_command, "in_action_command");
  }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{control_command});
    contract.input(action_command,
                   holoscan::LatestWithin{.max_age = std::chrono::nanoseconds::max()});
    contract.on_missing(action_command, holoscan::SafeDefault{});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    CapturedTick captured;
    if (!action_command.is_missing()) {
      auto received_action = action_command.receive();
      if (!received_action) {
        return holoscan::make_unexpected(std::move(received_action).error());
      }
      captured.action_command = std::move(received_action->data);
      captured.action_metadata = received_action->metadata;
    }
    auto received_control = control_command.receive();
    if (!received_control) {
      return holoscan::make_unexpected(std::move(received_control).error());
    }
    captured.control_command = std::move(received_control->data);
    captured.control_metadata = received_control->metadata;
    CommandCapture::record(std::move(captured));
    return {};
  }

  holoscan::Input<isaac::JointCommandCpuT> control_command;
  holoscan::Input<isaac::JointCommandCpuT> action_command;
};

std::vector<CapturedTick> run_graph(std::vector<Tick> ticks, Gr00tActionHandlerParams params,
                                    std::size_t expected_count) {
  CommandCapture::reset();
  holoscan::Graph graph{"gr00t-action-handler-test"};
  const auto source = graph.op<CommandSource>("source", std::move(ticks));
  auto handler = graph.op<Gr00tActionHandlerOp>("handler");
  const auto sink = graph.op<CommandSink>("sink");
  handler.set_params(std::move(params));
  graph.add_flow(source->control_command, handler->control_command);
  graph.add_flow(source->action_chunk, handler->action_chunk);
  graph.add_flow(handler->forwarded_control_command, sink->control_command);
  graph.add_flow(handler->action_command, sink->action_command);
  graph.partition("source").add(source);
  graph.partition("control").add(handler).add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  EXPECT_TRUE(plan.ok()) << plan.json();
  if (!plan.ok()) {
    return {};
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  auto captured = CommandCapture::wait_for(expected_count);
  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kStopped);
  return captured;
}

TEST(Gr00tActionHandler, AlwaysForwardsControlCommandsAndPreservesMetadata) {
  auto first_invalid = make_chunk(0.0F);
  first_invalid.kp.pop_back();
  auto replacement_invalid = make_chunk(1000.0F);
  replacement_invalid.kp.pop_back();
  const auto captured = run_graph(
      {
          {.control_command = make_command(2.0F), .flags = holoscan::SampleFlags::kDegraded},
          {.control_command = make_command(3.0F),
           .action_chunk = std::move(first_invalid),
           .flags = holoscan::SampleFlags::kDegraded},
          {.control_command = make_command(4.0F),
           .action_chunk = make_chunk(0.0F),
           .flags = holoscan::SampleFlags::kDegraded},
          {.control_command = make_command(5.0F),
           .action_chunk = std::move(replacement_invalid),
           .flags = holoscan::SampleFlags::kDegraded},
      },
      {}, 4U);

  ASSERT_EQ(captured.size(), 4U);
  EXPECT_FALSE(captured[0].action_command);
  EXPECT_FALSE(captured[1].action_command);
  ASSERT_TRUE(captured[2].action_command);
  ASSERT_TRUE(captured[3].action_command);
  EXPECT_LT(captured[3].action_command->position[0], 100.0F);
  for (std::size_t index = 0; index < captured.size(); ++index) {
    const auto& tick = captured[index];
    EXPECT_EQ(tick.control_command.position, make_command(static_cast<float>(index + 2U)).position);
    EXPECT_TRUE(tick.control_metadata.capture_time().valid());
    EXPECT_EQ(tick.control_metadata.frame_id, kFrameId);
    EXPECT_EQ(tick.control_metadata.trace_id, kTraceId);
    EXPECT_EQ(tick.control_metadata.flags, holoscan::SampleFlags::kDegraded);
  }
}

TEST(Gr00tActionHandler, SelectsOneInterpolatedActionPerControlCommand) {
  constexpr std::size_t kTickCount = 4U;
  std::vector<Tick> ticks(kTickCount);
  for (std::size_t index = 0; index < ticks.size(); ++index) {
    ticks[index].control_command = make_command(static_cast<float>(index));
  }
  ticks.front().action_chunk = make_chunk(0.0F, 2U, 4U, 1000.0F);

  const auto captured = run_graph(std::move(ticks), {.action_interval_s = 2.0}, kTickCount);

  ASSERT_EQ(captured.size(), kTickCount);
  for (const auto& tick : captured) {
    ASSERT_TRUE(tick.action_command);
    ASSERT_TRUE(tick.action_metadata);
    EXPECT_EQ(tick.action_metadata->capture_timestamp_ns,
              tick.control_metadata.capture_timestamp_ns);
    EXPECT_EQ(tick.action_command->velocity, (std::vector<float>{0.0F, 0.0F}));
    EXPECT_EQ(tick.action_command->kp, (std::vector<float>{40.0F, 41.0F}));
    EXPECT_EQ(tick.action_command->kd, (std::vector<float>{1.0F, 1.5F}));
  }
  const float first_position = captured.front().action_command->position[0];
  const float last_position = captured.back().action_command->position[0];
  EXPECT_GE(first_position, 0.0F);
  EXPECT_LT(first_position, 500.0F);
  EXPECT_GT(last_position, first_position + 5.0F);
  EXPECT_LT(last_position, 500.0F);
  EXPECT_FLOAT_EQ(captured.back().action_command->effort[0], last_position + 20.0F);
}

TEST(Gr00tActionHandler, UsesChunkArrivalForMissingOrFutureImageTime) {
  for (const bool future_timestamp : {false, true}) {
    std::vector<Tick> ticks(3U);
    for (std::size_t index = 0; index < ticks.size(); ++index) {
      ticks[index].control_command = make_command(static_cast<float>(index));
    }
    ticks.front().action_chunk = make_chunk(0.0F, 2U, 3U, 100.0F);
    ticks.front().stamp_image_time = future_timestamp;
    ticks.front().image_age = -1s;

    const auto captured = run_graph(std::move(ticks), {.action_interval_s = 0.1}, 3U);

    ASSERT_EQ(captured.size(), 3U);
    ASSERT_TRUE(captured.front().action_command);
    ASSERT_TRUE(captured.back().action_command);
    EXPECT_LT(captured.front().action_command->position[0], 50.0F);
    EXPECT_GT(captured.back().action_command->position[0], 10.0F);
  }
}

TEST(Gr00tActionHandler, BlendsFromThePreviousCommandTime) {
  std::vector<Tick> ticks(5U);
  for (std::size_t index = 0; index < ticks.size(); ++index) {
    ticks[index].control_command = make_command(static_cast<float>(index));
  }
  ticks[0].action_chunk = make_chunk(0.0F);
  ticks[2].action_chunk = make_chunk(1000.0F);
  ticks[2].image_age = 40ms;

  const auto captured =
      run_graph(std::move(ticks), {.action_interval_s = 0.02, .handover_blend_steps = 2.0}, 5U);

  ASSERT_EQ(captured.size(), 5U);
  for (const auto& tick : captured) {
    ASSERT_TRUE(tick.action_command);
  }
  EXPECT_LT(captured[2].action_command->position[0], 250.0F);
  EXPECT_GT(captured[2].action_command->position[0], captured[1].action_command->position[0]);
  EXPECT_GT(captured.back().action_command->position[0], 900.0F);
}

TEST(Gr00tActionHandler, SkipsBlendForAReplacementWithDifferentJointCount) {
  constexpr float kPreviousPosition = 2.0F;
  constexpr float kReplacementPosition = 1000.0F;
  std::vector<Tick> ticks(4U);
  for (std::size_t index = 0; index < ticks.size(); ++index) {
    ticks[index].control_command = make_command(static_cast<float>(index));
  }
  ticks[0].action_chunk = make_chunk(kPreviousPosition, 2U, 5U, 0.0F);
  ticks[2].action_chunk = make_chunk(kReplacementPosition, 3U, 5U, 0.0F);

  const auto captured =
      run_graph(std::move(ticks), {.action_interval_s = 0.02, .handover_blend_steps = 2.0}, 4U);

  ASSERT_EQ(captured.size(), 4U);
  EXPECT_TRUE(std::none_of(captured.begin(), captured.end(), [=](const auto& tick) {
    return tick.action_command && !tick.action_command->position.empty() &&
           tick.action_command->position[0] > kPreviousPosition &&
           tick.action_command->position[0] < kReplacementPosition;
  }));
  const auto replacement = std::ranges::find_if(captured, [](const auto& tick) {
    return tick.action_command && tick.action_command->position.size() == 3U;
  });
  ASSERT_NE(replacement, captured.end());
  EXPECT_GE(replacement->action_command->position[0], kReplacementPosition);
}

TEST(Gr00tActionHandler, EmitsZeroEffortWhenFeedforwardTorqueIsDisabled) {
  const auto captured =
      run_graph({{.control_command = make_command(1.0F), .action_chunk = make_chunk(0.0F)}},
                {.action_interval_s = 0.02, .feedforward_torque = false}, 1U);

  ASSERT_EQ(captured.size(), 1U);
  ASSERT_TRUE(captured[0].action_command);
  EXPECT_EQ(captured[0].action_command->effort, (std::vector<float>{0.0F, 0.0F}));
}

}  // namespace
}  // namespace isaac::os::ops
