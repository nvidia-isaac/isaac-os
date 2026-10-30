/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "joint_command_cpu_schema_traits.h"
#include "operators/start_up/start_up_op.h"

namespace isaac::os::ops::handover_test {
namespace {

using namespace std::chrono_literals;

constexpr std::int64_t kKnownCaptureTimeNs = 42;
constexpr std::uint64_t kKnownFrameId = 7U;
constexpr holoscan::TraceId kKnownTraceId = 11U;

struct Tick {
  isaac::JointCommandCpuT primary;
  holoscan::SampleFlags primary_flags{holoscan::SampleFlags::kNone};
  std::int64_t capture_time_ns{kKnownCaptureTimeNs};
  std::optional<isaac::JointStateCpuT> measured_state;
  std::optional<std::int64_t> state_capture_time_ns;
  std::optional<bool> enable_lower;
  std::optional<bool> enable_upper;
};

isaac::JointCommandCpuT make_command(float base, std::size_t count = 4U) {
  isaac::JointCommandCpuT command;
  command.position.resize(count);
  command.velocity.resize(count);
  command.effort.resize(count);
  command.kp.resize(count);
  command.kd.resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto offset = static_cast<float>(index);
    command.position[index] = base + offset;
    command.velocity[index] = base + 10.0F + offset;
    command.effort[index] = base + 20.0F + offset;
    command.kp[index] = base + 30.0F + offset;
    command.kd[index] = base + 40.0F + offset;
  }
  return command;
}

class CommandSource final : public holoscan::Operator<> {
 public:
  explicit CommandSource(std::vector<Tick> ticks) : ticks_(std::move(ticks)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(state, "out_joint_state");
    spec.output(primary, "out_primary_joint_command");
    spec.output(lower, "out_enable_lower");
    spec.output(upper, "out_enable_upper");
  }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 10ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    const Tick& tick = ticks_[std::min(index_, ticks_.size() - 1U)];
    if (tick.measured_state) {
      auto emitted = state.emit(
          *tick.measured_state,
          holoscan::EmitOptions{.capture_time = context.clock().at_ns(
                                    tick.state_capture_time_ns.value_or(tick.capture_time_ns))});
      if (!emitted) return emitted;
    }
    const holoscan::EmitOptions enable_options{.capture_time =
                                                   context.clock().at_ns(tick.capture_time_ns)};
    if (tick.enable_lower.has_value()) {
      auto emitted = lower.emit(*tick.enable_lower, enable_options);
      if (!emitted) return emitted;
    }
    if (tick.enable_upper.has_value()) {
      auto emitted = upper.emit(*tick.enable_upper, enable_options);
      if (!emitted) return emitted;
    }
    ++index_;
    return primary.emit(
        tick.primary,
        holoscan::EmitOptions{.capture_time = context.clock().at_ns(tick.capture_time_ns),
                              .frame_id = kKnownFrameId,
                              .trace = holoscan::TraceContext::root(kKnownTraceId),
                              .flags = tick.primary_flags});
  }

  holoscan::Output<isaac::JointStateCpuT> state;
  holoscan::Output<isaac::JointCommandCpuT> primary;
  holoscan::Output<bool> lower;
  holoscan::Output<bool> upper;

 private:
  std::vector<Tick> ticks_;
  std::size_t index_{};
};

struct CommandCapture {
  static void reset() {
    const std::lock_guard lock(mutex);
    commands.clear();
    capture_times.clear();
    frame_ids.clear();
    trace_ids.clear();
    flags.clear();
  }

  static void record(const isaac::JointCommandCpuT& command,
                     const holoscan::SampleMetadata& metadata) {
    {
      const std::lock_guard lock(mutex);
      commands.push_back(command);
      capture_times.push_back(metadata.capture_timestamp_ns);
      frame_ids.push_back(metadata.frame_id);
      trace_ids.push_back(metadata.trace_id);
      flags.push_back(metadata.flags);
    }
    ready.notify_all();
  }

  static std::vector<isaac::JointCommandCpuT> wait_for(std::size_t count) {
    std::unique_lock lock(mutex);
    if (!ready.wait_for(lock, 5s, [count] { return commands.size() >= count; })) {
      return {};
    }
    return {commands.begin(), commands.begin() + static_cast<std::ptrdiff_t>(count)};
  }

  static std::int64_t capture_time_at(std::size_t index) {
    const std::lock_guard lock(mutex);
    return capture_times.at(index);
  }

  static std::uint64_t frame_id_at(std::size_t index) {
    const std::lock_guard lock(mutex);
    return frame_ids.at(index);
  }

  static holoscan::TraceId trace_id_at(std::size_t index) {
    const std::lock_guard lock(mutex);
    return trace_ids.at(index);
  }

  static holoscan::SampleFlags flags_at(std::size_t index) {
    const std::lock_guard lock(mutex);
    return flags.at(index);
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<isaac::JointCommandCpuT> commands;
  static inline std::vector<std::int64_t> capture_times;
  static inline std::vector<std::uint64_t> frame_ids;
  static inline std::vector<holoscan::TraceId> trace_ids;
  static inline std::vector<holoscan::SampleFlags> flags;
};

class CommandSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(command, "in_joint_command"); }

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
    CommandCapture::record(received->data, received->metadata);
    return {};
  }

  holoscan::Input<isaac::JointCommandCpuT> command;
};

std::vector<isaac::JointCommandCpuT> run_graph(std::vector<Tick> ticks, std::size_t count,
                                               double duration = 1.0, bool with_enables = false) {
  CommandCapture::reset();
  holoscan::Graph graph{"startup-blend-test"};
  const auto source = graph.op<CommandSource>("source", std::move(ticks));
  auto blend = graph.op<StartUpOp>("blend", with_enables);
  const auto sink = graph.op<CommandSink>("sink");
  StartUpParams params;
  params.sdk_joint_order = {"j0", "j1", "j2", "j3"};
  params.upper_body_joints = {"j2", "j3"};
  params.handover_only = true;
  params.handover_duration = duration;
  blend.set_params(std::move(params));
  graph.add_flow(source->primary, blend->joint_command);
  graph.add_flow(source->state, blend->joint_state,
                 {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U});
  if (with_enables) {
    graph.add_flow(source->lower, blend->enable_lower);
    graph.add_flow(source->upper, blend->enable_upper);
  }
  graph.add_flow(blend->command, sink->command);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto plan = holoscan::compile(graph);
  EXPECT_TRUE(plan.ok()) << plan.json();
  if (!plan.ok()) return {};
  auto session = holoscan::run_async(plan);
  auto commands = CommandCapture::wait_for(count);
  session.request_stop();
  session.wait();
  for (const auto& diagnostic : session.diagnostics()) {
    if (session.termination() == holoscan::RunTermination::kFailed)
      ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
  }
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kStopped);
  return commands;
}

isaac::JointCommandCpuT target(float j2, float j3) {
  auto result = make_command(100.0F);
  result.position[2] = j2;
  result.position[3] = j3;
  return result;
}

isaac::JointStateCpuT pose(float j2 = 0.0F, float j3 = 0.0F) {
  isaac::JointStateCpuT result;
  result.position = {0.0F, 0.0F, j2, j3};
  return result;
}

TEST(StartUpHandover, StartsAtMeasuredPoseAndPreservesOtherFieldsAndMetadata) {
  const auto input = target(1, 2);
  const auto commands = run_graph({{.primary = input,
                                    .primary_flags = holoscan::SampleFlags::kDegraded,
                                    .capture_time_ns = 1000000000,
                                    .measured_state = pose(0.2F, -0.3F)}},
                                  1);
  ASSERT_EQ(commands.size(), 1U);
  EXPECT_FLOAT_EQ(commands[0].position[0], input.position[0]);
  EXPECT_FLOAT_EQ(commands[0].position[1], input.position[1]);
  EXPECT_FLOAT_EQ(commands[0].position[2], 0.2F);
  EXPECT_FLOAT_EQ(commands[0].position[3], -0.3F);
  EXPECT_FLOAT_EQ(commands[0].velocity[0], input.velocity[0]);
  EXPECT_FLOAT_EQ(commands[0].effort[0], input.effort[0]);
  EXPECT_FLOAT_EQ(commands[0].velocity[2], 0.0F);
  EXPECT_FLOAT_EQ(commands[0].effort[2], 0.0F);
  EXPECT_EQ(commands[0].kp, input.kp);
  EXPECT_EQ(commands[0].kd, input.kd);
  EXPECT_EQ(CommandCapture::capture_time_at(0), 1000000000);
  EXPECT_EQ(CommandCapture::frame_id_at(0), kKnownFrameId);
  EXPECT_EQ(CommandCapture::trace_id_at(0), kKnownTraceId);
  EXPECT_EQ(CommandCapture::flags_at(0), holoscan::SampleFlags::kDegraded);
}

TEST(StartUpHandover, UsesSampleTimeAndCommonFractionWithoutReanchoring) {
  const auto commands = run_graph(
      {{.primary = target(1, 2), .capture_time_ns = 1000000000, .measured_state = pose()},
       {.primary = target(1, 2), .capture_time_ns = 1020000000, .measured_state = pose(9, 9)},
       {.primary = target(1, 2), .capture_time_ns = 1070000000}},
      3);
  ASSERT_EQ(commands.size(), 3U);
  EXPECT_NEAR(commands[1].position[2], 0.001184F, 1e-6F);
  EXPECT_NEAR(commands[1].position[3], 0.002368F, 1e-6F);
  EXPECT_NEAR(commands[2].position[2], 0.014014F, 1e-6F);
  EXPECT_NEAR(commands[2].position[3], 0.028028F, 1e-6F);
}

TEST(StartUpHandover, TracksNewTargetsDuringHandover) {
  const auto commands =
      run_graph({{.primary = target(1, 2), .capture_time_ns = 1000000000, .measured_state = pose()},
                 {.primary = target(1, 2), .capture_time_ns = 1020000000},
                 {.primary = target(-1, -2), .capture_time_ns = 1040000000}},
                3);
  ASSERT_EQ(commands.size(), 3U);
  EXPECT_NEAR(commands[2].position[2], -0.004672F, 1e-6F);
  EXPECT_NEAR(commands[2].position[3], -0.009344F, 1e-6F);
}

TEST(StartUpHandover, PermanentlyPassesThroughAfterDuration) {
  const auto commands = run_graph(
      {{.primary = target(0.005F, 0.01F), .capture_time_ns = 1000000000, .measured_state = pose()},
       {.primary = target(0.005F, 0.01F), .capture_time_ns = 1020000000},
       {.primary = target(-1, -2), .capture_time_ns = 1040000000, .measured_state = pose(9, 9)},
       {.primary = target(4, 8), .capture_time_ns = 1060000000}},
      4, 0.02);
  ASSERT_EQ(commands.size(), 4U);
  EXPECT_FLOAT_EQ(commands[1].position[2], 0.005F);
  EXPECT_FLOAT_EQ(commands[1].position[3], 0.01F);
  EXPECT_FLOAT_EQ(commands[2].position[2], -1.0F);
  EXPECT_FLOAT_EQ(commands[2].position[3], -2.0F);
  EXPECT_FLOAT_EQ(commands[3].position[3], 8.0F);
}

TEST(StartUpHandover, WithholdsCommandsUntilFreshStateIsAvailable) {
  const auto commands = run_graph({{.primary = target(1, 2), .capture_time_ns = 1000000000},
                                   {.primary = target(1, 2),
                                    .capture_time_ns = 1100000000,
                                    .measured_state = pose(),
                                    .state_capture_time_ns = 1000000000},
                                   {.primary = target(1, 2),
                                    .capture_time_ns = 1120000000,
                                    .measured_state = pose(0.1F, 0.2F)}},
                                  1);
  ASSERT_EQ(commands.size(), 1U);
  EXPECT_EQ(CommandCapture::capture_time_at(0), 1120000000);
  EXPECT_FLOAT_EQ(commands[0].position[2], 0.1F);
}

TEST(StartUpHandover, DoesNotRewindTimeOnDuplicateOrBackwardSamples) {
  const auto commands =
      run_graph({{.primary = target(1, 2), .capture_time_ns = 1000000000, .measured_state = pose()},
                 {.primary = target(1, 2), .capture_time_ns = 1020000000},
                 {.primary = target(1, 2), .capture_time_ns = 1020000000},
                 {.primary = target(1, 2), .capture_time_ns = 900000000},
                 {.primary = target(1, 2), .capture_time_ns = 1040000000}},
                5);
  ASSERT_EQ(commands.size(), 5U);
  EXPECT_NEAR(commands[1].position[3], 0.002368F, 1e-6F);
  EXPECT_FLOAT_EQ(commands[2].position[3], commands[1].position[3]);
  EXPECT_FLOAT_EQ(commands[3].position[3], commands[1].position[3]);
  EXPECT_NEAR(commands[4].position[3], 0.009344F, 1e-6F);
}

TEST(StartUpHandover, InvalidTargetDoesNotPoisonBlend) {
  const auto commands =
      run_graph({{.primary = target(1, 2), .capture_time_ns = 1000000000, .measured_state = pose()},
                 {.primary = target(std::numeric_limits<float>::quiet_NaN(), 2),
                  .capture_time_ns = 1020000000},
                 {.primary = target(1, 2), .capture_time_ns = 1040000000}},
                3);
  ASSERT_EQ(commands.size(), 3U);
  EXPECT_EQ(commands[1].position, commands[0].position);
  EXPECT_EQ(CommandCapture::flags_at(1), holoscan::SampleFlags::kDegraded);
  EXPECT_NEAR(commands[2].position[3], 0.009344F, 1e-6F);
}

TEST(StartUpHandover, InvalidUpperTargetsPreserveLowerControlBeforeDuringAndAfterHandover) {
  for (const bool with_enables : {false, true}) {
    SCOPED_TRACE(with_enables);
    const auto invalid = target(std::numeric_limits<float>::quiet_NaN(), 9);
    std::vector<Tick> ticks{
        {.primary = invalid, .capture_time_ns = 1000000000, .measured_state = pose(0.2F, 0.4F)},
        {.primary = target(1, 2),
         .capture_time_ns = 2000000000,
         .measured_state = pose(0.3F, 0.6F)},
        {.primary = target(1, 2), .capture_time_ns = 2500000000},
        {.primary = invalid, .capture_time_ns = 2600000000},
        {.primary = target(1, 2), .capture_time_ns = 3000000000},
        {.primary = target(0.5F, 0.8F), .capture_time_ns = 3100000000},
        {.primary = invalid, .capture_time_ns = 3200000000},
        {.primary = target(0.6F, 0.9F), .capture_time_ns = 3300000000}};
    if (with_enables) {
      ticks[0].enable_lower = true;
      ticks[0].enable_upper = true;
    }
    const auto commands = run_graph(ticks, ticks.size(), 1.0, with_enables);
    ASSERT_EQ(commands.size(), ticks.size());
    EXPECT_FLOAT_EQ(commands[0].position[2], 0.2F);
    EXPECT_FLOAT_EQ(commands[0].position[3], 0.4F);
    EXPECT_FLOAT_EQ(commands[1].position[2], 0.3F);
    EXPECT_NEAR(commands[2].position[3], 1.3F, 1e-6F);
    EXPECT_EQ(commands[3].position, commands[2].position);
    EXPECT_EQ(commands[4].position, ticks[4].primary.position);
    EXPECT_EQ(commands[6].position, commands[5].position);
    EXPECT_EQ(commands[7].position, ticks[7].primary.position);
    for (std::size_t i = 0; i < commands.size(); ++i) {
      SCOPED_TRACE(i);
      for (std::size_t joint = 0; joint < 2; ++joint) {
        EXPECT_FLOAT_EQ(commands[i].position[joint], ticks[i].primary.position[joint]);
        EXPECT_FLOAT_EQ(commands[i].velocity[joint], ticks[i].primary.velocity[joint]);
        EXPECT_FLOAT_EQ(commands[i].effort[joint], ticks[i].primary.effort[joint]);
      }
      EXPECT_EQ(commands[i].kp, ticks[i].primary.kp);
      EXPECT_EQ(commands[i].kd, ticks[i].primary.kd);
      EXPECT_EQ(CommandCapture::capture_time_at(i), ticks[i].capture_time_ns);
      EXPECT_EQ(CommandCapture::frame_id_at(i), kKnownFrameId);
      EXPECT_EQ(CommandCapture::trace_id_at(i), kKnownTraceId);
    }
    for (const std::size_t i : {0U, 3U, 6U}) {
      SCOPED_TRACE(i);
      EXPECT_EQ(CommandCapture::flags_at(i), holoscan::SampleFlags::kDegraded);
      EXPECT_FLOAT_EQ(commands[i].velocity[2], 0.0F);
      EXPECT_FLOAT_EQ(commands[i].velocity[3], 0.0F);
      EXPECT_FLOAT_EQ(commands[i].effort[2], 0.0F);
      EXPECT_FLOAT_EQ(commands[i].effort[3], 0.0F);
    }
    EXPECT_EQ(CommandCapture::flags_at(7), holoscan::SampleFlags::kNone);
  }
}

TEST(StartUpHandover, InvalidLowerTargetIsStillRejected) {
  auto invalid = target(1, 2);
  invalid.position[0] = std::numeric_limits<float>::quiet_NaN();
  const auto commands = run_graph(
      {{.primary = invalid, .capture_time_ns = 1000000000, .measured_state = pose()},
       {.primary = target(1, 2), .capture_time_ns = 1020000000, .measured_state = pose()}},
      1);
  ASSERT_EQ(commands.size(), 1U);
  EXPECT_EQ(CommandCapture::capture_time_at(0), 1020000000);
  EXPECT_FLOAT_EQ(commands[0].position[0], target(1, 2).position[0]);
}

TEST(StartUpHandover, UsesWaypointSmoothstepOverConfiguredDuration) {
  const auto commands =
      run_graph({{.primary = target(1, 2), .capture_time_ns = 1000000000, .measured_state = pose()},
                 {.primary = target(1, 2), .capture_time_ns = 1500000000},
                 {.primary = target(1, 2), .capture_time_ns = 2000000000},
                 {.primary = target(1, 2), .capture_time_ns = 2500000000},
                 {.primary = target(1, 2), .capture_time_ns = 3000000000}},
                5, 2.0);
  ASSERT_EQ(commands.size(), 5U);
  EXPECT_FLOAT_EQ(commands[0].position[3], 0.0F);
  EXPECT_FLOAT_EQ(commands[1].position[3], 0.3125F);
  EXPECT_FLOAT_EQ(commands[2].position[3], 1.0F);
  EXPECT_FLOAT_EQ(commands[3].position[3], 1.6875F);
  EXPECT_FLOAT_EQ(commands[4].position[3], 2.0F);
}

TEST(StartUpHandover, InvalidStateCannotRefreshAnOldPose) {
  auto invalid = pose();
  invalid.position.pop_back();
  const auto commands = run_graph(
      {{.primary = target(1, 2),
        .capture_time_ns = 1000000000,
        .measured_state = pose(),
        .state_capture_time_ns = 900000000},
       {.primary = target(1, 2), .capture_time_ns = 1100000000, .measured_state = invalid},
       {.primary = target(1, 2),
        .capture_time_ns = 1120000000,
        .measured_state = pose(0.3F, 0.4F)}},
      1);
  ASSERT_EQ(commands.size(), 1U);
  EXPECT_EQ(CommandCapture::capture_time_at(0), 1120000000);
  EXPECT_FLOAT_EQ(commands[0].position[2], 0.3F);
}

TEST(StartUpHandover, FutureStateCannotInitializeHandover) {
  const auto commands = run_graph({{.primary = target(1, 2),
                                    .capture_time_ns = 1000000000,
                                    .measured_state = pose(),
                                    .state_capture_time_ns = 1100000000},
                                   {.primary = target(1, 2),
                                    .capture_time_ns = 1120000000,
                                    .measured_state = pose(0.3F, 0.4F)}},
                                  1);
  ASSERT_EQ(commands.size(), 1U);
  EXPECT_EQ(CommandCapture::capture_time_at(0), 1120000000);
  EXPECT_FLOAT_EQ(commands[0].position[2], 0.3F);
}

TEST(StartUpHandover, WaitsForBothEnablesAndBlendsFromTheFreshPoseAtRelease) {
  const auto commands =
      run_graph({{.primary = target(1, 2), .capture_time_ns = 1000000000, .measured_state = pose()},
                 {.primary = target(1, 2),
                  .capture_time_ns = 2000000000,
                  .measured_state = pose(),
                  .enable_lower = true},
                 {.primary = target(1, 2),
                  .capture_time_ns = 3000000000,
                  .measured_state = pose(),
                  .enable_upper = false},
                 {.primary = target(1, 2),
                  .capture_time_ns = 4000000000,
                  .measured_state = pose(0.2F, 0.4F),
                  .enable_upper = true},
                 {.primary = target(1, 2), .capture_time_ns = 4500000000},
                 {.primary = target(1, 2), .capture_time_ns = 5000000000},
                 {.primary = target(3, 4), .capture_time_ns = 5500000000}},
                7, 1.0, true);
  ASSERT_EQ(commands.size(), 7U);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(commands[i].position[0], target(1, 2).position[0]);
    EXPECT_FLOAT_EQ(commands[i].position[2], 0.0F);
    EXPECT_FLOAT_EQ(commands[i].position[3], 0.0F);
    EXPECT_FLOAT_EQ(commands[i].velocity[2], 0.0F);
    EXPECT_FLOAT_EQ(commands[i].effort[3], 0.0F);
  }
  EXPECT_EQ(CommandCapture::capture_time_at(3), 4000000000);
  EXPECT_FLOAT_EQ(commands[3].position[2], 0.2F);
  EXPECT_FLOAT_EQ(commands[3].position[3], 0.4F);
  EXPECT_NEAR(commands[4].position[2], 0.6F, 1e-6F);
  EXPECT_NEAR(commands[4].position[3], 1.2F, 1e-6F);
  EXPECT_FLOAT_EQ(commands[5].position[2], 1.0F);
  EXPECT_FLOAT_EQ(commands[6].position[3], 4.0F);
}

TEST(StartUpHandover, UpperEnableRepeatedlyTogglesHoldWithoutInterruptingLowerBody) {
  std::vector<Tick> ticks;
  for (int i = 0; i <= 170; ++i) {
    Tick tick{.primary = i < 3 ? target(0.6F, 1.2F) : target(1, 2),
              .capture_time_ns = 1000000000 + i * 100000000LL};
    if (i == 0) {
      tick.measured_state = pose(0.2F, 0.4F);
      tick.enable_lower = true;
      tick.enable_upper = true;
    }
    if (i == 4 || i == 60 || i == 110) tick.enable_upper = true;
    if (i == 5) {
      tick.enable_lower = false;  // Releasing the buttons must not revoke either enable.
      tick.enable_upper = false;
    }
    ticks.push_back(std::move(tick));
  }
  const auto commands = run_graph(ticks, ticks.size(), 0.2, true);
  ASSERT_EQ(commands.size(), ticks.size());
  EXPECT_FLOAT_EQ(commands[2].position[3], 1.2F);
  EXPECT_FLOAT_EQ(commands[3].position[3], 2.0F);
  EXPECT_NEAR(commands[4].position[3], 1.99F, 1e-6F);  // Slew from the latest output.
  EXPECT_FLOAT_EQ(commands[59].position[2], 0.2F);
  EXPECT_FLOAT_EQ(commands[59].position[3], 0.4F);
  EXPECT_FLOAT_EQ(commands[59].velocity[3], 0.0F);
  EXPECT_FLOAT_EQ(commands[59].effort[3], 0.0F);
  EXPECT_NEAR(commands[60].position[3], 0.41F, 1e-6F);
  EXPECT_EQ(commands[109].position, ticks[109].primary.position);
  EXPECT_EQ(commands[109].velocity, ticks[109].primary.velocity);
  EXPECT_EQ(commands[109].effort, ticks[109].primary.effort);
  EXPECT_FLOAT_EQ(commands.back().position[3], 0.4F);
  for (std::size_t i = 0; i < commands.size(); ++i) {
    SCOPED_TRACE(i);
    for (std::size_t joint = 0; joint < 2; ++joint) {
      EXPECT_FLOAT_EQ(commands[i].position[joint], ticks[i].primary.position[joint]);
      EXPECT_FLOAT_EQ(commands[i].velocity[joint], ticks[i].primary.velocity[joint]);
      EXPECT_FLOAT_EQ(commands[i].effort[joint], ticks[i].primary.effort[joint]);
    }
    EXPECT_EQ(commands[i].kp, ticks[i].primary.kp);
    EXPECT_EQ(commands[i].kd, ticks[i].primary.kd);
    EXPECT_EQ(CommandCapture::capture_time_at(i), ticks[i].capture_time_ns);
    if (i >= 4) {
      EXPECT_LE(std::abs(commands[i].position[3] - commands[i - 1].position[3]), 0.050001F);
    }
  }
}

TEST(StartUpHandover, UpperEnableCanInterruptInitialBlendAndResumeWithoutRestartingIt) {
  const auto commands =
      run_graph({{.primary = target(1, 2),
                  .capture_time_ns = 1000000000,
                  .measured_state = pose(),
                  .enable_lower = true,
                  .enable_upper = true},
                 {.primary = target(1, 2), .capture_time_ns = 1500000000},
                 {.primary = target(1, 2), .capture_time_ns = 1600000000, .enable_upper = true},
                 {.primary = target(1, 2), .capture_time_ns = 1700000000, .enable_upper = true},
                 {.primary = target(1, 2), .capture_time_ns = 2000000000}},
                5, 1.0, true);
  ASSERT_EQ(commands.size(), 5U);
  EXPECT_FLOAT_EQ(commands[1].position[3], 1.0F);
  EXPECT_NEAR(commands[2].position[3], 0.99F, 1e-6F);
  EXPECT_NEAR(commands[3].position[3], 1.01F, 1e-6F);
  EXPECT_NEAR(commands[4].position[3], 1.16F, 1e-6F);
}

TEST(StartUpHandover, HoldIgnoresDuplicateEnablesAndDoesNotRewindCommandTime) {
  const auto commands =
      run_graph({{.primary = target(1, 2),
                  .capture_time_ns = 1000000000,
                  .measured_state = pose(),
                  .enable_lower = true,
                  .enable_upper = true},
                 {.primary = target(1, 2), .capture_time_ns = 2000000000},
                 {.primary = target(1, 2), .capture_time_ns = 2100000000, .enable_upper = true},
                 {.primary = target(1, 2), .capture_time_ns = 2100000000, .enable_upper = true},
                 {.primary = target(1, 2), .capture_time_ns = 2050000000},
                 {.primary = target(1, 2), .capture_time_ns = 2200000000}},
                6, 1.0, true);
  ASSERT_EQ(commands.size(), 6U);
  EXPECT_NEAR(commands[2].position[3], 1.99F, 1e-6F);
  EXPECT_FLOAT_EQ(commands[3].position[3], commands[2].position[3]);
  EXPECT_FLOAT_EQ(commands[4].position[3], commands[2].position[3]);
  EXPECT_NEAR(commands[5].position[3], 1.97F, 1e-6F);
}

TEST(StartUpHandover, HoldStartsFromLastOutputAfterRepeatedOrBackwardCommandTime) {
  for (const std::int64_t capture_ns : {2000000000LL, 1900000000LL}) {
    SCOPED_TRACE(capture_ns);
    const auto commands = run_graph(
        {{.primary = target(0.1F, 0.2F),
          .capture_time_ns = 1000000000,
          .measured_state = pose(),
          .enable_lower = true,
          .enable_upper = true},
         {.primary = target(0.1F, 0.2F), .capture_time_ns = 2000000000},
         {.primary = target(0.3F, 0.4F), .capture_time_ns = capture_ns},
         {.primary = target(0.3F, 0.4F), .capture_time_ns = 2100000000, .enable_upper = true}},
        4, 1.0, true);
    ASSERT_EQ(commands.size(), 4U);
    EXPECT_FLOAT_EQ(commands[2].position[3], 0.4F);
    EXPECT_NEAR(commands[3].position[3], 0.39F, 1e-6F);
    EXPECT_FLOAT_EQ(commands[3].position[0], commands[2].position[0]);
  }
}

TEST(StartUpHandover, EarlyUpperEnableIsLatchedButCannotBypassFreshStateOrLowerEnable) {
  const auto commands = run_graph(
      {{.primary = target(1, 2), .capture_time_ns = 0, .enable_upper = true},
       {.primary = target(1, 2), .capture_time_ns = 1000000000, .measured_state = pose()},
       {.primary = target(1, 2), .capture_time_ns = 2000000000, .enable_lower = true},
       {.primary = target(1, 2), .capture_time_ns = 3000000000, .measured_state = pose(0.3F, 0.6F)},
       {.primary = target(1, 2), .capture_time_ns = 3500000000}},
      4, 1.0, true);
  ASSERT_EQ(commands.size(), 4U);
  EXPECT_FLOAT_EQ(commands[0].position[2], 0.0F);
  EXPECT_FLOAT_EQ(commands[1].position[2], 0.0F);
  EXPECT_EQ(CommandCapture::capture_time_at(2), 3000000000);
  EXPECT_FLOAT_EQ(commands[2].position[2], 0.3F);
  EXPECT_FLOAT_EQ(commands[2].position[3], 0.6F);
  EXPECT_NEAR(commands[3].position[2], 0.65F, 1e-6F);
}

TEST(StartUpHandover, DegradedCommandsKeepLowerBodyFlowingWithoutSpendingTheBlend) {
  const auto commands = run_graph(
      {{.primary = target(1, 2),
        .primary_flags = holoscan::SampleFlags::kDegraded,
        .capture_time_ns = 1000000000,
        .measured_state = pose(0.1F, 0.2F)},
       {.primary = target(2, 4),
        .primary_flags = holoscan::SampleFlags::kDegraded,
        .capture_time_ns = 4000000000,
        .measured_state = pose(0.2F, 0.4F)},
       {.primary = target(1, 2), .capture_time_ns = 5000000000, .measured_state = pose(0.3F, 0.6F)},
       {.primary = target(1, 2), .capture_time_ns = 5500000000},
       {.primary = target(1, 2), .capture_time_ns = 6000000000}},
      5);
  ASSERT_EQ(commands.size(), 5U);
  for (std::size_t i = 0; i < commands.size(); ++i) {
    EXPECT_FLOAT_EQ(commands[i].position[0], target(1, 2).position[0]);
    EXPECT_FLOAT_EQ(commands[i].velocity[0], target(1, 2).velocity[0]);
    EXPECT_FLOAT_EQ(commands[i].effort[0], target(1, 2).effort[0]);
    EXPECT_EQ(commands[i].kp, target(1, 2).kp);
    EXPECT_EQ(commands[i].kd, target(1, 2).kd);
  }
  EXPECT_FLOAT_EQ(commands[0].position[2], 0.1F);
  EXPECT_FLOAT_EQ(commands[1].position[2], 0.1F);
  EXPECT_EQ(CommandCapture::flags_at(1), holoscan::SampleFlags::kDegraded);
  EXPECT_FLOAT_EQ(commands[2].position[2], 0.3F);
  EXPECT_NEAR(commands[3].position[2], 0.65F, 1e-6F);
  EXPECT_FLOAT_EQ(commands[4].position[2], 1.0F);
}

TEST(StartUpHandover, EnablesCannotStartTheBlendWhileCommandsAreDegraded) {
  const auto commands =
      run_graph({{.primary = target(1, 2),
                  .primary_flags = holoscan::SampleFlags::kDegraded,
                  .capture_time_ns = 1000000000,
                  .measured_state = pose(),
                  .enable_lower = true,
                  .enable_upper = true},
                 {.primary = target(1, 2),
                  .primary_flags = holoscan::SampleFlags::kDegraded,
                  .capture_time_ns = 4000000000,
                  .measured_state = pose()},
                 {.primary = target(1, 2), .capture_time_ns = 5000000000, .measured_state = pose()},
                 {.primary = target(1, 2), .capture_time_ns = 5500000000}},
                4, 1.0, true);
  ASSERT_EQ(commands.size(), 4U);
  EXPECT_FLOAT_EQ(commands[0].position[2], 0.0F);
  EXPECT_FLOAT_EQ(commands[1].position[2], 0.0F);
  EXPECT_FLOAT_EQ(commands[2].position[2], 0.0F);
  EXPECT_FLOAT_EQ(commands[3].position[2], 0.5F);
}

}  // namespace
}  // namespace isaac::os::ops::handover_test
