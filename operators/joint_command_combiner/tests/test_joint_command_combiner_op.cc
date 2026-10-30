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

#include "joint_command_cpu_schema_traits.h"
#include "operators/joint_command_combiner/joint_command_combiner_op.h"

namespace isaac::os::ops {
namespace {

using namespace std::chrono_literals;

constexpr std::int64_t kKnownCaptureTimeNs = 42;
constexpr std::uint64_t kKnownFrameId = 7U;
constexpr holoscan::TraceId kKnownTraceId = 11U;

struct Tick {
  isaac::JointCommandCpuT primary;
  std::optional<isaac::JointCommandCpuT> overlay;
  holoscan::SampleFlags primary_flags{holoscan::SampleFlags::kNone};
  holoscan::SampleFlags overlay_flags{holoscan::SampleFlags::kNone};
};

isaac::JointCommandCpuT make_command(float base, std::size_t count = 4U) {
  isaac::JointCommandCpuT command;
  command.position.resize(count);
  command.velocity.resize(count);
  command.effort.resize(count);
  command.kp.resize(count);
  command.kd.resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    const float offset = static_cast<float>(index);
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
    spec.output(primary, "out_primary_joint_command");
    spec.output(overlay, "out_overlay_joint_command");
  }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 10ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    const Tick& tick = ticks_[std::min(index_, ticks_.size() - 1U)];
    if (tick.overlay) {
      auto emitted = overlay.emit(
          *tick.overlay,
          holoscan::EmitOptions{.capture_time = context.clock().at_ns(kKnownCaptureTimeNs),
                                .frame_id = kKnownFrameId,
                                .trace = holoscan::TraceContext::root(kKnownTraceId),
                                .flags = tick.overlay_flags});
      if (!emitted) {
        return emitted;
      }
    }
    ++index_;
    return primary.emit(
        tick.primary,
        holoscan::EmitOptions{.capture_time = context.clock().at_ns(kKnownCaptureTimeNs),
                              .frame_id = kKnownFrameId,
                              .trace = holoscan::TraceContext::root(kKnownTraceId),
                              .flags = tick.primary_flags});
  }

  holoscan::Output<isaac::JointCommandCpuT> primary;
  holoscan::Output<isaac::JointCommandCpuT> overlay;

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

JointCommandCombinerParams make_params(double startup_delay_s = 0.0) {
  return {
      .joint_order = {"j0", "j1", "j2", "j3"},
      .joint_names_from_primary = {"j0", "j1"},
      .joint_names_from_overlay = {"j2", "j3"},
      .overlay_startup_delay_s = startup_delay_s,
  };
}

std::vector<isaac::JointCommandCpuT> run_graph(std::vector<Tick> ticks, std::size_t expected_count,
                                               JointCommandCombinerParams params = make_params()) {
  CommandCapture::reset();
  holoscan::Graph graph{"joint-command-combiner-test"};
  const auto source = graph.op<CommandSource>("source", std::move(ticks));
  auto combiner = graph.op<JointCommandCombinerOp>("combiner");
  const auto sink = graph.op<CommandSink>("sink");
  combiner.set_params(std::move(params));
  graph.add_flow(source->primary, combiner->primary_command);
  graph.add_flow(source->overlay, combiner->overlay_command);
  graph.add_flow(combiner->command, sink->command);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  EXPECT_TRUE(plan.ok()) << plan.json();
  if (!plan.ok()) {
    return {};
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  auto commands = CommandCapture::wait_for(expected_count);
  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kStopped);
  return commands;
}

void expect_invalid_params(JointCommandCombinerParams params) {
  holoscan::Graph graph{"invalid-joint-partition-test"};
  const auto source =
      graph.op<CommandSource>("source", std::vector<Tick>{{.primary = make_command(1.0F)}});
  auto combiner = graph.op<JointCommandCombinerOp>("combiner");
  const auto sink = graph.op<CommandSink>("sink");
  combiner.set_params(std::move(params));
  graph.add_flow(source->primary, combiner->primary_command);
  graph.add_flow(source->overlay, combiner->overlay_command);
  graph.add_flow(combiner->command, sink->command);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  holoscan::RunSession session = holoscan::run_async(plan);
  session.wait();
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kFailed);
}

void expect_command_field_sources(const isaac::JointCommandCpuT& command, float primary_base,
                                  float overlay_base) {
  ASSERT_EQ(command.position.size(), 4U);
  for (std::size_t index = 0; index < 4U; ++index) {
    const float source = index < 2U ? primary_base : overlay_base;
    EXPECT_FLOAT_EQ(command.position[index], source + static_cast<float>(index));
    EXPECT_FLOAT_EQ(command.velocity[index], source + 10.0F + static_cast<float>(index));
    EXPECT_FLOAT_EQ(command.effort[index], source + 20.0F + static_cast<float>(index));
    EXPECT_FLOAT_EQ(command.kp[index], source + 30.0F + static_cast<float>(index));
    EXPECT_FLOAT_EQ(command.kd[index], source + 40.0F + static_cast<float>(index));
  }
}

TEST(JointCommandCombiner, UsesPrimaryUntilOverlayArrivesThenKeepsLatest) {
  const auto commands = run_graph(
      {
          {.primary = make_command(1.0F)},
          {.primary = make_command(2.0F), .overlay = make_command(100.0F)},
          {.primary = make_command(3.0F)},
          {.primary = make_command(4.0F), .overlay = make_command(200.0F)},
      },
      4U);

  ASSERT_EQ(commands.size(), 4U);
  expect_command_field_sources(commands[0], 1.0F, 1.0F);
  expect_command_field_sources(commands[1], 2.0F, 100.0F);
  expect_command_field_sources(commands[2], 3.0F, 100.0F);
  expect_command_field_sources(commands[3], 4.0F, 200.0F);
  EXPECT_EQ(CommandCapture::flags_at(0U), holoscan::SampleFlags::kDegraded);
  EXPECT_EQ(CommandCapture::flags_at(1U), holoscan::SampleFlags::kNone);
  EXPECT_EQ(CommandCapture::flags_at(2U), holoscan::SampleFlags::kNone);
}

TEST(JointCommandCombiner, KeepsLastValidOverlayCommand) {
  auto invalid_overlay = make_command(200.0F);
  invalid_overlay.kd.pop_back();
  const auto commands = run_graph(
      {
          {.primary = make_command(1.0F), .overlay = make_command(100.0F)},
          {.primary = make_command(2.0F), .overlay = std::move(invalid_overlay)},
      },
      2U);

  ASSERT_EQ(commands.size(), 2U);
  expect_command_field_sources(commands[0], 1.0F, 100.0F);
  expect_command_field_sources(commands[1], 2.0F, 100.0F);
}

TEST(JointCommandCombiner, RejectsInvalidOverlayMetadata) {
  const auto commands = run_graph(
      {
          {.primary = make_command(1.0F), .overlay = make_command(100.0F)},
          {.primary = make_command(2.0F),
           .overlay = make_command(200.0F),
           .overlay_flags = holoscan::SampleFlags::kInvalid},
          {.primary = make_command(3.0F)},
      },
      3U);

  ASSERT_EQ(commands.size(), 3U);
  expect_command_field_sources(commands[0], 1.0F, 100.0F);
  expect_command_field_sources(commands[1], 2.0F, 100.0F);
  expect_command_field_sources(commands[2], 3.0F, 100.0F);
}

TEST(JointCommandCombiner, DoesNotEmitAnInvalidPrimaryCommand) {
  auto invalid_primary = make_command(1.0F);
  invalid_primary.position.pop_back();
  const auto commands = run_graph(
      {
          {.primary = make_command(0.0F), .primary_flags = holoscan::SampleFlags::kInvalid},
          {.primary = std::move(invalid_primary)},
          {.primary = make_command(2.0F)},
      },
      1U);

  ASSERT_EQ(commands.size(), 1U);
  expect_command_field_sources(commands[0], 2.0F, 2.0F);
}

TEST(JointCommandCombiner, StartupDelayLeavesAllJointsOnPrimary) {
  const auto commands = run_graph(
      {{.primary = make_command(1.0F), .overlay = make_command(100.0F)}}, 1U, make_params(3600.0));

  ASSERT_EQ(commands.size(), 1U);
  expect_command_field_sources(commands[0], 1.0F, 1.0F);
  EXPECT_EQ(CommandCapture::flags_at(0U), holoscan::SampleFlags::kDegraded);
}

TEST(JointCommandCombiner, InvalidFirstOverlayDoesNotClearFallbackDegradation) {
  auto short_overlay = make_command(100.0F);
  short_overlay.position.pop_back();
  const auto commands =
      run_graph({{.primary = make_command(1.0F), .overlay = short_overlay},
                 {.primary = make_command(2.0F),
                  .overlay = make_command(100.0F),
                  .overlay_flags = holoscan::SampleFlags::kInvalid},
                 {.primary = make_command(3.0F), .overlay = make_command(100.0F)}},
                3U);
  ASSERT_EQ(commands.size(), 3U);
  EXPECT_EQ(CommandCapture::flags_at(0U), holoscan::SampleFlags::kDegraded);
  EXPECT_EQ(CommandCapture::flags_at(1U), holoscan::SampleFlags::kDegraded);
  EXPECT_EQ(CommandCapture::flags_at(2U), holoscan::SampleFlags::kNone);
}

TEST(JointCommandCombiner, PrimaryOnlyConfigurationDoesNotRequireAnOverlay) {
  auto params = make_params();
  params.joint_names_from_primary = params.joint_order;
  params.joint_names_from_overlay.clear();
  const auto commands = run_graph({{.primary = make_command(1.0F)}}, 1U, params);
  ASSERT_EQ(commands.size(), 1U);
  EXPECT_EQ(CommandCapture::flags_at(0U), holoscan::SampleFlags::kNone);
}

TEST(JointCommandCombiner, PropagatesPrimaryMetadata) {
  const auto commands = run_graph(
      {{.primary = make_command(1.0F), .primary_flags = holoscan::SampleFlags::kDegraded}}, 1U);

  ASSERT_EQ(commands.size(), 1U);
  EXPECT_EQ(CommandCapture::capture_time_at(0U), kKnownCaptureTimeNs);
  EXPECT_EQ(CommandCapture::frame_id_at(0U), kKnownFrameId);
  EXPECT_EQ(CommandCapture::trace_id_at(0U), kKnownTraceId);
  EXPECT_EQ(CommandCapture::flags_at(0U), holoscan::SampleFlags::kDegraded);
}

TEST(JointCommandCombiner, PreservesOverlayDegradedFlag) {
  const auto commands = run_graph({{.primary = make_command(1.0F),
                                    .overlay = make_command(100.0F),
                                    .overlay_flags = holoscan::SampleFlags::kDegraded}},
                                  1U);

  ASSERT_EQ(commands.size(), 1U);
  EXPECT_EQ(CommandCapture::flags_at(0U), holoscan::SampleFlags::kDegraded);
}

TEST(JointCommandCombiner, RejectsAnInvalidJointPartition) {
  auto params = make_params();
  params.joint_names_from_overlay = {"j1", "j2", "j3"};
  expect_invalid_params(std::move(params));
}

TEST(JointCommandCombiner, RejectsDuplicateJointNames) {
  auto params = make_params();
  params.joint_names_from_primary = {"j0", "j0", "j1"};
  expect_invalid_params(std::move(params));
}

}  // namespace
}  // namespace isaac::os::ops
