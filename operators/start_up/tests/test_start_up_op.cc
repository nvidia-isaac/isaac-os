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
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "operators/start_up/start_up_op.h"
#include "operators/start_up/start_up_params_config_codec.h"

namespace isaac::os::ops {
namespace {

using namespace std::chrono_literals;

constexpr std::size_t kStartupJointCount = 3U;
constexpr float kControllerPosition = 9.0F;
const std::vector<float> kFinalPose{1.0F, 2.0F, 3.0F};
constexpr auto kTickPeriod = 5ms;

StartUpParams make_params(double auto_enable_delay,
                          std::vector<std::string> upper_body_joints = {}) {
  StartUpParams params;
  params.sdk_joint_order = {"joint_0", "joint_1", "joint_2"};
  params.kp = {{".*", 20.0}};
  params.kd = {{".*", 2.0}};
  params.hold_kp = {{".*", 60.0}};
  params.hold_kd = {{".*", 4.0}};
  params.upper_body_joints = std::move(upper_body_joints);
  params.auto_enable_delay = auto_enable_delay;
  params.control_dt = 0.005;
  params.dwell_duration = 0.005;
  params.vel_settled_threshold = 0.01;
  params.waypoints = {StartUpWaypoint{
      .name = "stand",
      .joint_positions = {{"joint_0", 1.0}, {"joint_1", 2.0}, {"joint_2", 3.0}},
      .duration = 0.015,
  }};
  return params;
}

isaac::JointCommandCpuT make_controller_command() {
  isaac::JointCommandCpuT value;
  value.position.assign(kStartupJointCount, kControllerPosition);
  value.velocity.assign(kStartupJointCount, 1.0F);
  value.effort.assign(kStartupJointCount, 2.0F);
  value.kp.assign(kStartupJointCount, 5.0F);
  value.kd.assign(kStartupJointCount, 0.5F);
  return value;
}

class ControllerSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kTickPeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto now = context.clock().now();
    if (!now) {
      return holoscan::make_unexpected(std::move(now).error());
    }
    auto captured = *now;
    captured.timestamp_ns -= 1000000;
    return output.emit(
        make_controller_command(),
        holoscan::EmitOptions{.capture_time = captured,
                              .frame_id = static_cast<std::uint64_t>(captured.timestamp_ns),
                              .trace = holoscan::TraceContext::root(42U),
                              .flags = holoscan::SampleFlags::kDegraded});
  }

  holoscan::Output<isaac::JointCommandCpuT> output;
};

class StateSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kTickPeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    isaac::JointStateCpuT state;
    state.position.assign(kStartupJointCount, 0.0F);
    state.velocity.assign(kStartupJointCount, 0.0F);
    auto now = context.clock().now();
    if (!now) {
      return holoscan::make_unexpected(std::move(now).error());
    }
    return output.emit(std::move(state), holoscan::EmitOptions{.capture_time = *now});
  }

  holoscan::Output<isaac::JointStateCpuT> output;
};

class EnableSource final : public holoscan::Operator<> {
 public:
  EnableSource(bool enable_upper, bool enable_lower)
      : enable_upper_(enable_upper), enable_lower_(enable_lower) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(lower, "out_lower");
    spec.output(upper, "out_upper");
  }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kTickPeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    if (sent_) {
      return {};
    }
    sent_ = true;
    auto now = context.clock().now();
    if (!now) {
      return holoscan::make_unexpected(std::move(now).error());
    }
    const holoscan::EmitOptions options{.capture_time = *now};
    auto emitted = lower.emit(enable_lower_, options);
    if (!emitted) {
      return emitted;
    }
    if (!enable_upper_) {
      return {};
    }
    return upper.emit(true, options);
  }

  holoscan::Output<bool> lower;
  holoscan::Output<bool> upper;

 private:
  bool enable_upper_;
  bool enable_lower_;
  bool sent_{};
};

struct CommandCapture {
  static void reset() {
    const std::lock_guard lock(mutex);
    commands.clear();
    metadata.clear();
  }

  static void record(isaac::JointCommandCpuT value, holoscan::SampleMetadata sample_metadata) {
    {
      const std::lock_guard lock(mutex);
      commands.push_back(std::move(value));
      metadata.push_back(sample_metadata);
    }
    ready.notify_all();
  }

  static bool wait_for(std::size_t count) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 5s, [count] { return commands.size() >= count; });
  }
  static bool wait_for_passthrough(std::size_t count = 1U) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 5s, [count] {
      const auto matching = std::ranges::count_if(commands, [](const auto& value) {
        return !value.position.empty() && value.position[0] == kControllerPosition;
      });
      return static_cast<std::size_t>(matching) >= count;
    });
  }

  static std::vector<isaac::JointCommandCpuT> snapshot() {
    const std::lock_guard lock(mutex);
    return commands;
  }

  static std::vector<holoscan::SampleMetadata> metadata_snapshot() {
    const std::lock_guard lock(mutex);
    return metadata;
  }

  static inline std::vector<holoscan::SampleMetadata> metadata;
  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<isaac::JointCommandCpuT> commands;
};

class CommandSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{input});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = input.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    CommandCapture::record(std::move(received->data), received->metadata);
    return {};
  }

  holoscan::Input<isaac::JointCommandCpuT> input;
};

struct RunningGraph {
  holoscan::Graph graph{"start-up-test"};
  holoscan::ExecutionPlan plan;
  std::unique_ptr<holoscan::RunSession> session;
  void report_failure() const {
    if (session == nullptr || session->termination() == holoscan::RunTermination::kStopped) {
      return;
    }
    const auto vertices = plan.graph().vertices();
    for (const holoscan::Diagnostic& diagnostic : session->diagnostics()) {
      std::string_view name{"unknown"};
      for (const auto& vertex : vertices) {
        if (vertex.id == diagnostic.source.vertex) {
          name = vertex.name;
          break;
        }
      }
      ADD_FAILURE() << diagnostic.token << " at " << name;
    }
  }
};

template <typename Params>
std::unique_ptr<RunningGraph> run_graph(Params params, bool with_enable_source,
                                        bool enable_upper = true, bool enable_lower = true) {
  CommandCapture::reset();
  auto running = std::make_unique<RunningGraph>();
  auto& graph = running->graph;
  const auto controller = graph.op<ControllerSource>("controller");
  const auto state = graph.op<StateSource>("state");
  auto start_up = graph.op<StartUpOp>("start_up", with_enable_source);
  const auto sink = graph.op<CommandSink>("sink");
  start_up.set_params(std::move(params));
  const holoscan::ConnectionOptions latest_sample{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  graph.add_flow(controller->output, start_up->joint_command);
  graph.add_flow(state->output, start_up->joint_state, latest_sample);
  graph.add_flow(start_up->command, sink->input);
  graph.partition("controller").add(controller);
  graph.partition("state").add(state);
  graph.partition("control").add(start_up).add(sink);
  if (with_enable_source) {
    const auto enable = graph.op<EnableSource>("enable", enable_upper, enable_lower);
    graph.add_flow(enable->lower, start_up->enable_lower, latest_sample);
    graph.add_flow(enable->upper, start_up->enable_upper, latest_sample);
    graph.partition("enable").add(enable);
  }
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  running->plan = holoscan::compile(graph);
  EXPECT_TRUE(running->plan.ok()) << running->plan.json();
  if (running->plan.ok()) {
    running->session = std::make_unique<holoscan::RunSession>(holoscan::run_async(running->plan));
  }
  return running;
}

TEST(StartUp, CommandsItsOwnPoseBeforeTheRobotIsEnabled) {
  auto running = run_graph(make_params(0.0), true);
  ASSERT_NE(running->session, nullptr);
  ASSERT_TRUE(CommandCapture::wait_for(3U));
  const auto commands = CommandCapture::snapshot();
  running->session->request_stop();
  running->session->wait();

  ASSERT_FALSE(commands.empty());
  for (const auto& value : commands) {
    ASSERT_EQ(value.position.size(), kStartupJointCount);
    EXPECT_NE(value.position[0], kControllerPosition);
    EXPECT_FLOAT_EQ(value.velocity[0], 0.0F);
    EXPECT_FLOAT_EQ(value.effort[0], 0.0F);
  }
}

TEST(StartUp, PassesTheControllerCommandOnceItIsEnabled) {
  auto running = run_graph(make_params(0), false);
  ASSERT_NE(running->session, nullptr);
  const bool reached = CommandCapture::wait_for_passthrough();
  running->session->request_stop();
  running->session->wait();
  running->report_failure();

  EXPECT_TRUE(reached);
  EXPECT_EQ(running->session->termination(), holoscan::RunTermination::kStopped);
}

TEST(StartUp, WalksToTheFinalPoseBeforeHandingOver) {
  auto running = run_graph(make_params(0), false);
  ASSERT_NE(running->session, nullptr);
  ASSERT_TRUE(CommandCapture::wait_for_passthrough());
  const auto commands = CommandCapture::snapshot();
  running->session->request_stop();
  running->session->wait();

  const auto handover = std::ranges::find_if(commands, [](const auto& value) {
    return !value.position.empty() && value.position[0] == kControllerPosition;
  });
  ASSERT_NE(handover, commands.begin());
  const auto& settled = *std::prev(handover);
  EXPECT_FLOAT_EQ(settled.position[0], kFinalPose[0]);
  EXPECT_FLOAT_EQ(settled.position[2], kFinalPose[2]);
  EXPECT_FLOAT_EQ(settled.kp[0], 60.0F);
}

TEST(StartUp, HoldsTheUpperBodyAfterTheLowerBodyIsEnabled) {
  auto running = run_graph(make_params(0.0, {"joint_2"}), true, /*enable_upper=*/false);
  ASSERT_NE(running->session, nullptr);
  ASSERT_TRUE(CommandCapture::wait_for_passthrough(10U));
  running->session->request_stop();
  running->session->wait();
  const auto commands = CommandCapture::snapshot();

  const auto handover = std::ranges::find_if(commands, [](const auto& value) {
    return !value.position.empty() && value.position[0] == kControllerPosition;
  });
  ASSERT_NE(handover, commands.end());
  for (auto command = handover; command != commands.end(); ++command) {
    EXPECT_FLOAT_EQ(command->position[0], kControllerPosition);
    EXPECT_FLOAT_EQ(command->position[2], kFinalPose[2]);
    EXPECT_FLOAT_EQ(command->velocity[2], 0.0F);
    EXPECT_FLOAT_EQ(command->effort[2], 0.0F);
  }
}

TEST(StartUp, PreservesRunningCommandMetadataWithAndWithoutUpperHold) {
  for (const bool hold_upper : {false, true}) {
    auto running = run_graph(make_params(0.0, hold_upper ? std::vector<std::string>{"joint_2"}
                                                         : std::vector<std::string>{}),
                             true, /*enable_upper=*/false);
    ASSERT_NE(running->session, nullptr);
    const bool received = CommandCapture::wait_for_passthrough(3U);
    running->session->request_stop();
    running->session->wait();
    ASSERT_TRUE(received);
    const auto commands = CommandCapture::snapshot();
    const auto metadata = CommandCapture::metadata_snapshot();
    ASSERT_EQ(commands.size(), metadata.size());
    for (std::size_t i = 0; i < commands.size(); ++i) {
      if (commands[i].position[0] != kControllerPosition) {
        continue;
      }
      SCOPED_TRACE(i);
      EXPECT_TRUE(metadata[i].capture_time().valid());
      EXPECT_EQ(metadata[i].capture_timestamp_ns, static_cast<std::int64_t>(metadata[i].frame_id));
      EXPECT_EQ(metadata[i].trace_id, 42U);
      EXPECT_EQ(metadata[i].flags, holoscan::SampleFlags::kDegraded);
      EXPECT_FLOAT_EQ(commands[i].position[2], hold_upper ? kFinalPose[2] : kControllerPosition);
    }
  }
}

TEST(StartUp, ExpandsNativePatternsAndPreservesZeroAndFallbackGains) {
  const auto document = holoscan::config_io::parse(R"(
start_up:
  sdk_joint_order: [joint_0, joint_1, joint_2]
  kp: {'.*': 20.0, 'joint_2': 0.0}
  kd: {'.*': 0.0}
  control_dt: 0.005
  dwell_duration: 0.005
  auto_enable_delay: 60.0
  vel_settled_threshold: 0.01
  waypoints:
    - name: stand
      duration: 0.015
      joint_positions: {'.*': 1.0, 'joint_1': 2.0, 'joint_2': 3.0}
)",
                                                   holoscan::config_io::Format::kYaml);
  auto parameters = holoscan::params::decode<StartUpParams>(document, "/start_up");
  auto running = run_graph(std::move(parameters), false);
  ASSERT_NE(running->session, nullptr);
  const bool received = CommandCapture::wait_for(8U);
  running->session->request_stop();
  running->session->wait();
  ASSERT_TRUE(received);
  const auto commands = CommandCapture::snapshot();
  ASSERT_GE(commands.size(), 8U);
  EXPECT_NE(commands[2].position, kFinalPose);
  EXPECT_EQ(commands[3].position, kFinalPose);
  for (const auto& command : commands) {
    EXPECT_EQ(command.kp, (std::vector<float>{20.0F, 20.0F, 0.0F}));
    EXPECT_EQ(command.kd, (std::vector<float>{0.0F, 0.0F, 0.0F}));
  }
}

TEST(StartUp, RejectsMissingGainMaps) {
  for (const bool omit_kp : {false, true}) {
    auto params = make_params(0);
    (omit_kp ? params.kp : params.kd).reset();
    auto running = run_graph(std::move(params), false);
    ASSERT_NE(running->session, nullptr);
    running->session->wait();
    EXPECT_EQ(running->session->termination(), holoscan::RunTermination::kFailed);
    EXPECT_TRUE(CommandCapture::snapshot().empty());
  }
}

TEST(StartUp, RejectsUnmatchedPatternsAndInvalidDurations) {
  for (int invalid = 0; invalid < 4; ++invalid) {
    auto params = make_params(0);
    switch (invalid) {
      case 0:
        params.kp->emplace("unknown_joint", 1.0);
        break;
      case 1:
        params.upper_body_joints = {"unknown_joint"};
        break;
      case 2:
        params.control_dt = 0.0;
        break;
      case 3:
        params.waypoints[0].duration = -1.0;
        break;
    }
    auto running = run_graph(std::move(params), false);
    ASSERT_NE(running->session, nullptr);
    running->session->wait();
    EXPECT_EQ(running->session->termination(), holoscan::RunTermination::kFailed);
  }
}

TEST(StartUp, IgnoresAutomaticEnableDelayWhenEnableSignalsAreConnected) {
  for (const double delay :
       {0.0, -1.0, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::max()}) {
    SCOPED_TRACE(delay);
    auto running = run_graph(make_params(delay), true, false, false);
    ASSERT_NE(running->session, nullptr);
    const bool reached = CommandCapture::wait_for(20U);
    running->session->request_stop();
    running->session->wait();
    ASSERT_TRUE(reached);
    const auto commands = CommandCapture::snapshot();
    EXPECT_EQ(commands.back().position, kFinalPose);
    for (const auto& command : commands) {
      EXPECT_NE(command.position[0], kControllerPosition);
    }
  }
}

TEST(StartUp, DelaysAutomaticEnableFromTheFinalPoseAndReleasesBothBodies) {
  auto running = run_graph(make_params(0.05, {"joint_2"}), false);
  ASSERT_NE(running->session, nullptr);
  const bool reached = CommandCapture::wait_for_passthrough(10U);
  running->session->request_stop();
  running->session->wait();
  ASSERT_TRUE(reached);
  const auto commands = CommandCapture::snapshot();
  const auto metadata = CommandCapture::metadata_snapshot();
  const auto final_pose = std::ranges::find_if(
      commands, [](const auto& command) { return command.position == kFinalPose; });
  const auto handover = std::ranges::find_if(
      commands, [](const auto& command) { return command.position[0] == kControllerPosition; });
  ASSERT_NE(final_pose, commands.end());
  ASSERT_NE(handover, commands.end());
  ASSERT_GT(handover, final_pose);
  // The last held command is emitted on the tick that expires the delay.
  EXPECT_GE(metadata[handover - commands.begin() - 1].capture_timestamp_ns -
                metadata[final_pose - commands.begin()].capture_timestamp_ns,
            50000000);
  EXPECT_GT(commands.back().position[2], kFinalPose[2]);
}

TEST(StartUp, RejectsInvalidAutomaticEnableDelays) {
  for (const auto delay :
       {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::max()}) {
    auto running = run_graph(make_params(delay), false);
    ASSERT_NE(running->session, nullptr);
    running->session->wait();
    EXPECT_EQ(running->session->termination(), holoscan::RunTermination::kFailed);
    EXPECT_TRUE(CommandCapture::snapshot().empty());
  }
}

TEST(StartUp, RejectsAmbiguousHandoverConfigurationAndInvalidLimits) {
  for (int invalid = 0; invalid < 4; ++invalid) {
    auto params = make_params(0, {"joint_2"});
    params.handover_only = true;
    params.waypoints.clear();
    switch (invalid) {
      case 0:
        params.handover_duration = 0.0;
        break;
      case 1:
        params.handover_duration = -1.0;
        break;
      case 2:
        params.upper_body_joints.clear();
        break;
      case 3:
        params.waypoints = make_params(0).waypoints;
        break;
      default:
        break;
    }
    auto running = run_graph(std::move(params), false);
    ASSERT_NE(running->session, nullptr);
    running->session->wait();
    EXPECT_EQ(running->session->termination(), holoscan::RunTermination::kFailed);
    EXPECT_TRUE(CommandCapture::snapshot().empty());
  }
}

}  // namespace
}  // namespace isaac::os::ops
