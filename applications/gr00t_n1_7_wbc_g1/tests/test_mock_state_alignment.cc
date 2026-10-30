/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "applications/gr00t_n1_7_wbc_g1/mock/mock_operators.h"

namespace isaac::os::ops {
namespace {

class StateSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input"); }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = input.receive();
    if (!received) return holoscan::make_unexpected(received.error());
    return {};
  }
  holoscan::Input<isaac::JointStateCpuT> input;
};

class AlignedStateProbeOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(joint_state, "joint_state");
    spec.input(imu, "imu");
  }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    // Match WBC's contract without requiring inference or a GPU.
    result.trigger(holoscan::OnEach{joint_state});
    result.input(imu, holoscan::LatestWithin{.max_age = std::chrono::milliseconds{25}});
    result.on_missing(imu, holoscan::Skip{});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto state = joint_state.receive();
    if (!state) return holoscan::make_unexpected(state.error());
    auto inertial = imu.receive();
    if (!inertial) return holoscan::make_unexpected(inertial.error());
    ++count_;
    // Detect a skipped first tick, later gaps, and selection of the previous IMU.
    EXPECT_EQ(state->metadata.sequence, count_);
    EXPECT_EQ(inertial->metadata.sequence, count_);
    EXPECT_EQ(state->metadata.capture_time(), inertial->metadata.capture_time());
    if (count_ == 10U) context.request_completion();
    return {};
  }
  holoscan::Input<isaac::JointStateCpuT> joint_state;
  holoscan::Input<holoscan::schema::ImuT> imu;

 private:
  std::uint64_t count_{};
};

TEST(MockStateAlignment, MatchingImuIsAvailableForEveryJointState) {
  auto shutdown = holoscan::ManagedShutdownSignals::create();
  ASSERT_TRUE(shutdown);
  holoscan::Graph graph{"mock-state-alignment"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  auto source = graph.op<MockRobotStateOp>("source", 43U);
  auto probe = graph.op<AlignedStateProbeOp>("probe");
  auto sink = graph.op<StateSinkOp>("gr00t_sink");
  graph.add_flow(source->wbc_joint_state, probe->joint_state);
  graph.add_flow(source->imu, probe->imu);
  graph.add_flow(source->gr00t_joint_state, sink->input);
  graph.partition("wbc").add(source).add(probe).add(sink);
  const auto plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = holoscan::run_async(plan, *shutdown);
  session.wait();
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kRequestedCompleted);
}

}  // namespace
}  // namespace isaac::os::ops
