/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <functional>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <optional>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/sensor_io/sensor_schema_package.hpp"
#include "holoscan/time/realtime_clock.hpp"

#include "common/ros2_bridge/ros2_bridge.h"

namespace isaac::os::ops {
namespace {

class SinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(1U); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{in});
    return contract;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    return sample ? holoscan::expected<void, holoscan::Error>{}
                  : holoscan::make_unexpected(sample.error());
  }
  holoscan::Input<isaac::JointStateCpuT> in;
};

// Samples its input with LatestWithin, an event-time selector that needs a capture time.
class LatestWithinSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(1U); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(20)});
    contract.input(in, holoscan::LatestWithin{.max_age = std::chrono::milliseconds(100)});
    contract.on_missing(in, holoscan::SafeDefault{});
    return contract;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
  holoscan::Input<isaac::PoseCpuT> in;
};

class SourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out, "out"); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(20)});
    return contract;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return out.emit(isaac::JointStateCpuT{});
  }
  holoscan::Output<isaac::JointStateCpuT> out;
};

class ImageSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(1U); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{in});
    return contract;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    return sample ? holoscan::expected<void, holoscan::Error>{}
                  : holoscan::make_unexpected(sample.error());
  }
  holoscan::Input<holoscan::schema::ImageT> in;
};

class PoseSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(1U); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{in});
    return contract;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    return sample ? holoscan::expected<void, holoscan::Error>{}
                  : holoscan::make_unexpected(sample.error());
  }
  holoscan::Input<isaac::PoseCpuT> in;
};

class TwistSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out, "out"); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(20)});
    return contract;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return out.emit(isaac::TwistCpuT{});
  }
  holoscan::Output<isaac::TwistCpuT> out;
};

}  // namespace
}  // namespace isaac::os::ops

namespace isaac::os::ros2_bridge {
namespace {

// Authors a graph body through a caller-supplied callback so each test can wire differently.
class WiringGraph final : public holoscan::Graph {
 public:
  using Body = std::function<void(WiringGraph&)>;
  explicit WiringGraph(Body body) : Graph("wiring_test"), body_(std::move(body)) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    body_(*this);
  }

 private:
  Body body_;
};

TEST(Ros2BridgeWiring, CreatesAnImportedTopicOnTheFirstFromRos2) {
  Ros2Bridge<> bridge("wiring_test_node");
  WiringGraph graph([&](WiringGraph& self) {
    auto sink = self.op<ops::SinkOp>("sink");
    bridge.from_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
        self, sink->in, {.port_name = "joint_state", .ros2_topic = "/wiring_test_in"});
  });
  EXPECT_TRUE(holoscan::compile(graph).ok());
}

// Fan-out via from_ros2's memoization is graph-representable -- Graph::add_flow accepts a
// second edge from the same ImportedTopic to a different consumer without throwing, and
// memoization is what keeps that second call from re-importing "holomq://joint_state" and
// colliding on GRAPH_ENDPOINT_NAME_DUPLICATE instead. What from_ros2 cannot do is make the
// resulting plan actually compile: an ImportedTopic's inherited external-channel contract admits
// exactly one boundary-to-Operator relation, so fanning it out to two consumers is rejected at
// compile() with EXTERNAL_CHANNEL_PLAN_INVALID -- confirmed empirically against this repo's
// pinned Holoscan SDK below, not just asserted from a doc comment. (An older SDK generation
// rejected this same shape with PARTITION_MANUAL_AUTHORING_REQUIRED, fixable by binding a manual
// Graph::partition(); that diagnostic is superseded in the currently pinned SDK, and manual
// partitioning does not avoid EXTERNAL_CHANNEL_PLAN_INVALID either.)
TEST(Ros2BridgeWiring, RejectsFanOutWithoutAManuallyAuthoredPartition) {
  Ros2Bridge<> bridge("wiring_test_node");
  WiringGraph graph([&](WiringGraph& self) {
    auto first_sink = self.op<ops::SinkOp>("first_sink");
    auto second_sink = self.op<ops::SinkOp>("second_sink");
    bridge.from_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
        self, first_sink->in, {.port_name = "joint_state", .ros2_topic = "/wiring_test_in"});
    bridge.from_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
        self, second_sink->in, {.port_name = "joint_state", .ros2_topic = "/wiring_test_in"});
  });
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_FALSE(plan.ok());
  // Pins down *which* rejection this is, not just that compile() failed for some reason -- a
  // different diagnostic here would mean the doc comment above no longer describes reality.
  const auto diagnostics = plan.diagnostics();
  EXPECT_TRUE(std::any_of(
      diagnostics.begin(), diagnostics.end(),
      [](const auto& diagnostic) { return diagnostic.token == "EXTERNAL_CHANNEL_PLAN_INVALID"; }))
      << "Expected an EXTERNAL_CHANNEL_PLAN_INVALID diagnostic among " << diagnostics.size()
      << " diagnostic(s).";
}

// PoseStamped carries header.stamp, so a spec naming a capture clock makes the imported topic
// require capture time and an event-time selector downstream compiles. Without the clock the
// import has no capture time and the same graph is rejected.
TEST(Ros2BridgeWiring, LatestWithinOnAnImportNeedsACaptureTimeClock) {
  const auto compile_with = [](std::optional<holoscan::ClockKey> clock) {
    Ros2Bridge<> bridge("wiring_test_node");
    WiringGraph graph([&](WiringGraph& self) {
      auto sink = self.op<ops::LatestWithinSinkOp>("sink");
      bridge.from_ros2<geometry_msgs::msg::PoseStamped, isaac::PoseCpuT>(
          self, sink->in,
          {.port_name = "pose",
           .ros2_topic = "/wiring_test_pose",
           .capture_time_clock = std::move(clock)});
    });
    return holoscan::compile(graph);
  };
  EXPECT_TRUE(compile_with(holoscan::ClockKey::named("clock")).ok());
  const holoscan::ExecutionPlan plan = compile_with(std::nullopt);
  ASSERT_FALSE(plan.ok());
  const auto diagnostics = plan.diagnostics();
  EXPECT_TRUE(std::any_of(diagnostics.begin(), diagnostics.end(), [](const auto& diagnostic) {
    return diagnostic.token == "TEMPORAL_CAPTURE_TIME_REQUIRED";
  }));
}

// A type with no header.stamp can still declare a capture clock: the bridge stamps each message
// with its receive time, so a DrainAll consumer binds on the import.
TEST(Ros2BridgeWiring, AcceptsACaptureTimeClockOnATypeWithoutAHeader) {
  Ros2Bridge<> bridge("wiring_test_node");
  WiringGraph graph([&](WiringGraph& self) {
    auto sink = self.op<ops::PoseSinkOp>("sink");
    bridge.from_ros2<geometry_msgs::msg::Pose, isaac::PoseCpuT>(
        self, sink->in,
        {.port_name = "pose",
         .ros2_topic = "/wiring_test_pose",
         .capture_time_clock = holoscan::ClockKey::named("clock")});
  });
  EXPECT_TRUE(holoscan::compile(graph).ok());
}

TEST(Ros2BridgeWiring, CreatesAnExportedTopicOnTheFirstToRos2) {
  Ros2Bridge<> bridge("wiring_test_node");
  WiringGraph graph([&](WiringGraph& self) {
    auto source = self.op<ops::SourceOp>("source");
    bridge.to_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
        self, source->out, {.port_name = "joint_command", .ros2_topic = "/wiring_test_out"});
  });
  EXPECT_TRUE(holoscan::compile(graph).ok());
}

// Internally, Holoscan rejects a second edge into an already-targeted Input regardless of
// source (Graph::add_flow's own GRAPH_INPUT_FANIN_UNSUPPORTED), so a second to_ros2 for an
// egress port_name already wired must fail loudly rather than silently rewiring, corrupting
// bridge state, or crashing. That inner error is thrown during compose() and is not the caller
// -- compile() catches it and rethrows a generic wrapping RuntimeError{kFailure,
// "GRAPH_COMPOSE_FAILED"}, so the specific GRAPH_INPUT_FANIN_UNSUPPORTED detail is not visible
// at this API boundary (confirmed empirically against this repo's pinned Holoscan SDK below);
// this test can only assert on the wrapper actually observable here.
TEST(Ros2BridgeWiring, ThrowsWhenAnEgressPortIsWiredFromASecondProducer) {
  Ros2Bridge<> bridge("wiring_test_node");
  WiringGraph graph([&](WiringGraph& self) {
    auto first_source = self.op<ops::SourceOp>("first_source");
    auto second_source = self.op<ops::SourceOp>("second_source");
    bridge.to_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
        self, first_source->out, {.port_name = "joint_command", .ros2_topic = "/wiring_test_out"});
    bridge.to_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
        self, second_source->out, {.port_name = "joint_command", .ros2_topic = "/wiring_test_out"});
  });
  try {
    (void)holoscan::compile(graph);
    FAIL() << "Expected holoscan::RuntimeError to be thrown.";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(error.code(), holoscan::ErrorCode::kFailure);
    EXPECT_NE(std::string(error.what()).find("GRAPH_COMPOSE_FAILED"), std::string::npos)
        << "error.what() was: " << error.what();
  }
}

TEST(Ros2BridgeWiring, CreatesAnImportedTensorSchemaTopicOnTheFirstFromRos2) {
  Ros2Bridge<> bridge("wiring_test_tensor_node");
  WiringGraph graph([&](WiringGraph& self) {
    auto sink = self.op<ops::ImageSinkOp>("sink");
    bridge.from_ros2<sensor_msgs::msg::Image, holoscan::schema::ImageT>(
        self, sink->in, {.port_name = "camera", .ros2_topic = "/wiring_test_camera"});
  });
  EXPECT_TRUE(holoscan::compile(graph).ok());
}

// Regression test: Pose and PoseStamped both convert to isaac.PoseCpu (same for
// Twist/TwistStamped and Transform/TransformStamped), so ImportedTopicVariant/
// ExportedTopicVariant must collapse them to one alternative per HsdkType. Before that
// dedup, wiring both through the same bridge gave the underlying std::variant a duplicate
// alternative type, making its converting construction ambiguous -- a hard compile error, not
// something this test could catch by merely running (see TopicVariantFor's comment in
// ros2_bridge.h).
TEST(Ros2BridgeWiring, WiresBothPoseAndPoseStampedToTheSameHsdkType) {
  Ros2Bridge<> bridge("wiring_test_node");
  WiringGraph graph([&](WiringGraph& self) {
    auto pose_sink = self.op<ops::PoseSinkOp>("pose_sink");
    auto pose_stamped_sink = self.op<ops::PoseSinkOp>("pose_stamped_sink");
    bridge.from_ros2<geometry_msgs::msg::Pose, isaac::PoseCpuT>(
        self, pose_sink->in, {.port_name = "pose", .ros2_topic = "/wiring_test_pose"});
    bridge.from_ros2<geometry_msgs::msg::PoseStamped, isaac::PoseCpuT>(
        self, pose_stamped_sink->in,
        {.port_name = "pose_stamped", .ros2_topic = "/wiring_test_pose_stamped"});
  });
  EXPECT_TRUE(holoscan::compile(graph).ok());
}

// Same regression as WiresBothPoseAndPoseStampedToTheSameHsdkType above, but for to_ros2()
// (egress) and a different shared HsdkType (Twist/TwistStamped -> isaac.TwistCpu) -- proving the
// ExportedTopicVariant side of the fix independently of the ImportedTopicVariant side already
// covered there.
TEST(Ros2BridgeWiring, WiresBothTwistAndTwistStampedToTheSameHsdkType) {
  Ros2Bridge<> bridge("wiring_test_node");
  WiringGraph graph([&](WiringGraph& self) {
    auto twist_source = self.op<ops::TwistSourceOp>("twist_source");
    auto twist_stamped_source = self.op<ops::TwistSourceOp>("twist_stamped_source");
    bridge.to_ros2<geometry_msgs::msg::Twist, isaac::TwistCpuT>(
        self, twist_source->out, {.port_name = "twist", .ros2_topic = "/wiring_test_twist"});
    bridge.to_ros2<geometry_msgs::msg::TwistStamped, isaac::TwistCpuT>(
        self, twist_stamped_source->out,
        {.port_name = "twist_stamped", .ros2_topic = "/wiring_test_twist_stamped"});
  });
  EXPECT_TRUE(holoscan::compile(graph).ok());
}

}  // namespace
}  // namespace isaac::os::ros2_bridge
