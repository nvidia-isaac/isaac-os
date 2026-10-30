/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <sensor_msgs/msg/joint_state.hpp>
#include <thread>
#include <utility>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/time/realtime_clock.hpp"

#include "common/ros2_bridge/ros2_bridge.h"

namespace isaac::os::ops {
namespace {

// Echoes whatever it receives straight back out, connecting the bridge's ingress and egress
// wiring into one chain (imported topic -> echo -> exported topic) rather than two disconnected
// operators. That connectivity is what lets a graph wire both directions and still compile
// without a manually authored execution partition -- two *unconnected* single-port operators are
// two independent Holoscan execution roots, which Holoscan cannot auto-partition (see the
// PARTITION_MANUAL_AUTHORING_REQUIRED discussion on Ros2Bridge::from_ros2 in
// ros2_bridge.h). Mirrors EchoOp in test_ros2_bridge_roundtrip.cc; not shared across files.
class EchoOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample) return holoscan::make_unexpected(sample.error());
    return out.emit(sample->data);
  }
  holoscan::Input<isaac::JointStateCpuT> in;
  holoscan::Output<isaac::JointStateCpuT> out;
};

}  // namespace
}  // namespace isaac::os::ops

namespace isaac::os::ros2_bridge {
namespace {

// An empty graph: no consumer or producer, so nothing calls from_ros2/to_ros2. Used only
// by the two tests below that exercise start()/stop() over a bridge with nothing wired at all.
class EmptyGraph final : public holoscan::Graph {
 public:
  explicit EmptyGraph(Ros2Bridge<>& bridge) : Graph("session_test"), bridge_(bridge) {}

 protected:
  void compose() override {}

 private:
  Ros2Bridge<>& bridge_;
};

// Wires an ingress and an egress port through one EchoOp, so start() has both an ingress
// subscription and an egress publisher to attach in a single running session. in_spec/out_spec
// are supplied per-test so each test can use its own, non-colliding ROS2 topic names.
class WiredGraph final : public holoscan::Graph {
 public:
  WiredGraph(Ros2Bridge<>& bridge, Ros2TopicSpec in_spec, Ros2TopicSpec out_spec)
      : Graph("session_test"),
        bridge_(bridge),
        in_spec_(std::move(in_spec)),
        out_spec_(std::move(out_spec)) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto echo = op<ops::EchoOp>("echo");
    bridge_.from_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(*this, echo->in,
                                                                           in_spec_);
    bridge_.to_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(*this, echo->out,
                                                                         out_spec_);
  }

 private:
  Ros2Bridge<>& bridge_;
  Ros2TopicSpec in_spec_;
  Ros2TopicSpec out_spec_;
};

TEST(Ros2BridgeLifecycle, StartsAndStopsCleanly) {
  Ros2Bridge<> bridge("test_session_node");
  EmptyGraph graph(bridge);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  bridge.start(plan.endpoint_catalog());
  // bridge's destructor runs at end of scope; test passes if it doesn't hang or crash.
}

TEST(Ros2BridgeLifecycle, ThrowsOnDoubleStart) {
  Ros2Bridge<> bridge("test_session_double_start_node");
  EmptyGraph graph(bridge);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  bridge.start(plan.endpoint_catalog());
  EXPECT_THROW(bridge.start(plan.endpoint_catalog()), std::runtime_error);
}

// Confirms both wired topics actually get wired to live ROS2 endpoints and stay that way -- not
// just that construction succeeds, but that an independent "external" ROS2 node standing in for
// a separate process can discover the bridge's subscription and publisher. An empty graph (as in
// StartsAndStopsCleanly above) can never exercise either path.
//
// found_subscriber specifically guards against a real bug found in this codebase: the
// ROS2 subscription's shared_ptr returned by rclcpp::Node::create_subscription, if
// discarded instead of retained by the session, is destroyed at the end of the statement
// that created it -- rclcpp keeps only a weak reference internally -- silently dropping the
// entire ROS2->Holoscan direction with no compile or runtime error anywhere else.
// found_publisher has no equivalent history; it's a plain sanity check that the
// Holoscan->ROS2 direction wired up too.
TEST(Ros2BridgeLifecycle, WiresConfiguredTopicsAndKeepsThemAlive) {
  Ros2Bridge<> bridge("test_session_wiring_node");
  WiredGraph graph(bridge, {.port_name = "in_port", .ros2_topic = "/session_test_in"},
                   {.port_name = "out_port", .ros2_topic = "/session_test_out"});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  bridge.start(plan.endpoint_catalog());

  EnsureRclcppInit();
  auto external_node = std::make_shared<rclcpp::Node>("session_test_external_observer");

  // ROS2 graph discovery is asynchronous even within one process, so poll with a bound
  // rather than asserting on the first check.
  bool found_subscriber = false;
  bool found_publisher = false;
  for (int i = 0; i < 100 && !(found_subscriber && found_publisher); ++i) {
    found_subscriber = external_node->count_subscribers("/session_test_in") > 0;
    found_publisher = external_node->count_publishers("/session_test_out") > 0;
    if (!(found_subscriber && found_publisher)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  EXPECT_TRUE(found_subscriber)
      << "The bridge's ROS2 subscription on '/session_test_in' was not discoverable -- it may "
         "have been destroyed immediately after creation instead of kept alive by the session.";
  EXPECT_TRUE(found_publisher)
      << "The bridge's ROS2 publisher on '/session_test_out' was not discoverable.";
}

// A plain (non-tensor) topic ignores memory_kind. Setting kCudaDevice on one must not make the
// bridge touch the CUDA pool, which is unbound here: the egress thread would throw and take the
// process down.
TEST(Ros2BridgeLifecycle, APlainTopicWithADeviceMemoryKindStartsAndStops) {
  Ros2Bridge<> bridge("test_session_plain_device_kind_node");
  WiredGraph graph(bridge,
                   {.port_name = "in_port",
                    .ros2_topic = "/session_test_plain_in",
                    .memory_kind = Ros2MemoryKind::kCudaDevice},
                   {.port_name = "out_port",
                    .ros2_topic = "/session_test_plain_out",
                    .memory_kind = Ros2MemoryKind::kCudaDevice});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  bridge.start(plan.endpoint_catalog());
  // Let the egress thread reach its loop before the bridge stops.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  bridge.stop();
}

// Proves teardown, not just startup: WiresConfiguredTopicsAndKeepsThemAlive above only
// checks state *while* the bridge is started. Neither it nor StartsAndStopsCleanly (which
// only proves the destructor doesn't hang or crash) ever observes what stopping the bridge
// actually leaves behind. This test explicitly stops the bridge mid-test (via stop(), not
// scope exit) and then polls the same external-node discovery mechanism back down to zero,
// proving the ROS2 subscription and publisher were actually torn down -- not merely that
// construction worked.
TEST(Ros2BridgeLifecycle, StopStopsRos2Wiring) {
  Ros2Bridge<> bridge("test_session_teardown_node");
  WiredGraph graph(bridge, {.port_name = "in_port", .ros2_topic = "/session_test_teardown_in"},
                   {.port_name = "out_port", .ros2_topic = "/session_test_teardown_out"});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  bridge.start(plan.endpoint_catalog());

  EnsureRclcppInit();
  auto external_node = std::make_shared<rclcpp::Node>("session_test_teardown_observer");

  bool found_subscriber = false;
  bool found_publisher = false;
  for (int i = 0; i < 100 && !(found_subscriber && found_publisher); ++i) {
    found_subscriber = external_node->count_subscribers("/session_test_teardown_in") > 0;
    found_publisher = external_node->count_publishers("/session_test_teardown_out") > 0;
    if (!(found_subscriber && found_publisher)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  ASSERT_TRUE(found_subscriber) << "Setup failed: the bridge's subscription never became visible.";
  ASSERT_TRUE(found_publisher) << "Setup failed: the bridge's publisher never became visible.";

  bridge.stop();  // Explicit teardown, at a known point -- not scope exit.

  bool subscriber_gone = false;
  bool publisher_gone = false;
  for (int i = 0; i < 100 && !(subscriber_gone && publisher_gone); ++i) {
    subscriber_gone = external_node->count_subscribers("/session_test_teardown_in") == 0;
    publisher_gone = external_node->count_publishers("/session_test_teardown_out") == 0;
    if (!(subscriber_gone && publisher_gone)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  EXPECT_TRUE(subscriber_gone)
      << "The bridge's ROS2 subscription on '/session_test_teardown_in' was still discoverable "
         "after stop() -- teardown did not actually stop it.";
  EXPECT_TRUE(publisher_gone)
      << "The bridge's ROS2 publisher on '/session_test_teardown_out' was still discoverable "
         "after stop() -- teardown did not actually stop it.";
}

}  // namespace
}  // namespace isaac::os::ros2_bridge
