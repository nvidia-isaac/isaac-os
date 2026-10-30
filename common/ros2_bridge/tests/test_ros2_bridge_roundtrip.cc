/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/run.hpp"
#include "holoscan/time/realtime_clock.hpp"
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <rclcpp/rclcpp.hpp>

#include "common/ros2_bridge/ros2_bridge.h"

namespace isaac::os::ops {
namespace {

// Echoes whatever JointStateCpu it receives straight back out, so the external ROS2 side
// can observe its own published value having round-tripped through the graph. This is a
// real holoscan::Operator, hence isaac::os::ops -- everything else in this test file is not.
class EchoOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(joint_state_in, "joint_state_in");
    spec.output(joint_command_out, "joint_command_out");
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = joint_state_in.receive();
    if (!received) {
      return holoscan::make_unexpected(received.error());
    }
    return joint_command_out.emit(received->data);
  }

  holoscan::Input<isaac::JointStateCpuT> joint_state_in;
  holoscan::Output<isaac::JointStateCpuT> joint_command_out;
};

// Echoes whatever Image it receives straight back out, so the external ROS2 side can observe
// its own published value having round-tripped through the graph. This is a real
// holoscan::Operator, hence isaac::os::ops -- everything else in this test file is not.
class ImageEchoOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(camera_in, "camera_in");
    spec.output(camera_out, "camera_out");
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = camera_in.receive();
    if (!received) {
      return holoscan::make_unexpected(received.error());
    }
    return camera_out.emit(received->data);
  }

  holoscan::Input<holoscan::schema::ImageT> camera_in;
  holoscan::Output<holoscan::schema::ImageT> camera_out;
};

// Samples PoseCpu with LatestWithin, which only ever selects a sample that has a capture time, and
// records the capture time (ns) of the sample it selected into `selected_capture_ns`, 0 until then.
class LatestWithinProbeOp final : public holoscan::Operator<> {
 public:
  explicit LatestWithinProbeOp(std::shared_ptr<std::atomic<std::int64_t>> selected_capture_ns)
      : selected_capture_ns_(std::move(selected_capture_ns)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(pose_in, "pose_in").queue_depth(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(10)});
    contract.input(pose_in, holoscan::LatestWithin{.max_age = std::chrono::seconds(30)});
    contract.on_missing(pose_in, holoscan::SafeDefault{});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    if (pose_in.is_missing()) {
      return {};
    }
    auto received = pose_in.receive();
    if (!received) {
      return holoscan::make_unexpected(received.error());
    }
    const holoscan::TimePoint capture_time = received->metadata.capture_time();
    if (capture_time.valid()) {
      selected_capture_ns_->store(capture_time.timestamp_ns);
    }
    return {};
  }

  holoscan::Input<isaac::PoseCpuT> pose_in;

 private:
  std::shared_ptr<std::atomic<std::int64_t>> selected_capture_ns_;
};

// Counts the PoseCpu samples it drains, via DrainAll, which takes every queued sample and does not
// need a capture time.
class DrainAllProbeOp final : public holoscan::Operator<> {
 public:
  explicit DrainAllProbeOp(std::shared_ptr<std::atomic<std::int64_t>> drained)
      : drained_(std::move(drained)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(pose_in, "pose_in").queue_depth(8U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(10)});
    contract.input(pose_in, holoscan::DrainAll{});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto batch = pose_in.selected_batch();
    if (!batch) {
      return holoscan::make_unexpected(std::move(batch).error());
    }
    for ([[maybe_unused]] const auto& sample : *batch) {
      drained_->fetch_add(1);
    }
    return {};
  }

  holoscan::Input<isaac::PoseCpuT> pose_in;

 private:
  std::shared_ptr<std::atomic<std::int64_t>> drained_;
};

}  // namespace
}  // namespace isaac::os::ops

namespace isaac::os::ros2_bridge {
namespace {

class RoundtripGraph final : public holoscan::Graph {
 public:
  explicit RoundtripGraph(Ros2Bridge<>& bridge) : Graph("roundtrip_test"), bridge_(bridge) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto echo = op<isaac::os::ops::EchoOp>("echo");
    bridge_.from_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
        *this, echo->joint_state_in,
        {.port_name = "joint_state", .ros2_topic = "/roundtrip_test_joint_states"});
    bridge_.to_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
        *this, echo->joint_command_out,
        {.port_name = "joint_command", .ros2_topic = "/roundtrip_test_joint_commands"});
  }

 private:
  Ros2Bridge<>& bridge_;
};

TEST(Ros2BridgeRoundtrip, JointStatePublishedOnRos2FlowsThroughGraphAndBackOut) {
  EnsureRclcppInit();

  Ros2Bridge<> bridge("roundtrip_bridge_node");

  RoundtripGraph graph(bridge);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  bridge.start(plan.endpoint_catalog());
  auto run_session = holoscan::run_async(plan);

  auto external_node = std::make_shared<rclcpp::Node>("roundtrip_test_external_node");
  auto external_publisher = external_node->create_publisher<sensor_msgs::msg::JointState>(
      "/roundtrip_test_joint_states", rclcpp::QoS(10));

  std::mutex received_mutex;
  std::optional<sensor_msgs::msg::JointState> received;
  auto external_subscription = external_node->create_subscription<sensor_msgs::msg::JointState>(
      "/roundtrip_test_joint_commands", rclcpp::QoS(10),
      [&](const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
        std::lock_guard<std::mutex> lock(received_mutex);
        received = *msg;
      });

  rclcpp::executors::SingleThreadedExecutor external_executor;
  external_executor.add_node(external_node);
  std::atomic<bool> stop_external{false};
  std::thread external_spin([&]() {
    while (!stop_external.load()) {
      external_executor.spin_some();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });

  sensor_msgs::msg::JointState published;
  published.name = {"shoulder", "elbow"};
  published.position = {0.5, -0.25};
  published.velocity = {1.5, -0.75};
  published.effort = {1.0, -1.0};

  // Republish on a loop rather than publish-once-then-wait: DDS discovery (matching this
  // publisher/subscription to the bridge's own) isn't instantaneous, so a single publish right
  // after setup can be silently dropped before discovery completes. Repeating tolerates whatever
  // discovery delay actually occurs instead of racing a fixed one.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    external_publisher->publish(published);
    {
      std::lock_guard<std::mutex> lock(received_mutex);
      if (received.has_value()) break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  stop_external.store(true);
  external_spin.join();
  run_session.request_stop();
  run_session.wait();

  std::lock_guard<std::mutex> lock(received_mutex);
  ASSERT_TRUE(received.has_value());
  ASSERT_EQ(received->name.size(), 2U);
  EXPECT_EQ(received->name[0], published.name[0]);
  EXPECT_EQ(received->name[1], published.name[1]);
  ASSERT_EQ(received->position.size(), 2U);
  EXPECT_NEAR(received->position[0], published.position[0], 1e-4);
  EXPECT_NEAR(received->position[1], published.position[1], 1e-4);
  ASSERT_EQ(received->velocity.size(), 2U);
  EXPECT_NEAR(received->velocity[0], published.velocity[0], 1e-4);
  EXPECT_NEAR(received->velocity[1], published.velocity[1], 1e-4);
  ASSERT_EQ(received->effort.size(), 2U);
  EXPECT_NEAR(received->effort[0], published.effort[0], 1e-4);
  EXPECT_NEAR(received->effort[1], published.effort[1], 1e-4);
}

// Same echo graph as RoundtripGraph, but with caller-chosen topics and QoS on both bridge ends.
class QosRoundtripGraph final : public holoscan::Graph {
 public:
  QosRoundtripGraph(Ros2Bridge<>& bridge, std::string in_topic, std::string out_topic,
                    rclcpp::QoS ingress_qos, rclcpp::QoS egress_qos)
      : Graph("qos_roundtrip_test"),
        bridge_(bridge),
        in_topic_(std::move(in_topic)),
        out_topic_(std::move(out_topic)),
        ingress_qos_(std::move(ingress_qos)),
        egress_qos_(std::move(egress_qos)) {}

 private:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto echo = op<isaac::os::ops::EchoOp>("echo");
    bridge_.from_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
        *this, echo->joint_state_in,
        {.port_name = "joint_state", .ros2_topic = in_topic_, .qos = ingress_qos_});
    bridge_.to_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
        *this, echo->joint_command_out,
        {.port_name = "joint_command", .ros2_topic = out_topic_, .qos = egress_qos_});
  }

  Ros2Bridge<>& bridge_;
  std::string in_topic_;
  std::string out_topic_;
  rclcpp::QoS ingress_qos_;
  rclcpp::QoS egress_qos_;
};

struct QosRoundtripResult {
  // True if the echoed message reached the external subscription before the wait elapsed.
  bool delivered = false;
  // Reliability the bridge's egress publisher actually advertised, as discovered by the external
  // node; unset if it was never discovered. A best-effort external subscriber matches either
  // reliability, so delivery alone can't tell the egress profiles apart.
  std::optional<rclcpp::ReliabilityPolicy> egress_reliability;
};

// Publishes JointState from an external node with `external_publisher_qos` for up to `wait`, and
// reports whether the echo came back on an external subscription using `external_subscriber_qos`,
// plus the QoS the bridge's egress publisher advertised. A void function so a failed precondition
// can ASSERT out instead of continuing with a bad plan.
void RunQosRoundtrip(const std::string& test_name, const rclcpp::QoS& ingress_qos,
                     const rclcpp::QoS& egress_qos, const rclcpp::QoS& external_publisher_qos,
                     const rclcpp::QoS& external_subscriber_qos, std::chrono::milliseconds wait,
                     QosRoundtripResult& result) {
  EnsureRclcppInit();
  const std::string in_topic = "/qos_test_" + test_name + "_in";
  const std::string out_topic = "/qos_test_" + test_name + "_out";

  Ros2Bridge<> bridge("qos_bridge_" + test_name);
  QosRoundtripGraph graph(bridge, in_topic, out_topic, ingress_qos, egress_qos);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  bridge.start(plan.endpoint_catalog());
  auto run_session = holoscan::run_async(plan);

  auto external_node = std::make_shared<rclcpp::Node>("qos_external_" + test_name);
  auto external_publisher = external_node->create_publisher<sensor_msgs::msg::JointState>(
      in_topic, external_publisher_qos);
  std::atomic received{false};
  auto external_subscription = external_node->create_subscription<sensor_msgs::msg::JointState>(
      out_topic, external_subscriber_qos,
      [&](const std::shared_ptr<sensor_msgs::msg::JointState>) { received.store(true); });

  rclcpp::executors::SingleThreadedExecutor external_executor;
  external_executor.add_node(external_node);
  std::atomic stop_external{false};
  std::jthread external_spin([&]() {
    while (!stop_external.load()) {
      external_executor.spin_some();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });

  sensor_msgs::msg::JointState published;
  published.name = {"shoulder"};
  published.position = {0.5};
  published.velocity = {1.5};
  published.effort = {1.0};

  // Republished on a loop for the same discovery-delay reason as the roundtrip test above. The
  // bridge's egress publisher is the only publisher on out_topic, so the first discovered
  // endpoint is its.
  const auto deadline = std::chrono::steady_clock::now() + wait;
  while (std::chrono::steady_clock::now() < deadline &&
         !(received.load() && result.egress_reliability.has_value())) {
    external_publisher->publish(published);
    if (!result.egress_reliability.has_value()) {
      const auto endpoints = external_node->get_publishers_info_by_topic(out_topic);
      if (!endpoints.empty()) {
        result.egress_reliability =
            endpoints.front().qos_profile().get_rmw_qos_profile().reliability ==
                    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
                ? rclcpp::ReliabilityPolicy::BestEffort
                : rclcpp::ReliabilityPolicy::Reliable;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  stop_external.store(true);
  external_spin.join();
  run_session.request_stop();
  run_session.wait();
  result.delivered = received.load();
}

TEST(Ros2BridgeRoundtrip, BestEffortQosOnBothEndsDeliversFromBestEffortPeers) {
  QosRoundtripResult result;
  RunQosRoundtrip("best_effort_match", rclcpp::SensorDataQoS(), rclcpp::SensorDataQoS(),
                  rclcpp::SensorDataQoS(), rclcpp::SensorDataQoS(), std::chrono::seconds(5),
                  result);
  EXPECT_TRUE(result.delivered);
  ASSERT_TRUE(result.egress_reliability.has_value());
  EXPECT_EQ(*result.egress_reliability, rclcpp::ReliabilityPolicy::BestEffort);
}

// The default (reliable) ingress subscription is incompatible with a best-effort publisher, so
// nothing reaches the graph. Paired with the test above, which differs only in the ingress qos.
// The default egress qos is also checked to advertise reliable, so the best-effort assertion
// above can't be satisfied by an ignored qos.
TEST(Ros2BridgeRoundtrip, DefaultReliableIngressDoesNotReceiveFromBestEffortPublisher) {
  QosRoundtripResult result;
  RunQosRoundtrip("reliable_vs_best_effort", rclcpp::QoS(10), rclcpp::QoS(10),
                  rclcpp::SensorDataQoS(), rclcpp::SensorDataQoS(), std::chrono::seconds(2),
                  result);
  EXPECT_FALSE(result.delivered);
  ASSERT_TRUE(result.egress_reliability.has_value());
  EXPECT_EQ(*result.egress_reliability, rclcpp::ReliabilityPolicy::Reliable);
}

class CaptureTimeGraph final : public holoscan::Graph {
 public:
  CaptureTimeGraph(Ros2Bridge<>& bridge,
                   std::shared_ptr<std::atomic<std::int64_t>> selected_capture_ns)
      : Graph("capture_time_test"),
        bridge_(bridge),
        selected_capture_ns_(std::move(selected_capture_ns)) {}

 private:
  void compose() override {
    // use_time_since_epoch makes the clock read unix time, which is what a ROS2 header.stamp
    // holds. The default RealtimeClock counts from its own construction, so a ROS2 stamp would
    // look decades in the future and LatestWithin would never select it.
    const holoscan::ClockKey clock = add_clock<holoscan::RealtimeClock>(
        "clock", holoscan::RealtimeClock::Config{.use_time_since_epoch = true});
    set_default_clock(clock);
    auto probe = op<isaac::os::ops::LatestWithinProbeOp>("probe", selected_capture_ns_);
    bridge_.from_ros2<geometry_msgs::msg::PoseStamped, isaac::PoseCpuT>(
        *this, probe->pose_in,
        {.port_name = "pose",
         .ros2_topic = "/capture_time_test_pose",
         .capture_time_clock = clock});
  }

  Ros2Bridge<>& bridge_;
  std::shared_ptr<std::atomic<std::int64_t>> selected_capture_ns_;
};

// A PoseStamped's header.stamp reaches the graph as the sample's capture time, so an operator
// sampling it with LatestWithin selects it. The stamp is "now" on the system clock, which is
// what the graph's epoch-based RealtimeClock reads, so the sample is within LatestWithin's max_age.
TEST(Ros2BridgeRoundtrip, StampedMessageIsSelectedByLatestWithin) {
  EnsureRclcppInit();
  const auto selected_capture_ns = std::make_shared<std::atomic<std::int64_t>>(0);

  Ros2Bridge<> bridge("capture_time_bridge_node");
  CaptureTimeGraph graph(bridge, selected_capture_ns);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  bridge.start(plan.endpoint_catalog());
  auto run_session = holoscan::run_async(plan);

  auto external_node = std::make_shared<rclcpp::Node>("capture_time_test_external_node");
  auto external_publisher = external_node->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/capture_time_test_pose", rclcpp::QoS(10));

  // One fixed stamp slightly in the past, republished until it is selected for the same
  // discovery-delay reason as the roundtrip test above. A fixed value lets the test assert the
  // exact stamp reached the capture time, which an arrival-time stamp would not match.
  const std::int64_t stamp_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch() - std::chrono::milliseconds(100))
          .count();
  geometry_msgs::msg::PoseStamped published;
  published.header.stamp.sec = static_cast<std::int32_t>(stamp_ns / 1'000'000'000);
  published.header.stamp.nanosec = static_cast<std::uint32_t>(stamp_ns % 1'000'000'000);
  published.header.frame_id = "map";
  published.pose.position.x = 1.0;

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline && selected_capture_ns->load() == 0) {
    external_publisher->publish(published);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  run_session.request_stop();
  run_session.wait();

  ASSERT_NE(selected_capture_ns->load(), 0)
      << "LatestWithin never selected a sample: no capture time arrived";
  EXPECT_EQ(selected_capture_ns->load(), stamp_ns);
}

class DrainAllGraph final : public holoscan::Graph {
 public:
  DrainAllGraph(Ros2Bridge<>& bridge, std::shared_ptr<std::atomic<std::int64_t>> drained)
      : Graph("drain_all_test"), bridge_(bridge), drained_(std::move(drained)) {}

 private:
  void compose() override {
    const holoscan::ClockKey clock = add_clock<holoscan::RealtimeClock>(
        "clock", holoscan::RealtimeClock::Config{.use_time_since_epoch = true});
    set_default_clock(clock);
    auto probe = op<isaac::os::ops::DrainAllProbeOp>("probe", drained_);
    bridge_.from_ros2<geometry_msgs::msg::Pose, isaac::PoseCpuT>(
        *this, probe->pose_in,
        {.port_name = "pose", .ros2_topic = "/drain_all_test_pose", .capture_time_clock = clock});
  }

  Ros2Bridge<>& bridge_;
  std::shared_ptr<std::atomic<std::int64_t>> drained_;
};

// An unstamped message (Pose has no header) still reaches a DrainAll consumer when the topic
// declares a capture clock: the bridge stamps it with its receive time.
TEST(Ros2BridgeRoundtrip, UnstampedMessageReachesADrainAllConsumer) {
  EnsureRclcppInit();
  const auto drained = std::make_shared<std::atomic<std::int64_t>>(0);

  Ros2Bridge<> bridge("drain_all_bridge_node");
  DrainAllGraph graph(bridge, drained);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  bridge.start(plan.endpoint_catalog());
  auto run_session = holoscan::run_async(plan);

  auto external_node = std::make_shared<rclcpp::Node>("drain_all_test_external_node");
  auto external_publisher = external_node->create_publisher<geometry_msgs::msg::Pose>(
      "/drain_all_test_pose", rclcpp::QoS(10));

  // Republished on a loop for the same discovery-delay reason as the roundtrip test above.
  geometry_msgs::msg::Pose published;
  published.position.x = 1.0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline && drained->load() == 0) {
    external_publisher->publish(published);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  run_session.request_stop();
  run_session.wait();

  EXPECT_GT(drained->load(), 0) << "DrainAll never received a sample from the unstamped topic";
}

class ImageRoundtripGraph final : public holoscan::Graph {
 public:
  explicit ImageRoundtripGraph(Ros2Bridge<>& bridge)
      : Graph("image_roundtrip_test"), bridge_(bridge) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto echo = op<isaac::os::ops::ImageEchoOp>("echo");
    bridge_.from_ros2<sensor_msgs::msg::Image, holoscan::schema::ImageT>(
        *this, echo->camera_in,
        {.port_name = "camera_in", .ros2_topic = "/roundtrip_test_camera_in"});
    bridge_.to_ros2<sensor_msgs::msg::Image, holoscan::schema::ImageT>(
        *this, echo->camera_out,
        {.port_name = "camera_out", .ros2_topic = "/roundtrip_test_camera_out"});
  }

 private:
  Ros2Bridge<>& bridge_;
};

TEST(Ros2BridgeRoundtrip, ImagePublishedOnRos2FlowsThroughGraphAndBackOut) {
  EnsureRclcppInit();

  Ros2Bridge<> bridge("image_roundtrip_bridge_node");

  ImageRoundtripGraph graph(bridge);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  bridge.start(plan.endpoint_catalog());
  auto run_session = holoscan::run_async(plan);

  auto external_node = std::make_shared<rclcpp::Node>("image_roundtrip_test_external_node");
  auto external_publisher = external_node->create_publisher<sensor_msgs::msg::Image>(
      "/roundtrip_test_camera_in", rclcpp::QoS(10));

  std::mutex received_mutex;
  std::optional<sensor_msgs::msg::Image> received;
  auto external_subscription = external_node->create_subscription<sensor_msgs::msg::Image>(
      "/roundtrip_test_camera_out", rclcpp::QoS(10),
      [&](const std::shared_ptr<sensor_msgs::msg::Image> msg) {
        std::lock_guard<std::mutex> lock(received_mutex);
        received = *msg;
      });

  rclcpp::executors::SingleThreadedExecutor external_executor;
  external_executor.add_node(external_node);
  std::atomic<bool> stop_external{false};
  std::thread external_spin([&]() {
    while (!stop_external.load()) {
      external_executor.spin_some();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });

  sensor_msgs::msg::Image published;
  published.height = 2;
  published.width = 2;
  published.encoding = "rgb8";
  published.step = 6;
  published.data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};

  // Republish on a loop rather than publish-once-then-wait: DDS discovery (matching this
  // publisher/subscription to the bridge's own) isn't instantaneous, so a single publish right
  // after setup can be silently dropped before discovery completes. Repeating tolerates whatever
  // discovery delay actually occurs instead of racing a fixed one.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    external_publisher->publish(published);
    {
      std::lock_guard<std::mutex> lock(received_mutex);
      if (received.has_value()) break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  stop_external.store(true);
  external_spin.join();
  run_session.request_stop();
  run_session.wait();

  std::lock_guard<std::mutex> lock(received_mutex);
  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->height, published.height);
  EXPECT_EQ(received->width, published.width);
  EXPECT_EQ(received->encoding, published.encoding);
  EXPECT_EQ(received->step, published.step);
  ASSERT_EQ(received->data.size(), published.data.size());
  EXPECT_EQ(received->data, published.data);
}

}  // namespace
}  // namespace isaac::os::ros2_bridge
