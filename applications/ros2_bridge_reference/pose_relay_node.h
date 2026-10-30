/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

#include <holoscan/logger/logger.hpp>
#include <rclcpp/rclcpp.hpp>

namespace isaac::os::ros2_bridge_reference {

// A genuinely external ROS2 participant, not another Holoscan/ros2_bridge port: its own
// rclcpp::Node, subscribing the pose the graph published out and republishing it unchanged on a
// second topic for the graph to subscribe back in. It is a real, distinct DDS participant,
// indistinguishable on the wire from a separate OS process, without this app needing to manage a
// second process's lifecycle to be a single reproducible command.
class PoseRelayNode {
 public:
  PoseRelayNode(std::string_view node_name, std::string_view in_ros2_topic,
                std::string_view out_ros2_topic)
      : node_(std::make_shared<rclcpp::Node>(std::string(node_name))) {
    publisher_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
        std::string(out_ros2_topic), rclcpp::QoS(10));
    subscription_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
        std::string(in_ros2_topic), rclcpp::QoS(10),
        [this](const std::shared_ptr<geometry_msgs::msg::PoseStamped> msg) {
          publisher_->publish(*msg);
          relayed_count_.fetch_add(1);
        });
    executor_.add_node(node_);
    // Blocking spin(), matching Ros2Bridge's own executor thread (ros2_bridge.h):
    // reacts to an incoming publish as soon as the middleware wakes it, no polling delay added.
    spin_thread_ = std::thread([this] {
      executor_.spin();
      spin_finished_.store(true);
    });
  }

  ~PoseRelayNode() {
    // Same cancel()-retry shape as Ros2Bridge::stop() (ros2_bridge.h): a single cancel() can race
    // spin() actually entering its blocking DDS wait and be missed, so retry until spin_finished_
    // confirms it returned. Keeps retrying past kMaxCancelAttempts -- that threshold only
    // rate-limits the error log, since giving up and joining anyway would just trade this loop's
    // wait for an equally unbounded block inside join() once spin() is, in fact, still running.
    int attempts = 0;
    while (spin_thread_.joinable() && !spin_finished_.load()) {
      executor_.cancel();
      if (++attempts == kMaxCancelAttempts) {
        HOLOSCAN_LOG_ERROR(
            "The ros2_bridge_reference relay node's executor did not stop after {} attempts; "
            "shutdown may hang.",
            attempts);
      }
      std::this_thread::sleep_for(kCancelRetryInterval);
    }
    if (spin_thread_.joinable()) spin_thread_.join();
  }

  PoseRelayNode(const PoseRelayNode&) = delete;
  PoseRelayNode& operator=(const PoseRelayNode&) = delete;

  [[nodiscard]] std::uint64_t relayed_count() const { return relayed_count_.load(); }

 private:
  static constexpr int kMaxCancelAttempts = 400;  // Logs once past 400 * 5ms = 2s, keeps retrying.
  static constexpr auto kCancelRetryInterval = std::chrono::milliseconds(5);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr publisher_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subscription_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread spin_thread_;
  std::atomic<bool> spin_finished_{false};
  std::atomic<std::uint64_t> relayed_count_{0};
};

}  // namespace isaac::os::ros2_bridge_reference
