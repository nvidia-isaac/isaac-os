/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <string_view>
#include <thread>

#include <holoscan/logger/logger.hpp>
#include <rclcpp/rclcpp.hpp>

namespace isaac::os::ros2_bridge_reference {

// The external ROS2 participant for the GPU image path, as PoseRelayNode is for poses: its own
// rclcpp::Node, subscribing the images the graph published out and republishing them unchanged on a
// second topic for the graph to subscribe back in.
//
// It subscribes with acceptable_buffer_backends = "cuda", so a CUDA message arrives in GPU memory
// instead of being copied down to the host, and it republishes the message it was handed without
// copying it. If a message still arrives in host memory, the CUDA buffer backend plugin is missing
// from this process; that is counted so main.cc can say so.
//
// It shares a process with the bridge, so it uses intra-process communication, as the bridge's node
// does: images between them are handed over in place instead of through the CUDA buffer backend's
// same-process IPC path.
class ImageRelayNode {
 public:
  ImageRelayNode(std::string_view node_name, std::string_view in_ros2_topic,
                 std::string_view out_ros2_topic)
      : node_(std::make_shared<rclcpp::Node>(std::string(node_name),
                                             rclcpp::NodeOptions().use_intra_process_comms(true))) {
    publisher_ = node_->create_publisher<sensor_msgs::msg::Image>(std::string(out_ros2_topic),
                                                                  rclcpp::QoS(10));
    rclcpp::SubscriptionOptions options;
    options.acceptable_buffer_backends = "cuda";
    subscription_ = node_->create_subscription<sensor_msgs::msg::Image>(
        std::string(in_ros2_topic), rclcpp::QoS(10),
        [this](std::unique_ptr<sensor_msgs::msg::Image> msg) {
          if (msg->data.get_backend_type() == "cuda") {
            cuda_count_.fetch_add(1);
          } else {
            host_count_.fetch_add(1);
          }
          publisher_->publish(std::move(msg));
        },
        options);
    executor_.add_node(node_);
    spin_thread_ = std::thread([this] {
      executor_.spin();
      spin_finished_.store(true);
    });
  }

  ~ImageRelayNode() {
    // The same cancel()-retry shape as PoseRelayNode: a single cancel() can race spin() entering
    // its blocking wait, so retry until spin() confirms it returned.
    int attempts = 0;
    while (spin_thread_.joinable() && !spin_finished_.load()) {
      executor_.cancel();
      if (++attempts == kMaxCancelAttempts) {
        HOLOSCAN_LOG_ERROR(
            "The ros2_bridge_reference image relay node's executor did not stop after {} "
            "attempts; shutdown may hang.",
            attempts);
      }
      std::this_thread::sleep_for(kCancelRetryInterval);
    }
    if (spin_thread_.joinable()) spin_thread_.join();
  }

  ImageRelayNode(const ImageRelayNode&) = delete;
  ImageRelayNode& operator=(const ImageRelayNode&) = delete;

  // Images that arrived as CUDA buffers.
  [[nodiscard]] std::uint64_t cuda_count() const { return cuda_count_.load(); }
  // Images that arrived in host memory, which means the CUDA buffer backend is not working.
  [[nodiscard]] std::uint64_t host_count() const { return host_count_.load(); }

 private:
  static constexpr int kMaxCancelAttempts = 400;  // Logs once past 400 * 5ms = 2s, keeps retrying.
  static constexpr auto kCancelRetryInterval = std::chrono::milliseconds(5);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread spin_thread_;
  std::atomic<bool> spin_finished_{false};
  std::atomic<std::uint64_t> cuda_count_{0};
  std::atomic<std::uint64_t> host_count_{0};
};

}  // namespace isaac::os::ros2_bridge_reference

#endif  // ISAAC_ROS2_BRIDGE_CUDA_BUFFER
