/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <holoscan/core/graph.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "applications/ros2_bridge_reference/image_check_op.h"
#include "applications/ros2_bridge_reference/image_generator_op.h"
#include "applications/ros2_bridge_reference/pose_generator_op.h"
#include "applications/ros2_bridge_reference/pose_logger_op.h"
#include "common/ros2_bridge/ros2_bridge.h"

namespace isaac::os::ros2_bridge_reference {

// PoseGeneratorOp -> to_ros2 -> kPoseOutRos2Topic: what the graph publishes out to ROS2.
inline constexpr char kPoseOutRos2Topic[] = "/ros2_bridge_reference/pose_out";
// kPoseInRos2Topic -> from_ros2 -> PoseLoggerOp: what the graph subscribes back in from ROS2.
// A distinct topic from kPoseOutRos2Topic -- main.cc's PoseRelayNode is the external ROS2
// participant that bridges the two, republishing everything it receives on the former back out on
// the latter.
inline constexpr char kPoseInRos2Topic[] = "/ros2_bridge_reference/pose_in";

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
// The GPU image path, present only when ros2_bridge has the CUDA buffer backend. ImageGeneratorOp
// -> to_ros2 -> kImageOutRos2Topic -> ImageRelayNode -> kImageInRos2Topic -> from_ros2 ->
// ImageCheckOp, with the pixels in GPU memory the whole way.
inline constexpr char kImageOutRos2Topic[] = "/ros2_bridge_reference/image_out";
inline constexpr char kImageInRos2Topic[] = "/ros2_bridge_reference/image_in";
#endif

// Minimal worked example of common/ros2_bridge: PoseGeneratorOp emits an isaac.PoseCpu,
// to_ros2 publishes it as a geometry_msgs/PoseStamped on kPoseOutRos2Topic, from_ros2 subscribes
// kPoseInRos2Topic back into the graph, and PoseLoggerOp logs what came back -- a full Holoscan ->
// ROS2 -> Holoscan round trip through a genuinely external ROS2 node (main.cc's PoseRelayNode),
// which is how ros2_bridge is meant to be used: talking to a ROS2 participant outside the graph,
// not looping a topic back to itself.
//
// With `with_cuda_image` (only available when ros2_bridge has the CUDA buffer backend) the graph
// also round-trips a 640x480 Image through ROS2 as a CUDA buffer; see ImageGeneratorOp. The caller
// must then bind the image topics' devices when it compiles (main.cc).
class ReferenceGraph final : public holoscan::Graph {
 public:
  explicit ReferenceGraph(isaac::os::ros2_bridge::Ros2Bridge<>& bridge,
                          bool with_cuda_image = false)
      : Graph("ros2_bridge_reference"), bridge_(bridge), with_cuda_image_(with_cuda_image) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));

    auto generator = op<PoseGeneratorOp>("pose_generator");
    bridge_.to_ros2<geometry_msgs::msg::PoseStamped, isaac::PoseCpuT>(
        *this, generator->out_pose, {.port_name = "out_pose", .ros2_topic = kPoseOutRos2Topic});

    auto logger = op<PoseLoggerOp>("pose_logger");
    bridge_.from_ros2<geometry_msgs::msg::PoseStamped, isaac::PoseCpuT>(
        *this, logger->in_pose, {.port_name = "in_pose", .ros2_topic = kPoseInRos2Topic});

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
    if (with_cuda_image_) {
      // memory_kind = kCudaDevice keeps the pixels in GPU memory on both sides of the bridge.
      auto image_generator = op<ImageGeneratorOp>("image_generator", bridge_.pool_device());
      bridge_.to_ros2<sensor_msgs::msg::Image, holoscan::schema::ImageT>(
          *this, image_generator->out_image,
          {.port_name = "out_image",
           .ros2_topic = kImageOutRos2Topic,
           .memory_kind = isaac::os::ros2_bridge::Ros2MemoryKind::kCudaDevice});

      auto image_check = op<ImageCheckOp>("image_check");
      // The default publisher_memory (kCudaDevice) matches the relay node, which publishes CUDA
      // buffers.
      bridge_.from_ros2<sensor_msgs::msg::Image, holoscan::schema::ImageT>(
          *this, image_check->in_image,
          {.port_name = "in_image",
           .ros2_topic = kImageInRos2Topic,
           .memory_kind = isaac::os::ros2_bridge::Ros2MemoryKind::kCudaDevice});
    }
#endif
  }

 private:
  isaac::os::ros2_bridge::Ros2Bridge<>& bridge_;
  [[maybe_unused]] bool with_cuda_image_;
};

}  // namespace isaac::os::ros2_bridge_reference
