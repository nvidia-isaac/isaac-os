/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "holoscan/time/clock_registry.hpp"
#include "rclcpp/qos.hpp"

namespace isaac::os::ros2_bridge {

// Where a tensor topic's pixels live. A bridge-local enum, not holoscan::MemoryKind: it allows
// only the two kinds the bridge supports.
enum class Ros2MemoryKind { kHost, kCudaDevice };

// Per-topic wiring info passed directly to Ros2Bridge::from_ros2()/to_ros2() -- not a config
// list, not YAML-parsed. One spec is exactly what one from_ros2/to_ros2 call needs to wire one
// topic.
struct Ros2TopicSpec {
  std::string port_name;
  std::string ros2_topic;
  // Only meaningful for a tensor-schema-backed HsdkT; ignored for a plain pair. See
  // TensorConverter::contract() in tensor_converter.h for what this bound signifies.
  std::optional<std::uint64_t> max_tensor_bytes;
  // QoS for the ROS2 publisher (to_ros2) or subscription (from_ros2) this spec creates. Defaults
  // to a reliable, volatile, keep-last-10 profile. Use e.g. rclcpp::SensorDataQoS() to match a
  // best-effort sensor driver. If a port is wired by from_ros2 more than once, the last call's
  // qos is the one used.
  rclcpp::QoS qos{10};
  // Tensor-schema topics only. Defaults to kHost, today's behaviour. A kCudaDevice topic lives on
  // the bridge's CUDA buffer pool device (the Ros2Bridge constructor's pool_device); there is no
  // per-topic device.
  Ros2MemoryKind memory_kind = Ros2MemoryKind::kHost;
  // from_ros2 only, and only for a kCudaDevice topic: where the ROS publisher puts the pixels. The
  // default, kCudaDevice, asks ROS for CUDA buffers, so a publisher that sends GPU memory is
  // wrapped in place. Set kHost for a publisher that sends host memory (a typical camera driver):
  // the subscription then asks for host buffers and the bridge copies each frame to the GPU once.
  // A wrong value never changes the result, only the speed: ROS copies a frame the other way
  // first. The bridge logs a warning when it sees a host buffer on a topic declared kCudaDevice.
  Ros2MemoryKind publisher_memory = Ros2MemoryKind::kCudaDevice;
  // from_ros2 only. The graph clock (from add_clock) that ROS2 time is in. When set, each message
  // is emitted with a capture time: its header.stamp, or the time it was received for a type with
  // no header. LatestWithin and the other event-time selectors require one; unset, they fail at
  // compile(). The bridge cannot check the clocks match, so for wall-clock ROS2 use a RealtimeClock
  // with use_time_since_epoch = true.
  std::optional<holoscan::ClockKey> capture_time_clock;
};

}  // namespace isaac::os::ros2_bridge
