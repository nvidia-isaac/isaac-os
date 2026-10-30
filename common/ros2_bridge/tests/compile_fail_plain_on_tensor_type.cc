/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Must fail to compile: a plain converter registered for a tensor-schema type.
#include <tuple>

#include "common/ros2_bridge/ros2_bridge.h"

namespace isaac::os::ros2_bridge {
namespace {

struct PlainOnTensorType {
  using Ros2Type = sensor_msgs::msg::Image;
  using HsdkType = holoscan::schema::ImageT;
  static HsdkType to_hsdk(const Ros2Type&) { return {}; }
  static Ros2Type to_ros2(const HsdkType&) { return {}; }
};

struct BadConverter {
  using SupportedPairs = std::tuple<PlainOnTensorType>;
};

}  // namespace
}  // namespace isaac::os::ros2_bridge

int main() {
  using namespace isaac::os::ros2_bridge;
  Ros2Bridge<BadConverter> bridge;
  holoscan::Graph graph("compile_fail");
  holoscan::Input<holoscan::schema::ImageT> input;
  bridge.from_ros2<sensor_msgs::msg::Image, holoscan::schema::ImageT>(
      graph, input, {.port_name = "p", .ros2_topic = "/t"});
  return 0;
}
