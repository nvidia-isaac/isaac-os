/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Must fail to compile: a plain converter that lacks to_ros2().
#include <tuple>

#include "common/ros2_bridge/ros2_bridge.h"

namespace isaac::os::ros2_bridge {
namespace {

struct MissingToRos2 {
  using Ros2Type = sensor_msgs::msg::JointState;
  using HsdkType = isaac::JointStateCpuT;
  static HsdkType to_hsdk(const Ros2Type&) { return {}; }
};

struct BadConverter {
  using SupportedPairs = std::tuple<MissingToRos2>;
};

}  // namespace
}  // namespace isaac::os::ros2_bridge

int main() {
  using namespace isaac::os::ros2_bridge;
  Ros2Bridge<BadConverter> bridge;
  holoscan::Graph graph("compile_fail");
  holoscan::Input<isaac::JointStateCpuT> input;
  bridge.from_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
      graph, input, {.port_name = "p", .ros2_topic = "/t"});
  return 0;
}
