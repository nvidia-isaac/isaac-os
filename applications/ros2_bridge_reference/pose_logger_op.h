/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/logger/logger.hpp>

#include "pose_cpu_schema_traits.h"

namespace isaac::os::ros2_bridge_reference {

// Counts every isaac::PoseCpuT that made the round trip out through ros2_bridge's to_ros2,
// through the external ROS2 relay node, and back in through from_ros2. stop()'s tally (compared
// against PoseGeneratorOp's and PoseRelayNode's own counts, see main.cc) is the evidence that
// every message survived the round trip intact; no need to also log each one individually.
class PoseLoggerOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in_pose, "in_pose"); }

  void stop() override { HOLOSCAN_LOG_INFO("PoseLoggerOp received {} poses.", received_count_); }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = in_pose.receive();
    if (!received) {
      return holoscan::make_unexpected(received.error());
    }
    ++received_count_;
    return {};
  }

  holoscan::Input<isaac::PoseCpuT> in_pose;

 private:
  std::uint64_t received_count_ = 0;
};

}  // namespace isaac::os::ros2_bridge_reference
