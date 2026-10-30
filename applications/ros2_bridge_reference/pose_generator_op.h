/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>

#include "pose_cpu_schema_traits.h"  // Also brings in holoscan::schema::HeaderT.

namespace isaac::os::ros2_bridge_reference {

// Emits an isaac::PoseCpuT every tick describing constant-speed circular motion in the XY
// plane -- deliberately the only signal in this app, since the point is to exercise ros2_bridge's
// to_ros2/from_ros2 round trip, not to showcase every convertible message type (see
// mcap_logger_reference's CircularMotionOp for that).
class PoseGeneratorOp final : public holoscan::Operator<> {
 public:
  static constexpr auto kPeriod = std::chrono::milliseconds(50);  // 20 Hz.
  static constexpr double kRadiusM = 2.0;
  static constexpr double kAngularFrequency = std::numbers::pi;  // rad/s (0.5 Hz revolution).

  void setup(holoscan::OperatorSpec& spec) override { spec.output(out_pose, "out_pose"); }

  // Logs the delivery tally at graph teardown, alongside PoseLoggerOp::stop()'s own count --
  // together the two lines are the evidence that everything sent made it through the round trip.
  void stop() override { HOLOSCAN_LOG_INFO("PoseGeneratorOp sent {} poses.", sent_count_); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kPeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    const double elapsed_s =
        static_cast<double>(tick_count_) * std::chrono::duration<double>(kPeriod).count();
    const double theta = kAngularFrequency * elapsed_s;
    ++tick_count_;

    const std::int64_t wall_clock_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::system_clock::now().time_since_epoch())
                                           .count();

    const holoscan::schema::Vector3d position(kRadiusM * std::cos(theta),
                                              kRadiusM * std::sin(theta), 0.0);
    // Yaw quaternion facing the direction of travel: tangent leads position by 90 degrees for
    // CCW motion. holoscan.schema.Quaterniond is (x, y, z, w) on the wire already -- no
    // reordering needed.
    const double yaw = theta + (std::numbers::pi / 2.0);
    const holoscan::schema::Quaterniond orientation(0.0, 0.0, std::sin(yaw / 2.0),
                                                    std::cos(yaw / 2.0));

    isaac::PoseCpuT pose;
    pose.position = position;
    pose.orientation = orientation;
    pose.header = std::make_shared<holoscan::schema::HeaderT>();
    pose.header->frame_id = "world";
    pose.header->capture_timestamp_ns = wall_clock_ns;
    auto result = out_pose.emit(pose);
    if (result) ++sent_count_;
    return result;
  }

  holoscan::Output<isaac::PoseCpuT> out_pose;

 private:
  std::uint64_t tick_count_ = 0;
  std::uint64_t sent_count_ = 0;
};

}  // namespace isaac::os::ros2_bridge_reference
