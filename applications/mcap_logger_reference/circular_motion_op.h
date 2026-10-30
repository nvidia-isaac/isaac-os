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

#include "pose_cpu_schema_traits.h"  // Also brings in holoscan::schema::HeaderT.
#include "transform_cpu_schema_traits.h"
#include "twist_cpu_schema_traits.h"

namespace isaac::os::mcap_logger_reference {

// Emits isaac::PoseCpuT/TwistCpuT/TransformCpuT every tick, all three describing the same
// constant-speed circular motion in the XY plane -- one shared per-tick angle computation, not
// three independent generators, since position/velocity/orientation here are three views of one
// motion rather than three unrelated waveforms (contrast JointStateWaveOp's sine/cosine, which
// really are two independent signals). Same deterministic tick_count_ * kPeriod timing approach
// as JointStateWaveOp, for the same reproducibility reason.
class CircularMotionOp final : public holoscan::Operator<> {
 public:
  static constexpr auto kPeriod =
      std::chrono::milliseconds(50);  // 20 Hz, matching JointStateWaveOp.
  static constexpr double kRadiusM = 2.0;
  static constexpr double kAngularFrequency = std::numbers::pi;  // rad/s (0.5 Hz revolution).

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out_pose, "out_pose");
    spec.output(out_twist, "out_twist");
    spec.output(out_transform, "out_transform");
  }

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

    // capture_timestamp_ns is real wall-clock time (what Foxglove's PoseInFrame/SceneUpdate/
    // FrameTransform interpret every message's own `timestamp` field as: absolute Unix epoch
    // nanoseconds), deliberately NOT the deterministic elapsed_s/tick_count_ used for the motion
    // itself below -- elapsed_s starting near zero each run is exactly why that reproducibility is
    // valuable for the waveform values, but stuffing it into an absolute-time field renders as
    // "just after the Unix epoch" (1969/1970) in any Foxglove panel that reads it.
    const std::int64_t wall_clock_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::system_clock::now().time_since_epoch())
                                           .count();

    const double x = kRadiusM * std::cos(theta);
    const double y = kRadiusM * std::sin(theta);
    // d/dt of (R cos(wt), R sin(wt)): tangential velocity, CCW.
    const double vx = -kRadiusM * kAngularFrequency * std::sin(theta);
    const double vy = kRadiusM * kAngularFrequency * std::cos(theta);

    // Yaw quaternion facing the direction of travel: tangent leads position by 90 degrees for
    // CCW motion. holoscan.schema.Quaterniond is (x, y, z, w) on the wire already -- no
    // reordering needed, unlike the older isaac.PoseStamped tensor-backed convention.
    const double yaw = theta + (std::numbers::pi / 2.0);
    const holoscan::schema::Quaterniond orientation(0.0, 0.0, std::sin(yaw / 2.0),
                                                    std::cos(yaw / 2.0));
    const holoscan::schema::Vector3d position(x, y, 0.0);

    isaac::PoseCpuT pose;
    pose.position = position;
    pose.orientation = orientation;
    pose.header = std::make_shared<holoscan::schema::HeaderT>();
    pose.header->frame_id = "world";
    pose.header->capture_timestamp_ns = wall_clock_ns;
    auto pose_result = out_pose.emit(pose);
    if (!pose_result) {
      return pose_result;
    }

    isaac::TwistCpuT twist;
    twist.linear = holoscan::schema::Vector3d(vx, vy, 0.0);
    twist.angular = holoscan::schema::Vector3d(0.0, 0.0, kAngularFrequency);
    twist.header = std::make_shared<holoscan::schema::HeaderT>();
    twist.header->frame_id = "world";
    twist.header->capture_timestamp_ns = wall_clock_ns;
    auto twist_result = out_twist.emit(twist);
    if (!twist_result) {
      return twist_result;
    }

    isaac::TransformCpuT transform;
    transform.translation = position;
    transform.rotation = orientation;
    transform.header = std::make_shared<holoscan::schema::HeaderT>();
    transform.header->frame_id = "world";
    transform.header->capture_timestamp_ns = wall_clock_ns;
    transform.child_frame_id = "robot";
    return out_transform.emit(transform);
  }

  holoscan::Output<isaac::PoseCpuT> out_pose;
  holoscan::Output<isaac::TwistCpuT> out_twist;
  holoscan::Output<isaac::TransformCpuT> out_transform;

 private:
  std::uint64_t tick_count_ = 0;
};

}  // namespace isaac::os::mcap_logger_reference
