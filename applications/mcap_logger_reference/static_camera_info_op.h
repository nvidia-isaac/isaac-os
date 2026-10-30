/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>

#include <Eigen/Dense>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "camera_info_cpu_schema_traits.h"
#include "image_counter_op.h"  // For kImageWidth/kImageHeight -- intrinsics match the image stream.

namespace isaac::os::mcap_logger_reference {

// Emits a fixed isaac::CameraInfoCpuT every tick -- nominal pinhole intrinsics for a
// kImageWidth x kImageHeight camera, no distortion, paired with ImageCounterOp's stream to
// demonstrate a realistic camera/CameraInfo topic pairing. No motion to compute here, unlike
// CircularMotionOp -- every emitted sample is identical except its timestamp.
class StaticCameraInfoOp final : public holoscan::Operator<> {
 public:
  static constexpr auto kPeriod =
      std::chrono::milliseconds(50);  // 20 Hz, matching the other streams.
  static constexpr double kFocalLengthPx = 500.0;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out_camera_info, "out_camera_info");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kPeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    // Real wall-clock time, not a tick-derived value -- see CircularMotionOp's compute() comment
    // for why: Foxglove interprets this field as absolute Unix epoch nanoseconds.
    const std::int64_t wall_clock_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::system_clock::now().time_since_epoch())
                                           .count();

    const double cx = static_cast<double>(kImageWidth) / 2.0;
    const double cy = static_cast<double>(kImageHeight) / 2.0;

    isaac::CameraInfoCpuT camera_info;
    camera_info.header = std::make_shared<holoscan::schema::HeaderT>();
    camera_info.header->frame_id = "camera";
    camera_info.header->capture_timestamp_ns = wall_clock_ns;
    camera_info.height = kImageHeight;
    camera_info.width = kImageWidth;
    camera_info.distortion_model = "plumb_bob";
    camera_info.d = {0.0, 0.0, 0.0, 0.0, 0.0};  // No distortion.
    camera_info.k = holoscan::schema::Matrix3d(
        std::array<double, 9>{kFocalLengthPx, 0.0, cx, 0.0, kFocalLengthPx, cy, 0.0, 0.0, 1.0});
    camera_info.r = holoscan::schema::Matrix3d(std::array<double, 9>{
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});  // Identity: no rectification.
    // isaac.Matrix3x4d's native_type is Eigen::Matrix<double, 3, 4, RowMajor> directly (see
    // common/ros2_bridge/converter.h's CameraInfoCpu converter for the same pattern).
    camera_info.p << kFocalLengthPx, 0.0, cx, 0.0, 0.0, kFocalLengthPx, cy, 0.0, 0.0, 0.0, 1.0, 0.0;

    return out_camera_info.emit(camera_info);
  }

  holoscan::Output<isaac::CameraInfoCpuT> out_camera_info;
};

}  // namespace isaac::os::mcap_logger_reference
