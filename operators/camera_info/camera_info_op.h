/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_CAMERA_INFO_CAMERA_INFO_OP_H
#define ISAAC_OS_OPERATORS_CAMERA_INFO_CAMERA_INFO_OP_H

#include <string>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "camera_info_cpu_schema_traits.h"

namespace isaac::os::ops {

/// Calibration of one camera, as read from the rig calibration file.
struct CameraInfoParams {
  /// Coordinate frame the camera's images are expressed in.
  std::string frame_id;
  /// Focal length along the image x axis, in pixels.
  double fx{};
  /// Focal length along the image y axis, in pixels.
  double fy{};
  /// Optical center along the image x axis, in pixels.
  double cx{};
  /// Optical center along the image y axis, in pixels.
  double cy{};
  /// Distance to the reference camera of a rectified stereo pair, in meters. Zero for the camera
  /// the pair is referenced to.
  double stereo_baseline{};
  /// One of `none`, `brown`, `fisheye`, or `polynomial`. Published under the sensor_msgs name for
  /// the same model, since the message this operator emits mirrors sensor_msgs/CameraInfo.
  std::string distortion_model;
  /// Distortion coefficients, as many as the model requires.
  std::vector<double> distortion_parameters;
};

HOLOSCAN_PARAM_SCHEMA(
    CameraInfoParams, holoscan::param_schema("com.nvidia.isaac.CameraInfoParams", 1),
    holoscan::field<&CameraInfoParams::frame_id>("frame_id"),
    holoscan::field<&CameraInfoParams::fx>("fx"), holoscan::field<&CameraInfoParams::fy>("fy"),
    holoscan::field<&CameraInfoParams::cx>("cx"), holoscan::field<&CameraInfoParams::cy>("cy"),
    holoscan::field<&CameraInfoParams::stereo_baseline>("stereo_baseline"),
    holoscan::field<&CameraInfoParams::distortion_model>("distortion_model"),
    holoscan::field<&CameraInfoParams::distortion_parameters>("distortion_parameters"));

/// Publishes the calibration of one camera alongside each frame that camera produces.
///
/// The calibration itself is static, but it is published per frame so that a consumer receives it
/// on the same timeline as the pixels it describes, and so that a recorded stream carries the
/// calibration it was captured under rather than whatever the replaying process is configured
/// with. Only the frame extents and the capture time are read from the incoming frame; the pixels
/// are never touched.
class CameraInfoOp final : public holoscan::Operator<CameraInfoParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::ImageT> image;
  holoscan::Output<isaac::CameraInfoCpuT> camera_info;

 private:
  /// Calibration fields that do not change between frames, built once during start().
  isaac::CameraInfoCpuT prototype_;
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_CAMERA_INFO_CAMERA_INFO_OP_H
