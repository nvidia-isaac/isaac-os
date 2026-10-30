/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/camera_info/camera_info_op.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

#include <holoscan/sensor_io/provenance.hpp>

namespace isaac::os::ops {

namespace {

/// Maps a calibration file's model name onto the sensor_msgs name for the same model.
///
/// The two vocabularies differ, and the published message mirrors sensor_msgs/CameraInfo, so the
/// name travels in the form a consumer of that message expects rather than in the form the
/// calibration file happens to use.
std::string to_sensor_msgs_distortion_model(const std::string& name) {
  if (name == "none") {
    return "";
  }
  if (name == "brown") {
    return "plumb_bob";
  }
  if (name == "fisheye") {
    return "equidistant";
  }
  if (name == "polynomial") {
    return "rational_polynomial";
  }
  throw std::runtime_error("The camera calibration names an unsupported distortion model '" + name +
                           "'. Choose none, brown, fisheye, or polynomial.");
}

}  // namespace

void CameraInfoOp::setup(holoscan::OperatorSpec& spec) {
  // Only the descriptor is read, so the port states no tensor requirement: constraining the pixel
  // layout here would reject producers this operator is indifferent to.
  spec.input(image, "in_image").queue_depth(1U);
  spec.output(camera_info, "out_camera_info");
}

holoscan::Contract CameraInfoOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{image});
  return contract;
}

void CameraInfoOp::start() {
  const auto& config = params();
  if (config.frame_id.empty()) {
    // Says what the absence implies, because the runtime reduces a failed start to a token and
    // drops this text: a frame name has no default, so an empty one means the configuration
    // section was either missing or never applied, not that someone configured it empty.
    throw std::runtime_error(
        "A camera calibration does not name a coordinate frame. A frame name has no default, so "
        "its configuration section was either absent or not applied.");
  }
  prototype_ = isaac::CameraInfoCpuT{};
  prototype_.header = std::make_shared<holoscan::schema::HeaderT>();
  prototype_.header->frame_id = config.frame_id;
  prototype_.distortion_model = to_sensor_msgs_distortion_model(config.distortion_model);
  prototype_.d = config.distortion_parameters;

  const double fx = config.fx;
  const double fy = config.fy;
  const double cx = config.cx;
  const double cy = config.cy;

  prototype_.k =
      holoscan::schema::Matrix3d(std::array<double, 9>{fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0});
  // These frames are already rectified, so rectifying them again is the identity. The field is
  // required, and an all-zero matrix there would read as a degenerate rectification rather than as
  // an absent one.
  prototype_.r = holoscan::schema::Matrix3d(
      std::array<double, 9>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
  // The fourth column places this camera relative to the reference camera of a rectified pair. It
  // is zero for the reference camera itself and -fx * baseline for the other one.
  prototype_.p << fx, 0.0, cx, -fx * config.stereo_baseline,  //
      0.0, fy, cy, 0.0,                                       //
      0.0, 0.0, 1.0, 0.0;
}

holoscan::expected<void, holoscan::Error> CameraInfoOp::compute(
    holoscan::ExecutionContext& /*context*/) {
  auto sample = image.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }

  // The prototype's header is shared, so each message gets its own copy of it before the per-frame
  // capture time is written in: a shared one would let every message already emitted appear to
  // have been captured whenever the newest one was.
  isaac::CameraInfoCpuT message = prototype_;
  message.header = std::make_shared<holoscan::schema::HeaderT>(*prototype_.header);
  message.height = sample->data.height;
  message.width = sample->data.width;
  if (sample->data.header) {
    message.header->capture_timestamp_ns = sample->data.header->capture_timestamp_ns;
  }
  return camera_info.emit(message, holoscan::sensor_io::derive_from(*sample));
}

}  // namespace isaac::os::ops
