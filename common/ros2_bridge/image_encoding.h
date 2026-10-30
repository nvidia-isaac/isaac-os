/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/sensor_io/sensor_schema_package.hpp"
#include <fmt/format.h>

// sensor_msgs/msg/Image encoding strings <-> holoscan::schema::ImageEncoding. Only the 5
// encodings sensor_msgs names as plain lowercase strings are mapped -- everything else
// holoscan::schema::ImageEncoding defines (multi-planar, float, Bayer, ...) has no ROS2
// equivalent and is treated as unsupported. Bytes-per-pixel for these same 5 encodings lives in
// common/image_schema/image_schema.h's bytes_per_pixel(), not here -- that table isn't
// ROS2-specific, unlike the string<->enum mapping below.
namespace isaac::os::ros2_bridge::detail {

[[nodiscard]] inline holoscan::expected<holoscan::schema::ImageEncoding, holoscan::Error>
ros2_encoding_to_hsdk(const std::string& encoding) {
  if (encoding == "rgb8") return holoscan::schema::ImageEncoding_RGB8;
  if (encoding == "rgba8") return holoscan::schema::ImageEncoding_RGBA8;
  if (encoding == "bgr8") return holoscan::schema::ImageEncoding_BGR8;
  if (encoding == "bgra8") return holoscan::schema::ImageEncoding_BGRA8;
  if (encoding == "mono8") return holoscan::schema::ImageEncoding_MONO8;
  return holoscan::make_unexpected(holoscan::Error{
      holoscan::ErrorCode::kInvalidArgument,
      fmt::format("Unsupported sensor_msgs/msg/Image encoding for the ROS2 bridge: '{}'.",
                  encoding)});
}

[[nodiscard]] inline std::string hsdk_encoding_to_ros2(
    const holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_RGB8:
      return "rgb8";
    case holoscan::schema::ImageEncoding_RGBA8:
      return "rgba8";
    case holoscan::schema::ImageEncoding_BGR8:
      return "bgr8";
    case holoscan::schema::ImageEncoding_BGRA8:
      return "bgra8";
    case holoscan::schema::ImageEncoding_MONO8:
      return "mono8";
    default:
      return "";
  }
}

}  // namespace isaac::os::ros2_bridge::detail
