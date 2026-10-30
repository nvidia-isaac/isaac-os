/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_CUVSLAM_CALIBRATION_MAPPING_H
#define ISAAC_OS_OPERATORS_CUVSLAM_CALIBRATION_MAPPING_H

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace isaac::os::ops {

/// Reorder plumb_bob distortion coefficients into the order cuVSLAM's Brown model takes.
///
/// sensor_msgs/CameraInfo states plumb_bob as (k1, k2, p1, p2, k3), the order OpenCV uses.
/// cuVSLAM's Brown model takes the three radial terms first and the two tangential ones after:
/// (k1, k2, k3, p1, p2). Copied straight across, three of the five coefficients land on the wrong
/// term and the distortion is quietly wrong rather than rejected.
///
/// @param coefficients Distortion coefficients as published, which must number exactly five.
/// @return The coefficients in Brown order, or nothing if there are not five of them.
[[nodiscard]] inline std::optional<std::array<float, 5>> brown_from_plumb_bob(
    std::span<const double> coefficients) {
  if (coefficients.size() != 5) {
    return std::nullopt;
  }
  const auto value = [&](std::size_t index) { return static_cast<float>(coefficients[index]); };
  // (k1, k2, p1, p2, k3) -> (k1, k2, k3, p1, p2)
  return std::array<float, 5>{value(0), value(1), value(4), value(2), value(3)};
}

/// Convert distortion coefficients that need no reordering.
///
/// cuVSLAM's Polynomial model takes the first eight OpenCV coefficients in OpenCV's own order,
/// which is what sensor_msgs calls rational_polynomial, and its Fisheye model takes the four
/// equidistant coefficients in theirs. Neither needs rearranging, so both come through here.
[[nodiscard]] inline std::vector<float> narrow(std::span<const double> coefficients) {
  return std::vector<float>(coefficients.begin(), coefficients.end());
}

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_CUVSLAM_CALIBRATION_MAPPING_H
