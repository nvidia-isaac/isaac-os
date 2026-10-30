/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/so101_reach/so101_forward_kinematics.h"

#include <array>
#include <cmath>
#include <cstddef>

namespace isaac::os::ops {
namespace {

using Matrix = std::array<double, 9U>;
using Vector = std::array<double, 3U>;

struct Link {
  std::array<double, 4U> orientation;
  Vector position;
};

constexpr Matrix kIdentity{
    1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
};

// These fixed transforms and the fingertip offset are from the SO-101 URDF.
constexpr std::array<Link, 5U> kArmLinks{{
    {{1.76036e-12, 1.32679e-06, -1.0, -1.32679e-06}, {0.0388353, -8.97657e-09, 0.0624}},
    {{0.499998, -0.5, -0.5, -0.500002}, {-0.0303992, -0.0182778, -0.0542}},
    {{0.707105, 0.0, 0.0, 0.707108}, {-0.11257, -0.028, 0.0}},
    {{0.707105, 0.0, 0.0, -0.707108}, {-0.1349, 0.0052, 0.0}},
    {{0.0172101, -0.0172081, 0.706899, 0.706896}, {0.0, -0.0611, 0.0181}},
}};

constexpr Vector kFingertipOffset{-0.0079, -0.000218, -0.0981};

Matrix multiply(const Matrix& lhs, const Matrix& rhs) {
  Matrix result{};
  for (std::size_t row = 0U; row < 3U; ++row) {
    for (std::size_t column = 0U; column < 3U; ++column) {
      for (std::size_t index = 0U; index < 3U; ++index) {
        result[row * 3U + column] += lhs[row * 3U + index] * rhs[index * 3U + column];
      }
    }
  }
  return result;
}

Vector rotate(const Matrix& rotation, const Vector& vector) {
  return {
      rotation[0U] * vector[0U] + rotation[1U] * vector[1U] + rotation[2U] * vector[2U],
      rotation[3U] * vector[0U] + rotation[4U] * vector[1U] + rotation[5U] * vector[2U],
      rotation[6U] * vector[0U] + rotation[7U] * vector[1U] + rotation[8U] * vector[2U],
  };
}

Matrix rotation_from_quaternion(const std::array<double, 4U>& quaternion) {
  const double w = quaternion[0U];
  const double x = quaternion[1U];
  const double y = quaternion[2U];
  const double z = quaternion[3U];
  return {
      1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
      2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
      2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y),
  };
}

Matrix rotation_about_z(double angle) {
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  return {
      cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0,
  };
}

Vector add(const Vector& lhs, const Vector& rhs) {
  return {lhs[0U] + rhs[0U], lhs[1U] + rhs[1U], lhs[2U] + rhs[2U]};
}

}  // namespace

std::array<float, 3U> so101_fingertip_position(const std::array<float, 6U>& joint_position) {
  Matrix orientation = kIdentity;
  Vector position{};
  for (std::size_t index = 0U; index < kArmLinks.size(); ++index) {
    const Link& link = kArmLinks[index];
    position = add(position, rotate(orientation, link.position));
    orientation = multiply(orientation, rotation_from_quaternion(link.orientation));
    orientation = multiply(orientation, rotation_about_z(joint_position[index]));
  }
  const Vector fingertip = add(position, rotate(orientation, kFingertipOffset));
  return {static_cast<float>(fingertip[0U]), static_cast<float>(fingertip[1U]),
          static_cast<float>(fingertip[2U])};
}

}  // namespace isaac::os::ops
