/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ISAAC_OS_COMMON_POSE_TREE_MATH_INTERPOLATION_HPP
#define ISAAC_OS_COMMON_POSE_TREE_MATH_INTERPOLATION_HPP

#include <cmath>
#include <numbers>

#include "common/pose_tree/math/pose2.hpp"
#include "common/pose_tree/math/pose3.hpp"
#include "common/pose_tree/math/so2.hpp"
#include "common/pose_tree/math/so3.hpp"

namespace isaac::os::pose_tree {
namespace math {

/**
 * @brief Linear interpolation between two values.
 *
 * Returns a value between a and b at the relative position q.
 * This function will also work for q outside of the unit interval for extrapolation.
 *
 * @tparam K Scalar type for interpolation parameter.
 * @tparam T Type of values to interpolate.
 * @param q Interpolation parameter (0.0 returns a, 1.0 returns b).
 * @param a First value.
 * @param b Second value.
 * @return Interpolated value a + q * (b - a).
 */
template <typename K, typename T>
T interpolate(K q, T a, T b) {
  return a + q * (b - a);
}

}  // namespace math

/**
 * @brief Interpolate between two 2D rotations.
 *
 * Due to the nature of rotations this function is problematic if the two rotations are
 * about 180 degrees apart. In that case small deviations in the input may have large
 * deviations in the output.
 *
 * @tparam K Scalar type.
 * @param p Interpolation parameter (0.0 returns a, 1.0 returns b).
 * @param a First rotation.
 * @param b Second rotation.
 * @return Interpolated rotation.
 */
template <typename K>
SO2<K> interpolate(K p, const SO2<K>& a, const SO2<K>& b) {
  const K a0 = a.angle();
  const K a1 = b.angle();
  // The difference is wrapped so that interpolation takes the shorter way round: without it, two
  // angles either side of the discontinuity would be read as nearly a full turn apart and the
  // result would sweep the long way. std::remainder is that wrap, exactly -- it returns the
  // difference to the nearest multiple of a turn, which lies in [-pi, pi] by construction.
  return SO2<K>::from_angle(a0 + p * std::remainder(a1 - a0, K{2} * std::numbers::pi_v<K>));
}

/**
 * @brief Interpolate between two 3D rotations using spherical linear interpolation (slerp).
 *
 * @tparam K Scalar type.
 * @param p Interpolation parameter (0.0 returns a, 1.0 returns b).
 * @param a First rotation.
 * @param b Second rotation.
 * @return Interpolated rotation using quaternion slerp.
 */
template <typename K>
SO3<K> interpolate(K p, const SO3<K>& a, const SO3<K>& b) {
  return SO3<K>::from_quaternion(a.quaternion().slerp(p, b.quaternion()));
}

/**
 * @brief Interpolate between two 2D poses using independent interpolation.
 *
 * This uses "independent" interpolation of translation and rotation. This is only one of
 * multiple ways to interpolate between two rigid body poses.
 *
 * @tparam K Scalar type.
 * @param p Interpolation parameter (0.0 returns a, 1.0 returns b).
 * @param a First pose.
 * @param b Second pose.
 * @return Interpolated pose.
 */
template <typename K>
Pose2<K> interpolate(K p, const Pose2<K>& a, const Pose2<K>& b) {
  return Pose2<K>{interpolate(p, a.rotation, b.rotation),
                  math::interpolate(p, a.translation, b.translation)};
}

/**
 * @brief Interpolate between two 3D poses using independent interpolation.
 *
 * This uses "independent" interpolation of translation and rotation. This is only one of
 * multiple ways to interpolate between two rigid body poses.
 *
 * @tparam K Scalar type.
 * @param p Interpolation parameter (0.0 returns a, 1.0 returns b).
 * @param a First pose.
 * @param b Second pose.
 * @return Interpolated pose.
 */
template <typename K>
Pose3<K> interpolate(K p, const Pose3<K>& a, const Pose3<K>& b) {
  return Pose3<K>{interpolate(p, a.rotation, b.rotation),
                  math::interpolate(p, a.translation, b.translation)};
}

/**
 * @brief Spherical linear interpolation between two 2D poses.
 *
 * This uses the formula a^(1-p) * b^p where a and b are matrix transformations.
 * This can be simplified into a * (a^-1 * b)^p
 *
 * @tparam K Scalar type.
 * @param p Interpolation parameter (0.0 returns a, 1.0 returns b).
 * @param a First pose.
 * @param b Second pose.
 * @return Interpolated pose using matrix exponentiation.
 */
template <typename K>
Pose2<K> slerp_interpolate(K p, const Pose2<K>& a, const Pose2<K>& b) {
  const Pose2<K> delta = a.inverse() * b;
  return a * delta.pow(p);
}

/**
 * @brief Spherical linear interpolation between two 3D poses.
 *
 * This uses the formula a^(1-p) * b^p where a and b are matrix transformations.
 * This can be simplified into a * (a^-1 * b)^p
 *
 * @tparam K Scalar type.
 * @param p Interpolation parameter (0.0 returns a, 1.0 returns b).
 * @param a First pose.
 * @param b Second pose.
 * @return Interpolated pose using matrix exponentiation.
 */
template <typename K>
Pose3<K> slerp_interpolate(K p, const Pose3<K>& a, const Pose3<K>& b) {
  const Pose3<K> delta = a.inverse() * b;
  return a * delta.pow(p);
}

}  // namespace isaac::os::pose_tree

#endif /* ISAAC_OS_COMMON_POSE_TREE_MATH_INTERPOLATION_HPP */
