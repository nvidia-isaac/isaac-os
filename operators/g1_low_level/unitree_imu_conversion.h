/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_G1_LOW_LEVEL_UNITREE_IMU_CONVERSION_H
#define ISAAC_OS_OPERATORS_G1_LOW_LEVEL_UNITREE_IMU_CONVERSION_H

#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace isaac::os {

/// Orientation as Unitree LowState reports it, named so call sites cannot transpose components.
///
/// The schema's own quaternion is generated without mutators and stores its real part last, so it
/// can only be built from a positional argument list. Naming the components here moves that hazard
/// off every call site and into one converted function.
struct UnitreeOrientation {
  float w{};
  float x{};
  float y{};
  float z{};
};

/// Three-axis reading as Unitree LowState reports it, named for the same reason.
struct UnitreeVector {
  float x{};
  float y{};
  float z{};
};

/// Converts a LowState orientation into the sensor schema's quaternion.
///
/// LowState orders the components (w, x, y, z) and the schema stores them (x, y, z, w). Both are
/// four floats, so a positional copy between them compiles and silently rotates the robot's
/// reported attitude by one component.
[[nodiscard]] holoscan::schema::Quaternionf to_schema_orientation(
    const UnitreeOrientation& orientation);

/// Returns the schema's identity rotation, whose real part is the last component.
[[nodiscard]] holoscan::schema::Quaternionf identity_orientation();

/// Converts a LowState three-axis reading into the sensor schema's vector.
[[nodiscard]] holoscan::schema::Vector3f to_schema_vector(const UnitreeVector& value);

}  // namespace isaac::os

#endif  // ISAAC_OS_OPERATORS_G1_LOW_LEVEL_UNITREE_IMU_CONVERSION_H
