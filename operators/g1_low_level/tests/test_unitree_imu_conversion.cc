/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>

#include <gtest/gtest.h>

#include "operators/g1_low_level/unitree_imu_conversion.h"

namespace isaac::os {
namespace {

// Every component differs, so any rotation or transposition of the four values changes at least
// one assertion below. A symmetric fixture such as the identity quaternion would not.
constexpr float kW = 0.1F;
constexpr float kX = 0.2F;
constexpr float kY = 0.3F;
constexpr float kZ = 0.4F;

TEST(UnitreeImuConversion, OrientationKeepsEachComponentWithItsOwnAxis) {
  const holoscan::schema::Quaternionf orientation =
      to_schema_orientation({.w = kW, .x = kX, .y = kY, .z = kZ});

  // Read back by name: this holds whatever order the schema stores the components in.
  EXPECT_FLOAT_EQ(orientation.w(), kW);
  EXPECT_FLOAT_EQ(orientation.x(), kX);
  EXPECT_FLOAT_EQ(orientation.y(), kY);
  EXPECT_FLOAT_EQ(orientation.z(), kZ);
}

TEST(UnitreeImuConversion, OrientationMatchesTheSchemaWireOrder) {
  const holoscan::schema::Quaternionf orientation =
      to_schema_orientation({.w = kW, .x = kX, .y = kY, .z = kZ});

  // The schema documents its wire order as (x, y, z, w). Reading the struct as raw floats catches
  // a schema that reorders its components without renaming them, which a name-based read cannot.
  const auto* components = reinterpret_cast<const float*>(&orientation);
  EXPECT_FLOAT_EQ(components[0], kX);
  EXPECT_FLOAT_EQ(components[1], kY);
  EXPECT_FLOAT_EQ(components[2], kZ);
  EXPECT_FLOAT_EQ(components[3], kW);
  static_assert(sizeof(holoscan::schema::Quaternionf) == 4 * sizeof(float),
                "The orientation must stay four packed floats for the raw read above to hold.");
}

TEST(UnitreeImuConversion, IdentityIsARotationRatherThanAZeroNormQuaternion) {
  const holoscan::schema::Quaternionf identity = identity_orientation();

  EXPECT_FLOAT_EQ(identity.w(), 1.0F);
  EXPECT_FLOAT_EQ(identity.x(), 0.0F);
  EXPECT_FLOAT_EQ(identity.y(), 0.0F);
  EXPECT_FLOAT_EQ(identity.z(), 0.0F);

  // A misordered identity is all zeros in the real part, which has zero norm and normalizes to
  // NaN rather than describing any rotation at all.
  const float norm = std::sqrt((identity.w() * identity.w()) + (identity.x() * identity.x()) +
                               (identity.y() * identity.y()) + (identity.z() * identity.z()));
  EXPECT_FLOAT_EQ(norm, 1.0F);
}

TEST(UnitreeImuConversion, AnUprightRobotReportsANearUnitRealPart) {
  // Captured from a G1 standing still in simulation. It is close to identity but not exactly so:
  // the vector part carries roughly a degree and a half of heading, which is an ordinary resting
  // pose rather than an error. The tolerances below are wide enough to accept a robot that is
  // merely not square to the world, and far too tight to accept a transposed component.
  const holoscan::schema::Quaternionf upright = to_schema_orientation(
      {.w = 0.9999174F, .x = -0.0023117382F, .y = 3.2855845e-05F, .z = -0.012642204F});

  constexpr float kRestingPoseSlack = 0.05F;
  EXPECT_NEAR(upright.w(), 1.0F, kRestingPoseSlack);
  EXPECT_NEAR(upright.x(), 0.0F, kRestingPoseSlack);
  EXPECT_NEAR(upright.y(), 0.0F, kRestingPoseSlack);
  EXPECT_NEAR(upright.z(), 0.0F, kRestingPoseSlack);

  // The invariant the tolerances above are standing in for, stated without one: an upright robot
  // puts its rotation in the real part. Any transposition moves a near-unit value into the vector
  // part, so requiring the real part to dominate catches the failure whatever the resting heading.
  EXPECT_GT(upright.w(), 0.9F);
  EXPECT_LT(std::abs(upright.x()), 0.1F);
  EXPECT_LT(std::abs(upright.y()), 0.1F);
  EXPECT_LT(std::abs(upright.z()), 0.1F);
}

TEST(UnitreeImuConversion, ThreeAxisReadingsKeepTheirOrder) {
  const holoscan::schema::Vector3f value = to_schema_vector({.x = 1.0F, .y = 2.0F, .z = 3.0F});

  EXPECT_FLOAT_EQ(value.x(), 1.0F);
  EXPECT_FLOAT_EQ(value.y(), 2.0F);
  EXPECT_FLOAT_EQ(value.z(), 3.0F);

  const auto* components = reinterpret_cast<const float*>(&value);
  EXPECT_FLOAT_EQ(components[0], 1.0F);
  EXPECT_FLOAT_EQ(components[1], 2.0F);
  EXPECT_FLOAT_EQ(components[2], 3.0F);
  static_assert(sizeof(holoscan::schema::Vector3f) == 3 * sizeof(float),
                "The vector must stay three packed floats for the raw read above to hold.");
}

}  // namespace
}  // namespace isaac::os
