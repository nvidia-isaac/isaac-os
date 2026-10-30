/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <numbers>

#include <gtest/gtest.h>

#include "common/so101_driver/so101_unit_conversions.h"

namespace isaac::so101 {
namespace {

TEST(So101UnitConversions, HonorsCalibrationBoundsAndDirection) {
  constexpr MotorCalibration kNormal{
      .range_min = 1024U,
      .range_max = 3072U,
  };
  constexpr MotorCalibration kReversed{
      .range_min = 1024U,
      .range_max = 3072U,
      .drive_mode = 1U,
  };
  constexpr float kHalfPi = std::numbers::pi_v<float> / 2.0F;

  EXPECT_EQ(UnitConversions::rad_to_encoder(0.0F, kNormal), 2048);
  EXPECT_EQ(UnitConversions::rad_to_encoder(std::numbers::pi_v<float>, kNormal), 3072);
  EXPECT_NEAR(UnitConversions::encoder_to_rad(3072, kNormal), kHalfPi, 1.0e-5F);

  EXPECT_EQ(UnitConversions::rad_to_encoder(kHalfPi, kReversed), 1024);
  EXPECT_NEAR(UnitConversions::encoder_to_rad(1024, kReversed), kHalfPi, 1.0e-5F);
  const auto [lower, upper] = UnitConversions::reachable_rad_range(kReversed);
  EXPECT_NEAR(lower, -kHalfPi, 1.0e-5F);
  EXPECT_NEAR(upper, kHalfPi, 1.0e-5F);
}

TEST(So101UnitConversions, UsesCalibratedClosedEndpointAsGripperZero) {
  constexpr MotorCalibration kNormal{
      .range_min = 1796U,
      .range_max = 3244U,
  };
  constexpr MotorCalibration kReversed{
      .range_min = 1796U,
      .range_max = 3244U,
      .drive_mode = 1U,
  };
  constexpr float kTravel = static_cast<float>(kNormal.range_max - kNormal.range_min) *
                            UnitConversions::kRadiansPerEncoderCount;

  EXPECT_FLOAT_EQ(UnitConversions::gripper_encoder_to_rad(1796, kNormal), 0.0F);
  EXPECT_NEAR(UnitConversions::gripper_encoder_to_rad(3244, kNormal), kTravel, 1.0e-5F);
  EXPECT_EQ(UnitConversions::gripper_rad_to_encoder(0.0F, kNormal), 1796);
  EXPECT_EQ(UnitConversions::gripper_rad_to_encoder(kTravel, kNormal), 3244);

  EXPECT_FLOAT_EQ(UnitConversions::gripper_encoder_to_rad(3244, kReversed), 0.0F);
  EXPECT_NEAR(UnitConversions::gripper_encoder_to_rad(1796, kReversed), kTravel, 1.0e-5F);
  EXPECT_EQ(UnitConversions::gripper_rad_to_encoder(0.0F, kReversed), 3244);
  EXPECT_EQ(UnitConversions::gripper_rad_to_encoder(kTravel, kReversed), 1796);

  EXPECT_EQ(UnitConversions::gripper_rad_to_encoder(-0.2F, kNormal), 1796);
  EXPECT_EQ(UnitConversions::gripper_rad_to_encoder(kTravel + 0.2F, kNormal), 3244);
  EXPECT_EQ(UnitConversions::gripper_rad_to_encoder(-0.2F, kReversed), 3244);
  EXPECT_EQ(UnitConversions::gripper_rad_to_encoder(kTravel + 0.2F, kReversed), 1796);

  const auto [normal_lower, normal_upper] = UnitConversions::gripper_reachable_rad_range(kNormal);
  const auto [reversed_lower, reversed_upper] =
      UnitConversions::gripper_reachable_rad_range(kReversed);
  EXPECT_FLOAT_EQ(normal_lower, 0.0F);
  EXPECT_NEAR(normal_upper, kTravel, 1.0e-5F);
  EXPECT_FLOAT_EQ(reversed_lower, 0.0F);
  EXPECT_NEAR(reversed_upper, kTravel, 1.0e-5F);
}

}  // namespace
}  // namespace isaac::so101
