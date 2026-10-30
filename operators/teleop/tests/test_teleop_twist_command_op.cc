/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <string>

#include <gtest/gtest.h>

#include "operators/teleop/teleop_twist_command_op.h"

namespace isaac::os::ops {
namespace {

constexpr float kLinSpeed = 0.8F;
constexpr float kAngSpeed = 0.5F;

std::array<float, 3> keys(bool forward = false, bool back = false, bool left = false,
                          bool right = false, bool turn_left = false, bool turn_right = false,
                          bool stop = false) {
  return keyboard_velocity(forward, back, left, right, turn_left, turn_right, stop, kLinSpeed,
                           kAngSpeed);
}

TEST(KeyboardVelocity, StandsStillWhenNothingIsHeld) {
  EXPECT_EQ(keys(), (std::array<float, 3>{0.0F, 0.0F, 0.0F}));
}

TEST(KeyboardVelocity, DrivesInTheDirectionOfTheHeldKey) {
  EXPECT_EQ(keys(/*forward=*/true), (std::array<float, 3>{kLinSpeed, 0.0F, 0.0F}));
  EXPECT_EQ(keys(false, /*back=*/true), (std::array<float, 3>{-kLinSpeed, 0.0F, 0.0F}));
  EXPECT_EQ(keys(false, false, /*left=*/true), (std::array<float, 3>{0.0F, kLinSpeed, 0.0F}));
  EXPECT_EQ(keys(false, false, false, /*right=*/true),
            (std::array<float, 3>{0.0F, -kLinSpeed, 0.0F}));
  EXPECT_EQ(keys(false, false, false, false, /*turn_left=*/true),
            (std::array<float, 3>{0.0F, 0.0F, kAngSpeed}));
  EXPECT_EQ(keys(false, false, false, false, false, /*turn_right=*/true),
            (std::array<float, 3>{0.0F, 0.0F, -kAngSpeed}));
}

TEST(KeyboardVelocity, CancelsOppositeKeys) {
  EXPECT_EQ(keys(true, true), (std::array<float, 3>{0.0F, 0.0F, 0.0F}));
  EXPECT_EQ(keys(false, false, true, true), (std::array<float, 3>{0.0F, 0.0F, 0.0F}));
  EXPECT_EQ(keys(false, false, false, false, true, true), (std::array<float, 3>{0.0F, 0.0F, 0.0F}));
}

TEST(KeyboardVelocity, StopsWhateverElseIsHeld) {
  EXPECT_EQ(keys(true, false, true, false, true, false, /*stop=*/true),
            (std::array<float, 3>{0.0F, 0.0F, 0.0F}));
}

TEST(DefaultGamepadMapping, KnowsTheSupportedDrivers) {
  const TeleopGamepadMapping* playstation = default_gamepad_mapping("playstation");
  const TeleopGamepadMapping* generic = default_gamepad_mapping("hid-generic");
  ASSERT_NE(playstation, nullptr);
  ASSERT_NE(generic, nullptr);
  EXPECT_EQ(playstation->axes.size(), 3U);
  EXPECT_EQ(generic->axes.size(), 3U);
  EXPECT_NE(playstation->deadman.axis, generic->deadman.axis);
  EXPECT_NE(playstation->lower_body_enable_button, generic->lower_body_enable_button);
}

TEST(DefaultGamepadMapping, HasNoLayoutForAnUnknownDriver) {
  EXPECT_EQ(default_gamepad_mapping(""), nullptr);
  EXPECT_EQ(default_gamepad_mapping("xpad"), nullptr);
}

TEST(GamepadArmingLatch, StartsDisarmed) {
  const GamepadArmingLatch latch;
  EXPECT_FALSE(latch.armed());
}

TEST(GamepadArmingLatch, TogglesOnEachPressRatherThanWhileHeld) {
  GamepadArmingLatch latch;
  EXPECT_TRUE(latch.update(/*enable_pressed=*/true, /*connected=*/true));
  EXPECT_TRUE(latch.armed());
  EXPECT_FALSE(latch.update(true, true));
  EXPECT_TRUE(latch.armed());
  EXPECT_FALSE(latch.update(false, true));
  EXPECT_TRUE(latch.armed());
  EXPECT_TRUE(latch.update(true, true));
  EXPECT_FALSE(latch.armed());
}

TEST(GamepadArmingLatch, DisarmsWhenTheGamepadIsLost) {
  GamepadArmingLatch latch;
  latch.update(true, true);
  ASSERT_TRUE(latch.armed());
  EXPECT_TRUE(latch.update(/*enable_pressed=*/false, /*connected=*/false));
  EXPECT_FALSE(latch.armed());
  latch.update(true, false);
  EXPECT_FALSE(latch.armed());
}

TEST(NormalizeAxis, AppliesTheDeadzoneAndClampsToFullDeflection) {
  EXPECT_FLOAT_EQ(EvdevDevice::normalize_axis(128, 0, 255, 0, 0.10, false), 0.0F);
  EXPECT_FLOAT_EQ(EvdevDevice::normalize_axis(255, 0, 255, 0, 0.10, false), 1.0F);
  EXPECT_FLOAT_EQ(EvdevDevice::normalize_axis(0, 0, 255, 0, 0.10, false), -1.0F);
  EXPECT_FLOAT_EQ(EvdevDevice::normalize_axis(255, 0, 255, 0, 0.10, true), -1.0F);
}

TEST(IsPs5Dualsense, RecognisesOnlyTheSupportedController) {
  EXPECT_FALSE(is_ps5_dualsense(0x0000, 0x0000));
  EXPECT_FALSE(is_ps5_dualsense(0x045e, 0x028e));  // A different vendor's controller.
}

}  // namespace
}  // namespace isaac::os::ops
