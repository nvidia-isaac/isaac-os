/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <numbers>

#include "common/so101_driver/so101_types.h"

namespace isaac::so101 {

/// Converts STS-3215 encoder values to calibrated SO101 joint units.
///
/// Arm-joint positions are radians about the calibrated zero at encoder count 2048. The gripper
/// helpers instead use its calibrated closed endpoint as zero, matching LeRobot's range-based
/// gripper convention. Speeds are encoder counts per second and therefore have no center offset.
class UnitConversions {
 public:
  static constexpr float kEncoderCenter = 2048.0f;
  static constexpr float kRadiansPerEncoderCount = std::numbers::pi_v<float> / kEncoderCenter;

  /// Converts a calibrated encoder position to radians, clamping to joint limits.
  static float encoder_to_rad(std::int16_t value, const MotorCalibration& calibration);
  /// Converts a radian position to a calibrated, joint-limit-clamped encoder position.
  static std::int16_t rad_to_encoder(float value, const MotorCalibration& calibration);
  /// Converts a gripper encoder position to radians from its calibrated closed endpoint.
  static float gripper_encoder_to_rad(std::int16_t value, const MotorCalibration& calibration);
  /// Converts gripper radians from closed to a calibrated, joint-limit-clamped encoder position.
  static std::int16_t gripper_rad_to_encoder(float value, const MotorCalibration& calibration);
  /// Converts a signed encoder speed in counts per second to radians per second.
  static float encoder_speed_to_rad_per_s(std::int16_t value, const MotorCalibration& calibration);
  /// Returns the calibrated joint limits in ascending radians.
  static std::array<float, 2> reachable_rad_range(const MotorCalibration& calibration);
  /// Returns the gripper's calibrated range as non-negative radians from closed.
  static std::array<float, 2> gripper_reachable_rad_range(const MotorCalibration& calibration);
};

}  // namespace isaac::so101
