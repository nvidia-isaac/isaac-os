/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/so101_driver/so101_unit_conversions.h"

#include <algorithm>
#include <cmath>

namespace isaac::so101 {

namespace {

float direction_sign(const MotorCalibration& calibration) {
  // drive_mode == 1 reverses the motor's calibrated direction.
  return calibration.drive_mode == 1 ? -1.0f : 1.0f;
}

float gripper_closed_encoder(const MotorCalibration& calibration) {
  // LeRobot's RANGE_0_100 convention assigns zero to range_min in the normal direction and to
  // range_max when the calibrated direction is reversed.
  return calibration.drive_mode == 1 ? static_cast<float>(calibration.range_max)
                                     : static_cast<float>(calibration.range_min);
}

float encoder_to_rad_from_origin(std::int16_t value, const MotorCalibration& calibration,
                                 float origin) {
  const std::int16_t bounded = std::clamp(value, static_cast<std::int16_t>(calibration.range_min),
                                          static_cast<std::int16_t>(calibration.range_max));
  return direction_sign(calibration) * (static_cast<float>(bounded) - origin) *
         UnitConversions::kRadiansPerEncoderCount;
}

std::int16_t rad_to_encoder_from_origin(float value, const MotorCalibration& calibration,
                                        float origin) {
  const float encoder =
      direction_sign(calibration) * value / UnitConversions::kRadiansPerEncoderCount + origin;
  const int clamped =
      std::clamp(static_cast<int>(std::round(encoder)), static_cast<int>(calibration.range_min),
                 static_cast<int>(calibration.range_max));
  return static_cast<std::int16_t>(clamped);
}

}  // namespace

float UnitConversions::encoder_to_rad(std::int16_t value, const MotorCalibration& calibration) {
  return encoder_to_rad_from_origin(value, calibration, kEncoderCenter);
}

std::int16_t UnitConversions::rad_to_encoder(float value, const MotorCalibration& calibration) {
  return rad_to_encoder_from_origin(value, calibration, kEncoderCenter);
}

float UnitConversions::gripper_encoder_to_rad(std::int16_t value,
                                              const MotorCalibration& calibration) {
  return encoder_to_rad_from_origin(value, calibration, gripper_closed_encoder(calibration));
}

std::int16_t UnitConversions::gripper_rad_to_encoder(float value,
                                                     const MotorCalibration& calibration) {
  return rad_to_encoder_from_origin(value, calibration, gripper_closed_encoder(calibration));
}

float UnitConversions::encoder_speed_to_rad_per_s(std::int16_t value,
                                                  const MotorCalibration& calibration) {
  return direction_sign(calibration) * static_cast<float>(value) * kRadiansPerEncoderCount;
}

std::array<float, 2> UnitConversions::reachable_rad_range(const MotorCalibration& calibration) {
  const float lower = encoder_to_rad(static_cast<std::int16_t>(calibration.range_min), calibration);
  const float upper = encoder_to_rad(static_cast<std::int16_t>(calibration.range_max), calibration);
  return {std::min(lower, upper), std::max(lower, upper)};
}

std::array<float, 2> UnitConversions::gripper_reachable_rad_range(
    const MotorCalibration& calibration) {
  const float lower =
      gripper_encoder_to_rad(static_cast<std::int16_t>(calibration.range_min), calibration);
  const float upper =
      gripper_encoder_to_rad(static_cast<std::int16_t>(calibration.range_max), calibration);
  return {std::min(lower, upper), std::max(lower, upper)};
}

}  // namespace isaac::so101
