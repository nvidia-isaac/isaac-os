/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace isaac::so101 {

inline constexpr std::array<std::string_view, 6U> kSo101JointNames{
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"};

struct MotorCalibration {
  std::uint8_t id = 0;
  int homing_offset = 0;
  std::uint16_t range_min = 0;
  std::uint16_t range_max = 0;
  std::uint8_t drive_mode = 0;
};

}  // namespace isaac::so101
