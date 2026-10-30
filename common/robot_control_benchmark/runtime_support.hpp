/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>

#include "common/robot_control_benchmark/arm_kinematics.hpp"
#include "common/robot_control_benchmark/control_loop.hpp"

namespace isaac::os::benchmark {

/// Joint state owned by one serial control partition; callers must not access it concurrently.
class SimulatedRobot {
 public:
  [[nodiscard]] std::array<double, kNumJoints> read() const { return position_; }

  void write(const std::array<double, kNumJoints>& position) { position_ = position; }

 private:
  std::array<double, kNumJoints> position_ = ArmModel::home();
};

}  // namespace isaac::os::benchmark
