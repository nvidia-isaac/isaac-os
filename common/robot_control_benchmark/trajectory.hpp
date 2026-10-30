/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Circular XY trajectory at fixed height. motion_hz is the tool sweep rate, not
// the control tick rate.

#include <cmath>
#include <numbers>

#include "common/robot_control_benchmark/payloads.hpp"

namespace isaac::os::benchmark {

struct TrajectoryParams {
  double center_x = 0.45;
  double center_z = 0.50;
  double radius = 0.15;
  double motion_hz = 0.5;
};

inline Pose6 trajectory_pose(double t_sec, const TrajectoryParams& p = {}) {
  const double w = 2.0 * std::numbers::pi * p.motion_hz * t_sec;
  Pose6 pose{};
  pose.x = p.center_x + p.radius * std::cos(w);
  pose.y = p.radius * std::sin(w);
  pose.z = p.center_z;
  pose.rx = std::numbers::pi;
  pose.ry = 0.0;
  pose.rz = 0.0;
  return pose;
}

}  // namespace isaac::os::benchmark
