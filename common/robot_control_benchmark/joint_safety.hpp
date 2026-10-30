/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "common/robot_control_benchmark/control_loop.hpp"

namespace isaac::os::benchmark {

struct SafetyResult {
  std::uint32_t non_finite = 0;
  std::uint32_t range_clamps = 0;
  std::uint32_t step_clamps = 0;
  std::uint32_t recovery_steps = 0;

  [[nodiscard]] bool modified() const {
    return non_finite != 0 || range_clamps != 0 || step_clamps != 0 || recovery_steps != 0;
  }
};

class JointSafety {
 public:
  static constexpr double kMaxStepRad = 0.12;

  [[nodiscard]] SafetyResult validate_and_clamp(std::array<double, kNumJoints>& candidate,
                                                const JointState& measured) const {
    SafetyResult result;
    for (int joint = 0; joint < kNumJoints; ++joint) {
      double& command = candidate[static_cast<std::size_t>(joint)];
      const double current = measured.position[static_cast<std::size_t>(joint)];
      if (!std::isfinite(command)) {
        command = current;
        ++result.non_finite;
      }

      const double range_min = kMinPositionRad[joint];
      const double range_max = kMaxPositionRad[joint];
      const double step_min = current - kMaxStepRad;
      const double step_max = current + kMaxStepRad;
      const double feasible_min = std::max(range_min, step_min);
      const double feasible_max = std::min(range_max, step_max);

      if (feasible_min > feasible_max) {
        command = current < range_min ? step_max : step_min;
        ++result.recovery_steps;
        continue;
      }

      if (command < range_min || command > range_max) {
        ++result.range_clamps;
      }
      if (command < step_min || command > step_max) {
        ++result.step_clamps;
      }
      command = std::clamp(command, feasible_min, feasible_max);
    }
    return result;
  }

  // Representative KUKA LBR iiwa 7 position limits, also used to bound GPU IK candidates
  // before scoring them. Keep the solver and downstream safety stage on the same limits.
  static constexpr std::array<double, kNumJoints> kMinPositionRad{-2.967, -2.094, -2.967, -2.094,
                                                                  -2.967, -2.094, -3.054};
  static constexpr std::array<double, kNumJoints> kMaxPositionRad{2.967, 2.094, 2.967, 2.094,
                                                                  2.967, 2.094, 3.054};
};

}  // namespace isaac::os::benchmark
