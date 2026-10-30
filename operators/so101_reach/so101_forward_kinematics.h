/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_SO101_REACH_SO101_FORWARD_KINEMATICS_H
#define ISAAC_OS_OPERATORS_SO101_REACH_SO101_FORWARD_KINEMATICS_H

#include <array>

namespace isaac::os::ops {

/// Returns the SO-101 fingertip position in the robot-root frame.
[[nodiscard]] std::array<float, 3U> so101_fingertip_position(
    const std::array<float, 6U>& joint_position);

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_SO101_REACH_SO101_FORWARD_KINEMATICS_H
