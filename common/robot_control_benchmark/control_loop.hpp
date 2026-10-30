/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <chrono>
#include <cstdint>

#include "common/robot_control_benchmark/payloads.hpp"

namespace isaac::os::benchmark {

inline std::int64_t steady_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct JointState {
  std::int64_t sequence = 0;
  std::array<double, kNumJoints> position{};
};

}  // namespace isaac::os::benchmark
