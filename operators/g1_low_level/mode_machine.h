/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_G1_LOW_LEVEL_MODE_MACHINE_H
#define ISAAC_OS_OPERATORS_G1_LOW_LEVEL_MODE_MACHINE_H

#include <atomic>
#include <cstdint>

namespace isaac::os {

/// Shares the robot's reported control mode between the low-level subscriber and publisher.
///
/// LowState carries the mode the robot is currently in and LowCmd has to echo it back, so the
/// subscriber writes this value from its DDS callback thread and the publisher reads it from the
/// compute thread.
class ModeMachine {
 public:
  explicit ModeMachine(std::uint8_t initial_value = 6) : value_(initial_value) {}

  void set(std::uint8_t mode) { value_.store(mode); }

  [[nodiscard]] std::uint8_t get() const { return value_.load(); }

 private:
  std::atomic<std::uint8_t> value_;
};

}  // namespace isaac::os

#endif  // ISAAC_OS_OPERATORS_G1_LOW_LEVEL_MODE_MACHINE_H
