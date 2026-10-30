/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_COMMON_JOINT_COMMAND_CHUNK_CPU_JOINT_COMMAND_CHUNK_CPU_H
#define ISAAC_OS_COMMON_JOINT_COMMAND_CHUNK_CPU_JOINT_COMMAND_CHUNK_CPU_H

#include <cstddef>
#include <limits>

#include "joint_command_chunk_cpu_generated.h"

namespace isaac::os {

[[nodiscard]] inline bool validate_joint_command_chunk(
    const isaac::JointCommandChunkCpuT& message) noexcept {
  if (message.joint_count == 0U || message.action_count == 0U ||
      message.joint_count > std::numeric_limits<std::size_t>::max() / message.action_count) {
    return false;
  }
  const std::size_t value_count =
      static_cast<std::size_t>(message.joint_count) * message.action_count;
  return message.position.size() == value_count && message.effort.size() == value_count &&
         message.kp.size() == message.joint_count && message.kd.size() == message.joint_count;
}

}  // namespace isaac::os

#endif  // ISAAC_OS_COMMON_JOINT_COMMAND_CHUNK_CPU_JOINT_COMMAND_CHUNK_CPU_H
