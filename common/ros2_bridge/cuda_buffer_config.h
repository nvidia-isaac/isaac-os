/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "holoscan/core/errors.hpp"

#include "common/ros2_bridge/ros2_topic_spec.h"

namespace isaac::os::ros2_bridge {

// True when this build of the bridge links the CUDA buffer backend (the cuda_buffer package was
// found at configure time).
#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
inline constexpr bool kCudaBufferBackendBuiltIn = true;
#else
inline constexpr bool kCudaBufferBackendBuiltIn = false;
#endif

// Throws when a topic asks for device memory in a build that cannot provide it, so the check
// runs when the topic is wired instead of failing confusingly later. A RuntimeError, so that Core
// keeps the message when this is thrown from inside a graph's compose().
inline void RequireCudaBufferBackend(const Ros2TopicSpec& spec,
                                     bool built_in = kCudaBufferBackendBuiltIn) {
  if (spec.memory_kind == Ros2MemoryKind::kCudaDevice && !built_in) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kNotSupported,
        "ROS2 bridge topic '" + spec.port_name +
            "' asks for memory_kind kCudaDevice, but this build of the bridge has no CUDA buffer "
            "backend (the cuda_buffer package was not found at configure time). Use kHost, or "
            "build in an environment that provides cuda_buffer.");
  }
}

}  // namespace isaac::os::ros2_bridge
