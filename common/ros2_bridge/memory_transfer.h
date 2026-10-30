/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/ros2_bridge/ros2_topic_spec.h"

namespace isaac::os::ros2_bridge {

// Where a block of pixels is right now. `device` is ignored for kHost.
struct Placement {
  Ros2MemoryKind kind = Ros2MemoryKind::kHost;
  int device = 0;
};

enum class TransferAction {
  kWrapDevice,    // use a CUDA buffer where it is, no copy
  kWrapHost,      // use a host buffer where it is, no copy
  kHostCopy,      // host to host
  kHostToDevice,  // into a pooled CUDA buffer on the pool device
  kDeviceToHost,
  kDeviceToDevice,  // into a pooled CUDA buffer on the pool device
  kReject,
};

// What to do with a buffer that arrived from ROS, given the topic's memory kind. The CUDA buffer
// pool is bound to one GPU (`pool_device`) and the backend cannot import a buffer that lives on
// another GPU, so a CUDA buffer is only usable in place when it is on the pool device.
[[nodiscard]] inline TransferAction PlanIngress(Placement source, Ros2MemoryKind target,
                                                int pool_device) {
  using enum TransferAction;
  const bool source_on_device = source.kind == Ros2MemoryKind::kCudaDevice;
  if (target == Ros2MemoryKind::kCudaDevice) {
    if (!source_on_device) {
      return kHostToDevice;
    }
    return source.device == pool_device ? kWrapDevice : kReject;
  }
  return source_on_device ? kDeviceToHost : kWrapHost;
}

// What to do with a tensor on its way to ROS. Every CUDA message the bridge publishes comes from
// the pool, so it is on the pool device whatever GPU the tensor is on.
[[nodiscard]] inline TransferAction PlanEgress(Placement source, Ros2MemoryKind target) {
  using enum TransferAction;
  const bool source_on_device = source.kind == Ros2MemoryKind::kCudaDevice;
  if (target == Ros2MemoryKind::kCudaDevice) {
    return source_on_device ? kDeviceToDevice : kHostToDevice;
  }
  return source_on_device ? kDeviceToHost : kHostCopy;
}

// The buffer backends an ingress subscription asks ROS for. CUDA buffers only when the topic is a
// device topic whose publisher sends GPU memory; otherwise host buffers, which also keeps a host
// publisher off the slower CUDA-aware receive path.
[[nodiscard]] inline const char* AcceptableBufferBackends(const Ros2TopicSpec& spec) {
  return spec.memory_kind == Ros2MemoryKind::kCudaDevice &&
                 spec.publisher_memory == Ros2MemoryKind::kCudaDevice
             ? "cuda"
             : "cpu";
}

}  // namespace isaac::os::ros2_bridge
