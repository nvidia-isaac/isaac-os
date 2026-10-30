/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include <cuda_buffer/cuda_buffer_api.hpp>
#include <cuda_runtime.h>
#include <fmt/format.h>
#endif

namespace isaac::os::ros2_bridge {

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER

// Makes `device` the calling thread's current CUDA device for a scope and restores the previous
// one. Does nothing when `device` is already current (about 30 ns).
class ScopedDevice {
 public:
  explicit ScopedDevice(int device) {
    if (cudaGetDevice(&previous_) != cudaSuccess) return;
    if (previous_ != device && cudaSetDevice(device) == cudaSuccess) {
      changed_ = true;
    }
  }
  ~ScopedDevice() {
    if (changed_) {
      cudaSetDevice(previous_);
    }
  }
  ScopedDevice(const ScopedDevice&) = delete;
  ScopedDevice& operator=(const ScopedDevice&) = delete;

 private:
  int previous_ = 0;
  bool changed_ = false;
};

namespace detail {
// The device the CUDA buffer pool is bound to; -1 until something binds it.
inline std::atomic<int>& PoolDeviceStorage() {
  static std::atomic<int> device{-1};
  return device;
}

// Why a helper that needs the pool refused to run.
inline std::string PoolUnboundMessage() {
  return "The CUDA buffer pool is not bound to a GPU yet. Start the ROS2 bridge, which binds it to "
         "its pool_device, or call BindPoolDevice before converting CUDA buffers.";
}
}  // namespace detail

// True once the pool has been bound to a GPU. The helpers refuse to guess a GPU before that.
[[nodiscard]] inline bool PoolDeviceBound() { return detail::PoolDeviceStorage().load() >= 0; }

// The one GPU every CUDA topic uses. Throws while the pool is unbound; check PoolDeviceBound()
// first where unbound is an expected state.
[[nodiscard]] inline int PoolDevice() {
  const int device = detail::PoolDeviceStorage().load();
  if (device < 0) {
    throw std::runtime_error(detail::PoolUnboundMessage());
  }
  return device;
}

// A thread's current device is per thread and starts at 0, so a bridge thread on a non-zero pool
// device must switch once when it starts. Only call this once the pool is bound. The ROS buffer
// backend checks a buffer's device against the receiving thread's current device while
// deserializing, before any callback runs, so a callback-scoped device would be too late.
inline void UsePoolDeviceOnThisThread() {
  const int device = PoolDevice();
  if (const cudaError_t set = cudaSetDevice(device); set != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::runtime_error(fmt::format("Unable to make GPU {} current on this thread: {}", device,
                                         cudaGetErrorString(set)));
  }
}

// A non-blocking stream created by the calling thread on the pool device. The pool device is fixed
// before any bridge thread starts, so a cached stream stays on it. The stream is deliberately never
// destroyed: a tensor's backing can outlive the thread that made the stream (it is released from a
// transport thread, possibly after stop()), and destroying the stream at thread exit would leave
// that backing holding a dangling stream. The leak is one stream per thread.
[[nodiscard]] inline cudaStream_t ThisThreadStream() {
  thread_local cudaStream_t stream = nullptr;
  if (stream == nullptr) {
    ScopedDevice scope(PoolDevice());
    if (const cudaError_t made = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        made != cudaSuccess) {
      (void)cudaGetLastError();
      stream = nullptr;
      throw std::runtime_error(fmt::format("Unable to create a CUDA stream on GPU {}: {}",
                                           PoolDevice(), cudaGetErrorString(made)));
    }
  }
  return stream;
}

// The CUDA buffer pool is a process-wide singleton that binds to the device current at its first
// use. Makes that first use happen on `device`, verifies the binding, and records it so that
// PoolDevice() returns it. Throws if the pool is not on `device`.
inline void BindPoolDevice(int device) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count) {
    (void)cudaGetLastError();
    throw std::runtime_error(fmt::format(
        "The ROS2 bridge's pool_device is GPU {}, but this process sees {} CUDA device(s).", device,
        count));
  }
  ScopedDevice scope(device);  // leaves the caller's thread device alone
  // The first call creates the pool on the current device.
  const int bound = cuda_buffer_backend::CudaBufferImpl<std::uint8_t>::get_or_create_global_pool()
                        ->get_device_id();
  if (bound != device) {
    throw std::runtime_error(fmt::format(
        "The CUDA buffer pool is bound to GPU {}, but the ROS2 bridge needs it on GPU {}. "
        "Another component in this process used CUDA buffers first; start the bridge before it, "
        "or construct the bridge with pool_device {}.",
        bound, device, bound));
  }
  detail::PoolDeviceStorage().store(device);
}

#else  // no CUDA buffer backend: keep the host build compiling

inline void UsePoolDeviceOnThisThread() {
  // Nothing to do: a build without the CUDA buffer backend has no pool device to select.
}

#endif  // ISAAC_ROS2_BRIDGE_CUDA_BUFFER

}  // namespace isaac::os::ros2_bridge
