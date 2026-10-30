/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nppdefs.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <cuda.h>

namespace isaac::os::mcap_logger {

// A device address as the pointer type NPP's API takes. bit_cast rather than reinterpret_cast: it
// converts between the integer and the pointer of the same width, and refuses to compile if they
// ever differ, which a reinterpret_cast does not. Never dereferenced on the host; NPP resolves it
// on the device.
template <typename T>
[[nodiscard]] T* DevicePointer(CUdeviceptr address) noexcept {
  return std::bit_cast<T*>(address);
}

// Makes `context` current on the calling thread for as long as the returned scope lives, and pops
// it on every exit path, including the early returns and the exception unwind. Context stacks are
// per-thread, so the pop has to happen on the thread that pushed, which a scope guarantees. Empty
// when the push fails, and then the stack is exactly as it was found.
using ContextScope = std::unique_ptr<CUctx_st, void (*)(CUctx_st*)>;

inline void PopContext(CUctx_st*) { cuCtxPopCurrent(nullptr); }

[[nodiscard]] inline ContextScope PushContext(CUcontext context) {
  if (context == nullptr || cuCtxPushCurrent(context) != CUDA_SUCCESS) {
    return {nullptr, PopContext};
  }
  return {context, PopContext};
}

// Builds the NPP stream context for `cuda_device`, or nullopt when the driver cannot describe that
// device. NPP documents this struct as the application's job to populate ("NPP stream context
// structure must be filled in by application"), and CUDA 13's NPP_Plus ships no
// nppGetStreamContext() to do it for us -- nppcore.h exposes only nppGetLibVersion(). Left
// zero-initialized it claims device 0 and a compute capability of 0.0, which describes a different
// GPU than the conversions run on whenever the work is not on device 0.
//
// Every attribute is read before the struct exists, so there is no half-filled context to pass
// around: a caller gets a complete one or none at all.
//
// hStream stays 0: conversions run on the legacy default stream, which is what makes a following
// synchronous copy a valid sync point.
//
// Cheap enough to call per frame: the driver answers the seven queries from its own cache in about
// 120ns measured, against a frame that then spends milliseconds in conversion.
[[nodiscard]] inline std::optional<NppStreamContext> MakeNppStreamContext(
    std::uint32_t cuda_device) {
  CUdevice device = 0;
  if (cuDeviceGet(&device, static_cast<int>(cuda_device)) != CUDA_SUCCESS) {
    return std::nullopt;
  }
  const auto attribute = [device](CUdevice_attribute which, int& out) {
    return cuDeviceGetAttribute(&out, which, device) == CUDA_SUCCESS;
  };
  int multiprocessors = 0;
  int threads_per_multiprocessor = 0;
  int threads_per_block = 0;
  int shared_memory_per_block = 0;
  int compute_major = 0;
  int compute_minor = 0;
  if (!attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, multiprocessors) ||
      !attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, threads_per_multiprocessor) ||
      !attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, threads_per_block) ||
      !attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, shared_memory_per_block) ||
      !attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, compute_major) ||
      !attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, compute_minor)) {
    return std::nullopt;
  }
  return NppStreamContext{
      .hStream = nullptr,
      .nCudaDeviceId = static_cast<int>(cuda_device),
      .nMultiProcessorCount = multiprocessors,
      .nMaxThreadsPerMultiProcessor = threads_per_multiprocessor,
      .nMaxThreadsPerBlock = threads_per_block,
      .nSharedMemPerBlock = static_cast<std::size_t>(shared_memory_per_block),
      .nCudaDevAttrComputeCapabilityMajor = compute_major,
      .nCudaDevAttrComputeCapabilityMinor = compute_minor,
      .nStreamFlags = 0,
      .nReserved0 = 0,
  };
}

}  // namespace isaac::os::mcap_logger
