/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/memory/tensor_storage.hpp"
#include <cuda_buffer/cuda_buffer_api.hpp>
#include <cuda_runtime.h>
#include <fmt/format.h>

#include "common/ros2_bridge/cuda_device_utils.h"
#include "common/ros2_bridge/memory_transfer.h"
#include "common/ros2_bridge/ros2_topic_spec.h"
#include "common/ros2_bridge/ros_buffer_device_backing.h"

namespace isaac::os::ros2_bridge {

namespace detail {

// A host tensor over `data`; `keep` stays alive until the tensor is released.
template <typename Keep>
[[nodiscard]] inline holoscan::expected<holoscan::Tensor, holoscan::Error> HostTensorOver(
    void* data, std::size_t bytes, Keep keep) {
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(bytes)};
  return holoscan::make_tensor(
      shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt, holoscan::MemoryStorageType::kSystem, data,
      [keep = std::move(keep)](void*) -> holoscan::expected<void, holoscan::Error> { return {}; });
}

// A host buffer on a host topic, used in place: the tensor points at the buffer and keeps `owner`
// alive.
[[nodiscard]] inline holoscan::expected<holoscan::Tensor, holoscan::Error> WrapHostBuffer(
    const std::shared_ptr<const void>& owner, const rosidl::Buffer<std::uint8_t>& buffer) {
  return HostTensorOver(const_cast<std::uint8_t*>(buffer.data()), buffer.size(), owner);
}

// A CUDA buffer on the pool device used in place, or a host buffer on a device topic. The backing's
// ReadHandle does both: for a host buffer from_input_buffer allocates a pooled CUDA buffer and
// copies host-to-device on `stream`, and the handle keeps that buffer alive.
[[nodiscard]] inline holoscan::expected<holoscan::Tensor, holoscan::Error> WrapDeviceBuffer(
    const std::shared_ptr<const void>& owner, const rosidl::Buffer<std::uint8_t>& buffer,
    int device) {
  const cudaStream_t stream = ThisThreadStream();
  // Aliasing pointer: shares the message's ownership and points at its buffer member.
  const std::shared_ptr<const rosidl::Buffer<std::uint8_t>> aliased(owner, &buffer);
  auto backing = std::make_shared<RosBufferDeviceBacking>(aliased, device, stream);
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(buffer.size())};
  return holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                               holoscan::MemoryStorageType::kDevice, backing->pointer(), backing,
                               device);
}

// What the backend leaves behind when it refuses to import a buffer: an empty CUDA buffer. Shared
// by RosBufferToTensor and the converter, whose size check would otherwise report a generic
// mismatch. A host topic can see this before the pool is bound, so the pool's GPU is named only
// when it is known.
[[nodiscard]] inline holoscan::unexpected<holoscan::Error> EmptyCudaBufferError(
    const std::string& port_name) {
  const std::string gpu =
      PoolDeviceBound() ? fmt::format("GPU {}", PoolDevice()) : std::string("the pool's GPU");
  return holoscan::make_unexpected(holoscan::Error{
      holoscan::ErrorCode::kInvalidArgument,
      fmt::format("ROS2 bridge topic '{}' received an empty CUDA buffer. The CUDA buffer backend "
                  "refuses to import a buffer that lives on a different GPU than the receiving "
                  "thread's, and hands back an empty one. The bridge receives on its CUDA buffer "
                  "pool's GPU, so publish this topic from {}.",
                  port_name, gpu)});
}

// Where a ROS buffer's data is, or why the buffer cannot be used at all.
[[nodiscard]] inline holoscan::expected<Placement, holoscan::Error> BufferPlacement(
    const rosidl::Buffer<std::uint8_t>& buffer, const std::string& port_name) {
  namespace cb = cuda_buffer_backend;
  const std::string backend = buffer.get_backend_type();
  if (backend == "cuda") {
    const auto* impl = dynamic_cast<const cb::CudaBufferImpl<std::uint8_t>*>(buffer.get_impl());
    if (impl == nullptr) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                          "A buffer reports the cuda backend but holds no CUDA buffer."});
    }
    // Any zero-size CUDA buffer counts as the backend's refusal signal, as this helper's callers
    // see it; the converter checks the expected size first.
    if (buffer.size() == 0U) {
      return EmptyCudaBufferError(port_name);
    }
    return Placement{Ros2MemoryKind::kCudaDevice, impl->get_device_id()};
  }
  if (backend == "cpu") {
    // No empty check: a host topic must behave exactly as before, and the converter's own size
    // check decides what an empty host frame is.
    return Placement{Ros2MemoryKind::kHost, 0};
  }
  return holoscan::make_unexpected(
      holoscan::Error{holoscan::ErrorCode::kNotSupported,
                      fmt::format("The buffer backend '{}' is not supported.", backend)});
}

}  // namespace detail

// Turns a ROS buffer into a tensor in the memory the topic's spec asks for, following PlanIngress.
// `buffer` must be a member of the object `owner` keeps alive (pass the message's shared_ptr and
// its buffer member). Call it from a thread whose current device is the pool device; every thread
// the bridge starts is.
[[nodiscard]] inline holoscan::expected<holoscan::Tensor, holoscan::Error> RosBufferToTensor(
    const std::shared_ptr<const void>& owner, const rosidl::Buffer<std::uint8_t>& buffer,
    const Ros2TopicSpec& spec) {
  const bool device_topic = spec.memory_kind == Ros2MemoryKind::kCudaDevice;
  if (device_topic && !PoolDeviceBound()) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, detail::PoolUnboundMessage()});
  }
  // Only a device topic compares against, or tags a tensor with, the pool's GPU; a host topic
  // never reads this and may run before the pool is bound.
  const int pool_device = device_topic ? PoolDevice() : -1;
  try {
    const auto source = detail::BufferPlacement(buffer, spec.port_name);
    if (!source) {
      return holoscan::make_unexpected(source.error());
    }
    const TransferAction action = PlanIngress(*source, spec.memory_kind, pool_device);
    if (action == TransferAction::kReject) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kNotSupported,
          fmt::format("ROS2 bridge topic '{}' received a CUDA buffer on GPU {}, but the bridge's "
                      "CUDA buffer pool is on GPU {} and the CUDA buffer backend cannot import "
                      "across GPUs. Publish this topic from GPU {}.",
                      spec.port_name, source->device, pool_device, pool_device)});
    }
    if ((action == TransferAction::kWrapDevice || action == TransferAction::kHostToDevice) &&
        !PoolDeviceBound()) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, detail::PoolUnboundMessage()});
    }
    switch (action) {
      case TransferAction::kWrapHost:
        return detail::WrapHostBuffer(owner, buffer);
      case TransferAction::kWrapDevice:
      case TransferAction::kHostToDevice:
        return detail::WrapDeviceBuffer(owner, buffer, pool_device);
      case TransferAction::kDeviceToHost: {
        // to_vector() copies a CUDA buffer down on the buffer library's own stream and waits.
        auto storage = std::make_shared<std::vector<std::uint8_t>>(buffer.to_vector());
        return detail::HostTensorOver(storage->data(), storage->size(), storage);
      }
      case TransferAction::kHostCopy:
      case TransferAction::kDeviceToDevice:
      case TransferAction::kReject:
        break;  // PlanIngress never returns these here
    }
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "No ingress path for this buffer."});
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        fmt::format("Unable to turn a ROS buffer into a tensor: {}", error.what())});
  }
}

}  // namespace isaac::os::ros2_bridge

#endif  // ISAAC_ROS2_BRIDGE_CUDA_BUFFER
