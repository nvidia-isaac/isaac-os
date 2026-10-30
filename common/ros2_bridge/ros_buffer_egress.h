/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER

#include <cstddef>
#include <cstdint>
#include <exception>
#include <utility>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include <cuda_buffer/cuda_buffer_api.hpp>
#include <cuda_runtime.h>
#include <fmt/format.h>

#include "common/ros2_bridge/cuda_device_utils.h"
#include "common/ros2_bridge/memory_transfer.h"

namespace isaac::os::ros2_bridge {

// Puts a tensor's bytes into `destination` (a message's buffer member) in the memory the topic's
// spec asks for, following PlanEgress. Call it from a thread whose current device is the pool
// device; every thread the bridge starts is.
[[nodiscard]] inline holoscan::expected<void, holoscan::Error> TensorToRosBuffer(
    const holoscan::Tensor& tensor, const Ros2TopicSpec& spec,
    rosidl::Buffer<std::uint8_t>& destination) {
  namespace cb = cuda_buffer_backend;
  const auto bytes = static_cast<std::size_t>(tensor.nbytes());
  const DLDevice device = tensor.device();
  const bool on_gpu = device.device_type == kDLCUDA;
  const Placement source{on_gpu ? Ros2MemoryKind::kCudaDevice : Ros2MemoryKind::kHost,
                         on_gpu ? device.device_id : 0};
  // Without a readiness stream there is nothing to order a read of device memory after, and the
  // producer's work may still be running (a copy down on a host topic included), so refuse rather
  // than copy half-written pixels.
  if (on_gpu && !tensor.has_readiness_stream_contract()) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kNotSupported,
        fmt::format("An Image tensor on GPU {} has no readiness stream, so the bridge cannot "
                    "order a read of it after the producer's work.",
                    device.device_id)});
  }
  switch (PlanEgress(source, spec.memory_kind)) {
    case TransferAction::kHostCopy: {
      // No empty-tensor check on the host paths: a host topic must behave exactly as before. A
      // pinned host tensor can still be filled by work on its producer's stream, so wait for that
      // first; a pageable tensor has no such contract.
      if (tensor.has_readiness_stream_contract()) {
        if (auto synced = tensor.synchronize_for_host(); !synced) {
          return holoscan::make_unexpected(std::move(synced).error());
        }
      }
      const auto* const data = tensor.data_as<std::uint8_t>();
      destination.assign(data, data + bytes);
      return {};
    }
    case TransferAction::kDeviceToHost: {
      destination = rosidl::Buffer<std::uint8_t>(bytes);
      if (auto copied = tensor.copy_to_host(destination.data(), bytes); !copied) {
        return holoscan::make_unexpected(std::move(copied).error());
      }
      return {};
    }
    case TransferAction::kHostToDevice:
    case TransferAction::kDeviceToDevice: {
      // An empty tensor needs no check here: from_output_buffer throws on an empty buffer and the
      // catch below turns that into an error.
      if (!PoolDeviceBound()) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, detail::PoolUnboundMessage()});
      }
      try {
        const cudaStream_t stream = ThisThreadStream();
        if (tensor.has_readiness_stream_contract()) {
          // Enqueue-only: `stream` now waits for whatever the producer queued on its own stream.
          if (auto synced = tensor.synchronize_for_consumer_stream(stream); !synced) {
            return holoscan::make_unexpected(std::move(synced).error());
          }
        }
        rosidl::Buffer<std::uint8_t> buffer = cb::allocate_buffer(bytes);
        {
          auto write = cb::from_output_buffer(buffer, stream);
          // cudaMemcpyDefault copies from wherever the tensor is, including another GPU.
          if (const cudaError_t copied =
                  cudaMemcpyAsync(write.get_ptr(), tensor.data_as<std::uint8_t>(), bytes,
                                  cudaMemcpyDefault, stream);
              copied != cudaSuccess) {
            (void)cudaGetLastError();
            return holoscan::make_unexpected(
                holoscan::Error{holoscan::ErrorCode::kFailure,
                                fmt::format("Copying an Image tensor into a CUDA buffer failed: {}",
                                            cudaGetErrorString(copied))});
          }
        }  // the write handle's destructor records the write event on `stream`
        if (device.device_type != kDLCPU) {
          // The transport can release the producer's tensor as soon as this call returns, and an
          // imported reference does not know about our stream, so finish the read first. Pinned
          // host memory is copied asynchronously too, so it needs the same wait. The SDK's own
          // device copies do the same when a reference cannot track a retirement stream.
          if (const cudaError_t done = cudaStreamSynchronize(stream); done != cudaSuccess) {
            (void)cudaGetLastError();
            return holoscan::make_unexpected(
                holoscan::Error{holoscan::ErrorCode::kFailure,
                                fmt::format("Waiting for an Image tensor copy to finish failed: {}",
                                            cudaGetErrorString(done))});
          }
        }
        destination = std::move(buffer);
        return {};
      } catch (const std::exception& error) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            fmt::format("Unable to copy an Image tensor into a CUDA buffer: {}", error.what())});
      }
    }
    case TransferAction::kWrapDevice:
    case TransferAction::kWrapHost:
    case TransferAction::kReject:
      break;  // PlanEgress never returns these
  }
  return holoscan::make_unexpected(
      holoscan::Error{holoscan::ErrorCode::kFailure, "No egress path for this tensor."});
}

}  // namespace isaac::os::ros2_bridge

#endif  // ISAAC_ROS2_BRIDGE_CUDA_BUFFER
