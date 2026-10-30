/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/memory/tensor_storage.hpp"
#include <cuda_buffer/cuda_buffer_api.hpp>
#include <cuda_runtime.h>

#include "common/ros2_bridge/cuda_device_utils.h"

namespace isaac::os::ros2_bridge::detail {

// Keeps a ROS buffer alive as the backing of a device tensor and tells the Holoscan transport when
// the buffer's GPU memory is ready and when it may be reused.
//
// - The ReadHandle is created on `stream`, so `stream` has already waited for the producer's write
//   event. For a CUDA buffer the handle points at the buffer itself; for a host buffer
//   from_input_buffer allocates a pooled CUDA buffer, copies host-to-device on `stream`, and keeps
//   that buffer alive inside the handle. `readiness_stream()` returns `stream`, which is what the
//   transport's staging copy orders after.
// - The transport reports, through set_deallocation_stream(), the stream its copy runs on. At final
//   release the backing records an event on each such stream and makes `stream` wait on it
//   (enqueue-only, so the release never blocks the host), then destroys the ReadHandle. The handle
//   records the pool's read event on `stream`, so that event orders after the transport's copy. The
//   buffer's destructor then hands the read events to the pool's recycler thread, which waits on
//   them before the block returns to the pool.
//
// The hooks run on transport threads, whose current device is not the pool device. An event must be
// on the same device as the stream it is recorded on, and the ReadHandle's destructor creates and
// records one on the current device itself, so the destructor and synchronize_for_consumer enter a
// ScopedDevice for `device`. Waiting on and synchronizing a stream do not depend on the current
// device.
class RosBufferDeviceBacking final : public holoscan::TensorMemoryReference {
 public:
  // `buffer` is a ROS buffer (CUDA or host); for a ROS message it is an aliasing pointer that also
  // keeps the message alive. `stream` belongs to `device`, which is the pool device.
  RosBufferDeviceBacking(std::shared_ptr<const rosidl::Buffer<std::uint8_t>> buffer, int device,
                         cudaStream_t stream)
      : buffer_(std::move(buffer)), device_(device), stream_(stream) {
    // Runs on the ingress thread, which already has the pool device current.
    handle_.emplace(cuda_buffer_backend::from_input_buffer(*buffer_, stream_));
  }

  ~RosBufferDeviceBacking() override {
    std::vector<cudaStream_t> streams;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      streams = deallocation_streams_;
    }
    ScopedDevice scope(device_);
    for (const cudaStream_t other : streams) {
      // The buffer must not go back to the pool before the transport's copy on `other` finishes.
      // If the event machinery fails, wait for that copy on the host instead.
      cudaEvent_t event = nullptr;
      bool ordered = cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess;
      if (ordered) {
        ordered = cudaEventRecord(event, other) == cudaSuccess &&
                  cudaStreamWaitEvent(stream_, event, 0) == cudaSuccess;
        cudaEventDestroy(event);
      }
      if (!ordered) {
        (void)cudaGetLastError();
        (void)cudaStreamSynchronize(other);
      }
    }
    handle_.reset();  // records the pool's read event on stream_, so it must stay in the scope
    buffer_.reset();  // the buffer hands its read events to the pool's recycler thread
  }

  [[nodiscard]] std::uint8_t* pointer() { return const_cast<std::uint8_t*>(handle_->get_ptr()); }

  bool set_readiness_stream(cudaStream_t) override { return true; }
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }
  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override { return stream_; }

  bool set_deallocation_stream(cudaStream_t stream) override {
    std::lock_guard<std::mutex> lock(mutex_);
    deallocation_streams_.push_back(stream);
    return true;
  }

  // Enqueue-only: the consumer stream waits for the work queued so far on `stream_`.
  holoscan::expected<void, holoscan::Error> synchronize_for_consumer(
      cudaStream_t consumer_stream) override {
    ScopedDevice scope(device_);
    cudaEvent_t event = nullptr;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "Unable to create a CUDA event for an Image tensor."});
    }
    const bool ordered = cudaEventRecord(event, stream_) == cudaSuccess &&
                         cudaStreamWaitEvent(consumer_stream, event, 0) == cudaSuccess;
    cudaEventDestroy(event);
    if (!ordered) {
      (void)cudaGetLastError();
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "Unable to order CUDA streams for an Image tensor."});
    }
    return {};
  }

  // May block, by design: waits for the work queued so far on `stream_`.
  holoscan::expected<void, holoscan::Error> synchronize_for_host() override {
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "Waiting for an Image tensor's GPU work failed."});
    }
    return {};
  }

 private:
  std::shared_ptr<const rosidl::Buffer<std::uint8_t>> buffer_;
  std::optional<cuda_buffer_backend::ReadHandle> handle_;
  int device_;
  cudaStream_t stream_;
  std::mutex mutex_;
  std::vector<cudaStream_t> deallocation_streams_;
};

}  // namespace isaac::os::ros2_bridge::detail

#endif  // ISAAC_ROS2_BRIDGE_CUDA_BUFFER
