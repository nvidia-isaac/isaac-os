/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// The GPU image path of the reference app. It only exists in a build whose ros2_bridge has the
// CUDA buffer backend (ISAAC_ROS2_BRIDGE_CUDA_BUFFER, defined by the ros2_bridge CMake target).
#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <cuda_runtime.h>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/memory/tensor_storage.hpp>

#include "common/image_schema/image_schema.h"

namespace isaac::os::ros2_bridge_reference {

inline constexpr std::int32_t kImageWidth = 640;
inline constexpr std::int32_t kImageHeight = 480;
inline constexpr std::size_t kImageBytes =
    static_cast<std::size_t>(kImageWidth) * kImageHeight * 3U;

// The byte every pixel of frame `frame` holds, so a receiver can check the pixels as well as the
// size.
[[nodiscard]] inline std::uint8_t ImagePatternByte(std::int64_t frame) {
  return static_cast<std::uint8_t>(frame % 251 + 1);
}

// How many images the generator has emitted, readable by main.cc after the graph stops.
[[nodiscard]] inline std::atomic<std::uint64_t>& image_sent_count() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}

// A CUDA stream that lives until the operator and every frame it emitted have let go of it: a frame
// can be released by the transport after the operator is gone, and its cleanup waits on the stream.
[[nodiscard]] inline std::shared_ptr<CUstream_st> MakeSharedStream() {
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    return nullptr;
  }
  return std::shared_ptr<CUstream_st>(stream,
                                      [](cudaStream_t doomed) { cudaStreamDestroy(doomed); });
}

// One frame's device memory, freed once the producer's stream and every stream the transport
// reports have finished with it. It reports the producer's stream as the tensor's readiness stream,
// which the transport requires of every CUDA tensor it stages.
class DeviceFrameMemory final : public holoscan::TensorMemoryReference {
 public:
  DeviceFrameMemory(void* pointer, std::shared_ptr<CUstream_st> stream)
      : pointer_(pointer), owner_(std::move(stream)), stream_(owner_.get()) {}
  ~DeviceFrameMemory() override {
    std::vector<cudaStream_t> streams;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      streams = deallocation_streams_;
    }
    for (const cudaStream_t other : streams) {
      cudaStreamSynchronize(other);
    }
    cudaStreamSynchronize(stream_);
    cudaFree(pointer_);
  }
  DeviceFrameMemory(const DeviceFrameMemory&) = delete;
  DeviceFrameMemory& operator=(const DeviceFrameMemory&) = delete;

  bool set_readiness_stream(cudaStream_t stream) override {
    stream_ = stream;
    return true;
  }
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }
  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override { return stream_; }
  bool set_deallocation_stream(cudaStream_t stream) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    deallocation_streams_.push_back(stream);
    return true;
  }

 private:
  void* pointer_;
  std::shared_ptr<CUstream_st>
      owner_;  // keeps the producer's stream alive until this frame is freed
  cudaStream_t stream_;
  std::mutex mutex_;
  std::vector<cudaStream_t> deallocation_streams_;
};

// Emits a 640x480 rgb8 isaac Image every tick whose pixels live in GPU memory, filled by the GPU
// itself. The frame number rides in the header's capture timestamp, and every byte of the frame is
// ImagePatternByte(frame). The graph publishes it as a CUDA buffer, so the frame reaches ROS
// without a host copy.
class ImageGeneratorOp final : public holoscan::Operator<> {
 public:
  static constexpr auto kPeriod = std::chrono::milliseconds(50);  // 20 Hz.

  // `device` is the GPU the bridge's CUDA buffer pool is bound to (the bridge's pool_device).
  explicit ImageGeneratorOp(int device) : device_(device) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(out_image, "out_image"); }

  void stop() override {
    HOLOSCAN_LOG_INFO("ImageGeneratorOp sent {} images.", image_sent_count().load());
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kPeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    if (!stream_) {
      if (cudaSetDevice(device_) != cudaSuccess || !(stream_ = MakeSharedStream())) {
        return failure("Unable to create the image generator's CUDA stream.");
      }
    }
    void* pixels = nullptr;
    if (cudaMalloc(&pixels, kImageBytes) != cudaSuccess) {
      return failure("Unable to allocate the image's device memory.");
    }
    // The memory owns itself from here on, so an early return below still frees it.
    auto memory = std::make_shared<DeviceFrameMemory>(pixels, stream_);
    if (cudaMemsetAsync(pixels, ImagePatternByte(frame_), kImageBytes, stream_.get()) !=
        cudaSuccess) {
      return failure("Unable to fill the image on the GPU.");
    }

    const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(kImageBytes)};
    auto tensor = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                                        holoscan::MemoryStorageType::kDevice, pixels, memory);
    if (!tensor) {
      return holoscan::make_unexpected(tensor.error());
    }
    tensor->set_readiness_stream(stream_.get());

    holoscan::schema::ImageT image;
    image.width = kImageWidth;
    image.height = kImageHeight;
    image.encoding = holoscan::schema::ImageEncoding_RGB8;
    image.header = std::make_shared<holoscan::schema::HeaderT>();
    image.header->frame_id = "camera";
    image.header->capture_timestamp_ns = frame_;
    image.data = std::make_unique<holoscan::Tensor>(std::move(*tensor));
    ++frame_;
    auto result = out_image.emit(image);
    if (result) image_sent_count().fetch_add(1);
    return result;
  }

  holoscan::Output<holoscan::schema::ImageT> out_image;

 private:
  static holoscan::expected<void, holoscan::Error> failure(const char* message) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure, message});
  }

  int device_;
  std::shared_ptr<CUstream_st> stream_;
  std::int64_t frame_ = 0;
};

}  // namespace isaac::os::ros2_bridge_reference

#endif  // ISAAC_ROS2_BRIDGE_CUDA_BUFFER
