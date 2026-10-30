/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "holoscan/memory/tensor_storage.hpp"
#include <cuda_buffer/cuda_buffer_api.hpp>
#include <cuda_runtime.h>
#include <flatbuffers/flatbuffers.h>

#include "common/ros2_bridge/cuda_device_utils.h"
#include "common/ros2_bridge/ros_buffer_egress.h"
#include "common/ros2_bridge/tensor_converter.h"

namespace isaac::os::ros2_bridge {
namespace {

using ImageConverter = TensorConverter<sensor_msgs::msg::Image, holoscan::schema::ImageT>;

constexpr std::size_t kBytes = 12;

std::vector<std::uint8_t> Pattern() { return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; }

Ros2TopicSpec DeviceSpec() {
  Ros2TopicSpec spec;
  spec.memory_kind = Ros2MemoryKind::kCudaDevice;
  return spec;
}

// A device tensor over memory this test owns, reporting a readiness stream the way a producer's
// reference does. Its readiness stream has finished by the time the test builds it.
class DeviceReference final : public holoscan::TensorMemoryReference {
 public:
  DeviceReference(std::uint8_t* pointer, cudaStream_t stream, int device, bool has_readiness_stream)
      : pointer_(pointer),
        stream_(stream),
        device_(device),
        has_readiness_stream_(has_readiness_stream) {}
  ~DeviceReference() override {
    ScopedDevice scope(device_);
    cudaStreamSynchronize(stream_);
    cudaFree(pointer_);
    cudaStreamDestroy(stream_);
  }
  bool set_readiness_stream(cudaStream_t) override { return true; }
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override {
    return has_readiness_stream_;
  }
  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override { return stream_; }
  bool set_deallocation_stream(cudaStream_t) override { return true; }
  // Enqueue-only: the consumer stream waits for the readiness stream's current tail.
  holoscan::expected<void, holoscan::Error> synchronize_for_consumer(
      cudaStream_t consumer) override {
    ScopedDevice scope(device_);
    cudaEvent_t event = nullptr;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    cudaEventRecord(event, stream_);
    cudaStreamWaitEvent(consumer, event, 0);
    cudaEventDestroy(event);
    return {};
  }

 private:
  std::uint8_t* pointer_;
  cudaStream_t stream_;
  int device_;
  bool has_readiness_stream_;
};

holoscan::Tensor MakeDeviceTensor(int device = 0, bool has_readiness_stream = true) {
  ScopedDevice scope(device);
  std::uint8_t* memory = nullptr;
  EXPECT_EQ(cudaMalloc(reinterpret_cast<void**>(&memory), kBytes), cudaSuccess);
  const auto pattern = Pattern();
  EXPECT_EQ(cudaMemcpy(memory, pattern.data(), kBytes, cudaMemcpyHostToDevice), cudaSuccess);
  cudaStream_t stream = nullptr;
  EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  auto reference = std::make_shared<DeviceReference>(memory, stream, device, has_readiness_stream);
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(kBytes)};
  auto tensor =
      holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                            holoscan::MemoryStorageType::kDevice, memory, reference, device);
  EXPECT_TRUE(tensor.has_value());
  return std::move(*tensor);
}

holoscan::Tensor MakeHostTensor() {
  auto storage = std::make_shared<std::vector<std::uint8_t>>(Pattern());
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(kBytes)};
  auto tensor = holoscan::make_tensor(
      shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt, holoscan::MemoryStorageType::kSystem,
      storage->data(),
      [storage](void*) -> holoscan::expected<void, holoscan::Error> { return {}; });
  EXPECT_TRUE(tensor.has_value());
  return std::move(*tensor);
}

// A pinned host tensor (cudaMallocHost, device type kDLCUDAHost) over the test pattern.
holoscan::Tensor MakePinnedHostTensor() {
  std::uint8_t* memory = nullptr;
  EXPECT_EQ(cudaMallocHost(reinterpret_cast<void**>(&memory), kBytes), cudaSuccess);
  const auto pattern = Pattern();
  std::ranges::copy(pattern, memory);
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(kBytes)};
  auto tensor = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                                      holoscan::MemoryStorageType::kHost, memory,
                                      [memory](void*) -> holoscan::expected<void, holoscan::Error> {
                                        cudaFreeHost(memory);
                                        return {};
                                      });
  EXPECT_TRUE(tensor.has_value());
  return std::move(*tensor);
}

// A pinned host tensor that reports a readiness stream, counting how often it is asked to
// synchronize for the host.
class PinnedReference final : public holoscan::TensorMemoryReference {
 public:
  explicit PinnedReference(std::uint8_t* pointer) : pointer_(pointer) {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  }
  ~PinnedReference() override {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    cudaFreeHost(pointer_);
  }
  bool set_readiness_stream(cudaStream_t) override { return true; }
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }
  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override { return stream_; }
  bool set_deallocation_stream(cudaStream_t) override { return true; }
  holoscan::expected<void, holoscan::Error> synchronize_for_host() override {
    host_syncs.fetch_add(1);
    return {};
  }
  std::atomic<int> host_syncs{0};

 private:
  std::uint8_t* pointer_;
  cudaStream_t stream_ = nullptr;
};

// How many CUDA devices this machine has; 0 when there is no usable driver.
int CudaDeviceCount() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}

class RosBufferEgress : public ::testing::Test {
 protected:
  void SetUp() override {
    if (CudaDeviceCount() == 0) {
      GTEST_SKIP() << "no CUDA device";
    }
    detail::PoolDeviceStorage().store(
        0);  // what the bridge's start() records once it binds the pool
  }
  void TearDown() override { detail::PoolDeviceStorage().store(-1); }
};

TEST_F(RosBufferEgress, DeviceTensorBecomesAPooledCudaBuffer) {
  const holoscan::Tensor tensor = MakeDeviceTensor();
  rosidl::Buffer<std::uint8_t> out;
  ASSERT_TRUE(TensorToRosBuffer(tensor, DeviceSpec(), out).has_value());
  EXPECT_EQ(out.get_backend_type(), "cuda");
  EXPECT_EQ(out.to_vector(), Pattern());  // to_vector() copies a CUDA buffer down
}

TEST_F(RosBufferEgress, HostTensorBecomesAPooledCudaBufferOnADeviceTopic) {
  const holoscan::Tensor tensor = MakeHostTensor();
  rosidl::Buffer<std::uint8_t> out;
  ASSERT_TRUE(TensorToRosBuffer(tensor, DeviceSpec(), out).has_value());
  EXPECT_EQ(out.get_backend_type(), "cuda");
  EXPECT_EQ(out.to_vector(), Pattern());
}

TEST_F(RosBufferEgress, PinnedHostTensorBecomesAPooledCudaBufferOnADeviceTopic) {
  const holoscan::Tensor tensor = MakePinnedHostTensor();
  ASSERT_EQ(tensor.device().device_type, kDLCUDAHost);
  rosidl::Buffer<std::uint8_t> out;
  ASSERT_TRUE(TensorToRosBuffer(tensor, DeviceSpec(), out).has_value());
  EXPECT_EQ(out.get_backend_type(), "cuda");
  EXPECT_EQ(out.to_vector(), Pattern());
}

// A pinned host tensor with a readiness stream may still be filled by its producer, so a host
// topic waits for it before copying.
TEST_F(RosBufferEgress, PinnedHostTensorWithAReadinessStreamIsWaitedForOnAHostTopic) {
  std::uint8_t* memory = nullptr;
  ASSERT_EQ(cudaMallocHost(reinterpret_cast<void**>(&memory), kBytes), cudaSuccess);
  const auto pattern = Pattern();
  std::ranges::copy(pattern, memory);
  auto reference = std::make_shared<PinnedReference>(memory);
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(kBytes)};
  auto tensor = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                                      holoscan::MemoryStorageType::kHost, memory, reference);
  ASSERT_TRUE(tensor.has_value());
  rosidl::Buffer<std::uint8_t> out;
  ASSERT_TRUE(TensorToRosBuffer(*tensor, Ros2TopicSpec{}, out).has_value());
  EXPECT_EQ(reference->host_syncs.load(), 1);
  EXPECT_EQ(out.to_vector(), Pattern());
}

TEST_F(RosBufferEgress, DeviceTensorOnAHostTopicIsCopiedDown) {
  const holoscan::Tensor tensor = MakeDeviceTensor();
  rosidl::Buffer<std::uint8_t> out;
  ASSERT_TRUE(TensorToRosBuffer(tensor, Ros2TopicSpec{}, out).has_value());
  EXPECT_EQ(out.get_backend_type(), "cpu");
  EXPECT_EQ(out.to_vector(), Pattern());
}

// A device tensor that reports no readiness stream is refused on both a device and a host topic.
TEST_F(RosBufferEgress, ADeviceTensorWithoutAReadinessStreamIsRefused) {
  const holoscan::Tensor tensor = MakeDeviceTensor(0, /*has_readiness_stream=*/false);
  rosidl::Buffer<std::uint8_t> out;
  const auto status = TensorToRosBuffer(tensor, DeviceSpec(), out);
  ASSERT_FALSE(status.has_value());
  EXPECT_NE(status.error().message.find("readiness stream"), std::string::npos);
  // The same on a host topic: the copy down would read unordered memory too.
  EXPECT_FALSE(TensorToRosBuffer(tensor, Ros2TopicSpec{}, out).has_value());
}

TEST_F(RosBufferEgress, ADeviceDestinationIsRefusedWhileThePoolIsUnbound) {
  detail::PoolDeviceStorage().store(-1);
  const holoscan::Tensor tensor = MakeHostTensor();
  rosidl::Buffer<std::uint8_t> out;
  const auto status = TensorToRosBuffer(tensor, DeviceSpec(), out);
  ASSERT_FALSE(status.has_value());
  EXPECT_NE(status.error().message.find("not bound"), std::string::npos);
}

TEST_F(RosBufferEgress, AHostDestinationNeedsNoPool) {
  detail::PoolDeviceStorage().store(-1);
  const holoscan::Tensor tensor = MakeHostTensor();
  rosidl::Buffer<std::uint8_t> out;
  ASSERT_TRUE(TensorToRosBuffer(tensor, Ros2TopicSpec{}, out).has_value());
  EXPECT_EQ(out.to_vector(), Pattern());
}

// The converter-level behaviour: spec.memory_kind chooses the message's buffer type.
TEST_F(RosBufferEgress, FromTensorsForRootPublishesACudaBufferOnADeviceTopic) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->height = 2;
  msg->width = 2;
  msg->encoding = "rgb8";
  msg->step = 6;
  msg->data = Pattern();
  const auto built = ImageConverter::to_tensors(msg, Ros2TopicSpec{});
  ASSERT_TRUE(built.has_value());
  const auto* root = flatbuffers::GetRoot<holoscan::schema::Image>(built->first.data());

  std::vector<holoscan::Tensor> device_tensors;
  device_tensors.push_back(MakeDeviceTensor());
  const auto out = ImageConverter::from_tensors_for_root(root, device_tensors, DeviceSpec());
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->data.get_backend_type(), "cuda");
  EXPECT_EQ(out->data.to_vector(), Pattern());
}

// Two-GPU test: skipped on a single-GPU host. The fixture records the pool on device 0, so a tensor
// on GPU 1 takes the copy across GPUs.
TEST_F(RosBufferEgress, TensorOnAnotherGpuIsCopiedIntoThePoolBuffer) {
  if (CudaDeviceCount() < 2) GTEST_SKIP() << "needs two CUDA devices";
  const holoscan::Tensor tensor = MakeDeviceTensor(1);
  rosidl::Buffer<std::uint8_t> out;
  ASSERT_TRUE(TensorToRosBuffer(tensor, DeviceSpec(), out).has_value());
  EXPECT_EQ(out.get_backend_type(), "cuda");
  EXPECT_EQ(out.to_vector(), Pattern());
}

}  // namespace
}  // namespace isaac::os::ros2_bridge
