/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include <cuda_buffer/cuda_buffer_api.hpp>
#include <cuda_runtime.h>

#include "common/ros2_bridge/cuda_device_utils.h"
#include "common/ros2_bridge/ros_buffer_ingress.h"
#include "common/ros2_bridge/tensor_converter.h"

namespace isaac::os::ros2_bridge {
namespace {

namespace cb = cuda_buffer_backend;
using ImageConverter = TensorConverter<sensor_msgs::msg::Image, holoscan::schema::ImageT>;

std::vector<std::uint8_t> Pattern(std::size_t bytes) {
  std::vector<std::uint8_t> p(bytes);
  for (std::size_t i = 0; i < bytes; ++i) p[i] = static_cast<std::uint8_t>(i * 7U + 3U);
  return p;
}

std::shared_ptr<sensor_msgs::msg::Image> HostMessage(std::uint32_t width, std::uint32_t height) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->width = width;
  msg->height = height;
  msg->encoding = "rgb8";
  msg->step = width * 3U;
  msg->data = Pattern(static_cast<std::size_t>(width) * height * 3U);
  return msg;
}

// A message whose data is a pooled CUDA buffer holding the pattern.
std::shared_ptr<sensor_msgs::msg::Image> CudaMessage(std::uint32_t width, std::uint32_t height) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->width = width;
  msg->height = height;
  msg->encoding = "rgb8";
  msg->step = width * 3U;
  const std::size_t bytes = static_cast<std::size_t>(width) * height * 3U;
  msg->data = cb::allocate_buffer(bytes);
  cudaStream_t stream = ThisThreadStream();
  const auto pattern = Pattern(bytes);
  {
    auto write = cb::from_output_buffer(msg->data, stream);
    EXPECT_EQ(
        cudaMemcpyAsync(write.get_ptr(), pattern.data(), bytes, cudaMemcpyHostToDevice, stream),
        cudaSuccess);
  }
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  return msg;
}

std::vector<std::uint8_t> ReadTensor(const holoscan::Tensor& tensor) {
  std::vector<std::uint8_t> host(static_cast<std::size_t>(tensor.nbytes()));
  EXPECT_TRUE(tensor.copy_to_host(host.data(), host.size()).has_value());
  return host;
}

Ros2TopicSpec DeviceSpec() {
  Ros2TopicSpec spec{.port_name = "cam"};
  spec.memory_kind = Ros2MemoryKind::kCudaDevice;
  return spec;
}

class RosBufferIngress : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "no CUDA device";
    }
    detail::PoolDeviceStorage().store(
        0);  // what the bridge's start() records once it binds the pool
  }
  void TearDown() override { detail::PoolDeviceStorage().store(-1); }
};

TEST_F(RosBufferIngress, CudaMessageBecomesADeviceTensorOverItsOwnBuffer) {
  auto msg = CudaMessage(4, 4);
  const auto* impl = dynamic_cast<const cb::CudaBufferImpl<std::uint8_t>*>(msg->data.get_impl());
  ASSERT_NE(impl, nullptr);
  const std::uint8_t* const message_pointer = impl->get_cuda_buffer().get_device_ptr();

  auto built = ImageConverter::to_tensors(msg, DeviceSpec());
  ASSERT_TRUE(built.has_value());
  const holoscan::Tensor& tensor = built->second.at(0);
  EXPECT_EQ(tensor.device().device_type, kDLCUDA);
  EXPECT_EQ(tensor.data_as<std::uint8_t>(), message_pointer) << "the bridge must not copy";
  EXPECT_TRUE(tensor.has_readiness_stream_contract());
  EXPECT_EQ(ReadTensor(tensor), Pattern(48));
}

TEST_F(RosBufferIngress, HostMessageOnADeviceTopicIsCopiedToTheDevice) {
  auto msg = HostMessage(4, 4);
  auto built = ImageConverter::to_tensors(msg, DeviceSpec());
  ASSERT_TRUE(built.has_value());
  const holoscan::Tensor& tensor = built->second.at(0);
  EXPECT_EQ(tensor.device().device_type, kDLCUDA);
  EXPECT_EQ(ReadTensor(tensor), Pattern(48));
}

TEST_F(RosBufferIngress, CudaMessageOnAHostTopicIsCopiedDownToTheHost) {
  auto msg = CudaMessage(4, 4);
  const auto built = ImageConverter::to_tensors(msg, Ros2TopicSpec{});
  ASSERT_TRUE(built.has_value());
  const holoscan::Tensor& tensor = built->second.at(0);
  EXPECT_EQ(tensor.device().device_type, kDLCPU);
  EXPECT_EQ(ReadTensor(tensor), Pattern(48));
}

TEST_F(RosBufferIngress, ADeviceTopicIsRefusedWhileThePoolIsUnbound) {
  detail::PoolDeviceStorage().store(-1);
  auto msg = HostMessage(4, 4);
  const auto tensor = RosBufferToTensor(msg, msg->data, DeviceSpec());
  ASSERT_FALSE(tensor.has_value());
  EXPECT_NE(tensor.error().message.find("not bound"), std::string::npos);
}

// A CUDA buffer that still reaches a host topic is copied down without the pool too.
TEST_F(RosBufferIngress, ACudaMessageOnAHostTopicNeedsNoPool) {
  detail::PoolDeviceStorage().store(-1);
  auto msg = CudaMessage(4, 4);
  const auto tensor = RosBufferToTensor(msg, msg->data, Ros2TopicSpec{.port_name = "cam"});
  ASSERT_TRUE(tensor.has_value());
  EXPECT_EQ(tensor->device().device_type, kDLCPU);
  EXPECT_EQ(ReadTensor(*tensor), Pattern(48));
}

// A host topic never needs the pool, and wraps the message's own memory without a copy.
TEST_F(RosBufferIngress, AHostTopicNeedsNoPool) {
  detail::PoolDeviceStorage().store(-1);
  auto msg = HostMessage(4, 4);
  auto built = ImageConverter::to_tensors(msg, Ros2TopicSpec{});
  ASSERT_TRUE(built.has_value());
  const holoscan::Tensor& tensor = built->second.at(0);
  EXPECT_EQ(tensor.device().device_type, kDLCPU);
  EXPECT_EQ(tensor.data_as<std::uint8_t>(), msg->data.data());
}

// What the backend hands back when it refuses to import a buffer from another GPU. Through the
// converter the cause must survive the size check that runs before the helper.
TEST_F(RosBufferIngress, EmptyCudaMessageThroughToTensorsNamesTheLikelyCause) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->width = 4;
  msg->height = 4;
  msg->encoding = "rgb8";
  msg->step = 12;
  msg->data = rosidl::Buffer<std::uint8_t>(std::make_unique<cb::CudaBufferImpl<std::uint8_t>>());
  const auto built = ImageConverter::to_tensors(msg, DeviceSpec());
  ASSERT_FALSE(built.has_value());
  const std::string what = built.error().message;
  EXPECT_NE(what.find("cam"), std::string::npos);
  EXPECT_NE(what.find("empty CUDA buffer"), std::string::npos);
  EXPECT_EQ(what.find("data size does not match"), std::string::npos);
}

TEST_F(RosBufferIngress, EmptyCudaBufferIsRejectedAndNamesTheLikelyCause) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->data = rosidl::Buffer<std::uint8_t>(std::make_unique<cb::CudaBufferImpl<std::uint8_t>>());
  ASSERT_EQ(msg->data.get_backend_type(), "cuda");
  ASSERT_EQ(msg->data.size(), 0U);
  const auto tensor = RosBufferToTensor(msg, msg->data, DeviceSpec());
  ASSERT_FALSE(tensor.has_value());
  const std::string what = tensor.error().message;
  EXPECT_NE(what.find("cam"), std::string::npos);
  EXPECT_NE(what.find("empty CUDA buffer"), std::string::npos);
  EXPECT_NE(what.find("GPU 0"), std::string::npos);
}

// The release design: the pooled buffer must not be handed out again until the transport's
// deallocation stream has finished. A host function holds that stream until the test opens a gate,
// so "not reusable yet" is deterministic. A byte size no other test uses keeps this test's pool
// bucket private.
struct Gate {
  std::mutex mutex;
  std::condition_variable cv;
  bool open = false;
};
// Opens the gate on every exit path, so a failed assertion cannot leave the stream blocked. When
// the guard has to open it, it also waits for the stream, because the host callback still uses the
// gate and must be done before the gate is destroyed. A test that opened the gate itself waits for
// the stream itself and destroys it before the guard runs.
class GateOpener {
 public:
  GateOpener(Gate& gate, cudaStream_t stream) : gate_(gate), stream_(stream) {}
  GateOpener(const GateOpener&) = delete;
  GateOpener& operator=(const GateOpener&) = delete;
  ~GateOpener() {
    if (opened_) return;
    Open();
    cudaStreamSynchronize(stream_);
  }
  void Open() {
    if (opened_) return;
    opened_ = true;
    {
      std::scoped_lock lock(gate_.mutex);
      gate_.open = true;
    }
    gate_.cv.notify_all();
  }

 private:
  Gate& gate_;
  cudaStream_t stream_;
  bool opened_ = false;
};
void CUDART_CB BlockUntilOpen(void* user) {
  auto* gate = static_cast<Gate*>(user);
  std::unique_lock<std::mutex> lock(gate->mutex);
  gate->cv.wait(lock, [&] { return gate->open; });
}

TEST_F(RosBufferIngress, BufferIsNotReusedUntilTheDeallocationStreamFinishes) {
  constexpr std::uint32_t kWidth = 1013;  // 1013 * 997 * 3 bytes: a size no other test uses
  constexpr std::uint32_t kHeight = 997;
  constexpr std::size_t kBytes = static_cast<std::size_t>(kWidth) * kHeight * 3U;

  auto msg = CudaMessage(kWidth, kHeight);
  const auto* impl = dynamic_cast<const cb::CudaBufferImpl<std::uint8_t>*>(msg->data.get_impl());
  ASSERT_NE(impl, nullptr);
  const std::uint8_t* const original = impl->get_cuda_buffer().get_device_ptr();

  cudaStream_t gated = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&gated, cudaStreamNonBlocking), cudaSuccess);
  Gate gate;
  GateOpener opener(gate, gated);
  ASSERT_EQ(cudaLaunchHostFunc(gated, BlockUntilOpen, &gate), cudaSuccess);

  {
    auto built = ImageConverter::to_tensors(msg, DeviceSpec());
    ASSERT_TRUE(built.has_value());
    // What the transport does after enqueuing its copy.
    ASSERT_TRUE(built->second.at(0).set_deallocation_stream(gated));
    msg.reset();  // the tensor's backing holds the only remaining reference
  }  // tensor destroyed: the backing is released with the gate still closed

  // Closed gate: every buffer the pool hands out must be a different block.
  std::vector<rosidl::Buffer<std::uint8_t>> held;
  for (int i = 0; i < 4; ++i) {
    held.push_back(cb::allocate_buffer(kBytes));
    const auto* held_impl =
        dynamic_cast<const cb::CudaBufferImpl<std::uint8_t>*>(held.back().get_impl());
    ASSERT_NE(held_impl, nullptr);
    EXPECT_NE(held_impl->get_cuda_buffer().get_device_ptr(), original)
        << "the block was reused while the deallocation stream was still busy";
  }
  held.clear();

  opener.Open();
  ASSERT_EQ(cudaStreamSynchronize(gated), cudaSuccess);

  // Open gate: the original block becomes reusable once the pool's recycler has seen its events.
  bool reused = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!reused && std::chrono::steady_clock::now() < deadline) {
    std::vector<rosidl::Buffer<std::uint8_t>> batch;
    for (int i = 0; i < 8 && !reused; ++i) {
      batch.push_back(cb::allocate_buffer(kBytes));
      const auto* batch_impl =
          dynamic_cast<const cb::CudaBufferImpl<std::uint8_t>*>(batch.back().get_impl());
      reused = batch_impl != nullptr && batch_impl->get_cuda_buffer().get_device_ptr() == original;
    }
    if (!reused) std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  EXPECT_TRUE(reused) << "the block never became reusable after the stream finished";
  cudaStreamDestroy(gated);
}

}  // namespace
}  // namespace isaac::os::ros2_bridge
