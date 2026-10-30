/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/run.hpp"
#include "holoscan/memory/tensor_storage.hpp"
#include "holoscan/time/realtime_clock.hpp"
#include <cuda_buffer/cuda_buffer_api.hpp>
#include <cuda_runtime.h>
#include <rclcpp/rclcpp.hpp>

#include "common/ros2_bridge/cuda_device_utils.h"
#include "common/ros2_bridge/ros2_bridge.h"

namespace isaac::os::ops {
namespace {

// How many frames the graph saw in device memory and in host memory.
struct FrameCounts {
  std::atomic<long> device{0};
  std::atomic<long> host{0};
};

// Device memory this test allocates for the echo's output frame. It satisfies the readiness-stream
// contract the exported topic needs and frees the memory once every consumer is done with it.
class EchoFrameMemory final : public holoscan::TensorMemoryReference {
 public:
  EchoFrameMemory(std::byte* pointer, cudaStream_t stream) : pointer_(pointer), stream_(stream) {}
  ~EchoFrameMemory() override {
    std::vector<cudaStream_t> streams;
    {
      const std::scoped_lock lock(mutex_);
      streams = streams_;
    }
    for (const cudaStream_t other : streams) {
      cudaStreamSynchronize(other);
    }
    cudaStreamSynchronize(stream_);
    cudaFree(pointer_);
  }
  bool set_readiness_stream(cudaStream_t stream) override {
    stream_ = stream;
    return true;
  }
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }
  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override { return stream_; }
  bool set_deallocation_stream(cudaStream_t stream) override {
    const std::scoped_lock lock(mutex_);
    streams_.push_back(stream);
    return true;
  }

 private:
  std::byte* pointer_;
  cudaStream_t stream_;
  std::mutex mutex_;
  std::vector<cudaStream_t> streams_;
};

// Forwards each Image and records whether its tensor arrived in device memory.
class DeviceEchoOp final : public holoscan::Operator<> {
 public:
  explicit DeviceEchoOp(std::shared_ptr<FrameCounts> counts) : counts_(std::move(counts)) {}
  ~DeviceEchoOp() override {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(camera_in, "camera_in");
    spec.output(camera_out, "camera_out");
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = camera_in.receive();
    if (!received) {
      return holoscan::make_unexpected(received.error());
    }
    const auto& tensor = received->data.data;
    if (tensor && tensor->device().device_type == kDLCUDA) {
      counts_->device.fetch_add(1);
    } else {
      counts_->host.fetch_add(1);
    }
    if (!tensor || tensor->device().device_type != kDLCUDA) {
      return camera_out.emit(received->data);
    }
    // Emit a copy in memory this operator owns instead of the imported frame itself. Forwarding an
    // imported device frame straight to an exported topic can leave the import's staging pool
    // without a free slot, after which the ingress drops every frame.
    auto copy = CopyToOwnedDeviceFrame(received->data);
    if (!copy) {
      return holoscan::make_unexpected(copy.error());
    }
    return camera_out.emit(*copy);
  }
  holoscan::Input<holoscan::schema::ImageT> camera_in;
  holoscan::Output<holoscan::schema::ImageT> camera_out;

 private:
  holoscan::expected<holoscan::schema::ImageT, holoscan::Error> CopyToOwnedDeviceFrame(
      const holoscan::schema::ImageT& image) {
    const auto failure = [](const char* what) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure, what});
    };
    if (stream_ == nullptr &&
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      return failure("echo copy: stream creation failed");
    }
    const auto bytes = static_cast<std::size_t>(image.data->nbytes());
    void* pixels = nullptr;
    if (cudaMalloc(&pixels, bytes) != cudaSuccess) {
      return failure("echo copy: cudaMalloc failed");
    }
    auto memory = std::make_shared<EchoFrameMemory>(static_cast<std::byte*>(pixels), stream_);
    if (auto synced = image.data->synchronize_for_consumer_stream(stream_); !synced) {
      return holoscan::make_unexpected(synced.error());
    }
    if (cudaMemcpyAsync(pixels, image.data->data(), bytes, cudaMemcpyDeviceToDevice, stream_) !=
        cudaSuccess) {
      return failure("echo copy: cudaMemcpyAsync failed");
    }
    const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(bytes)};
    auto tensor = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                                        holoscan::MemoryStorageType::kDevice, pixels, memory);
    if (!tensor) {
      return holoscan::make_unexpected(tensor.error());
    }
    tensor->set_readiness_stream(stream_);
    holoscan::schema::ImageT copy;
    copy.width = image.width;
    copy.height = image.height;
    copy.encoding = image.encoding;
    copy.header = image.header;
    copy.data = std::make_unique<holoscan::Tensor>(std::move(*tensor));
    return copy;
  }

  std::shared_ptr<FrameCounts> counts_;
  cudaStream_t stream_ = nullptr;
};

}  // namespace
}  // namespace isaac::os::ops

namespace isaac::os::ros2_bridge {
namespace {

namespace cb = cuda_buffer_backend;

constexpr std::uint32_t kWidth = 640U;
constexpr std::uint32_t kHeight = 480U;
constexpr std::size_t kBytes = static_cast<std::size_t>(kWidth) * kHeight * 3U;

std::vector<std::uint8_t> Pattern(std::uint32_t frame) {
  std::vector<std::uint8_t> pixels(kBytes);
  for (std::size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = static_cast<std::uint8_t>(i * 7U + frame * 13U);
  }
  return pixels;
}

long EnvLong(const char* name, long fallback) {
  const char* value = std::getenv(name);
  return value != nullptr ? std::atol(value) : fallback;
}

// ROS2_BRIDGE_PUBLISHER_MEMORY=host runs the same test with a publisher that sends host buffers
// and a topic declared that way. A second ctest entry sets it; a bridge cannot be created twice in
// one process, so the two variants are separate runs of this binary.
// "host_undeclared" sends host buffers into a topic left at the default publisher_memory, the
// mistake the bridge warns about: the frames must still arrive correctly, only slower.
std::string PublisherMemoryMode() {
  const char* value = std::getenv("ROS2_BRIDGE_PUBLISHER_MEMORY");
  return value != nullptr ? value : "cuda";
}

bool HostPublisher() { return PublisherMemoryMode() != "cuda"; }

// ROS2_BRIDGE_INTRA_PROCESS=1 gives the bridge's node and the test's node intra-process
// communication, so the CUDA buffers between them are handed over in place instead of taking the
// backend's same-process IPC path. Also a separate ctest entry, for the same one-bridge-per-process
// reason as above.
rclcpp::NodeOptions TestNodeOptions() {
  return rclcpp::NodeOptions().use_intra_process_comms(std::getenv("ROS2_BRIDGE_INTRA_PROCESS") !=
                                                       nullptr);
}

class CudaRoundtripGraph final : public holoscan::Graph {
 public:
  CudaRoundtripGraph(Ros2Bridge<>& bridge, std::shared_ptr<isaac::os::ops::FrameCounts> counts)
      : Graph("cuda_roundtrip_test"), bridge_(bridge), counts_(std::move(counts)) {}

 private:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto echo = op<isaac::os::ops::DeviceEchoOp>("echo", counts_);
    bridge_.from_ros2<sensor_msgs::msg::Image, holoscan::schema::ImageT>(
        *this, echo->camera_in,
        {.port_name = "cuda_in",
         .ros2_topic = "/cuda_roundtrip_in",
         .memory_kind = Ros2MemoryKind::kCudaDevice,
         .publisher_memory = PublisherMemoryMode() == "host" ? Ros2MemoryKind::kHost
                                                             : Ros2MemoryKind::kCudaDevice});
    bridge_.to_ros2<sensor_msgs::msg::Image, holoscan::schema::ImageT>(
        *this, echo->camera_out,
        {.port_name = "cuda_out",
         .ros2_topic = "/cuda_roundtrip_out",
         .memory_kind = Ros2MemoryKind::kCudaDevice},
        // Keep the newest frame when the exported topic is full. The default blocking edge makes
        // the echo's emit fail, and with it the whole run, if the egress is a few milliseconds late
        // draining it, which happens on a busy machine.
        {.queue_policy = holoscan::QueuePolicy::kDropOldest});
  }

  Ros2Bridge<>& bridge_;
  std::shared_ptr<isaac::os::ops::FrameCounts> counts_;
};

struct ReceivedFrame {
  std::string backend;
  bool pixels_match = false;
};

// With ROS2_BRIDGE_PUBLISHER_MEMORY=host the publisher sends host buffers into a topic declared
// publisher_memory = kHost: the graph must still see device tensors, and the frames still leave as
// CUDA buffers.
TEST(Ros2BridgeCudaRoundtrip, CudaBufferFlowsThroughTheGraphAndBackOutAsACudaBuffer) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "no CUDA device";
  }
  const long frames = EnvLong("ROS2_BRIDGE_CUDA_FRAMES", 5);

  Ros2Bridge<> bridge("cuda_roundtrip_bridge", 0, TestNodeOptions());
  auto counts = std::make_shared<isaac::os::ops::FrameCounts>();
  CudaRoundtripGraph graph(bridge, counts);
  // The egress binding names the operator whose output feeds the exported topic; the ingress
  // binding names the imported topic's endpoint (its port name) and the bridge's pool device.
  // A canary for SDK changes: without the ingress binding the SDK refuses the imported device topic
  // (the failure the README tells users to avoid). If a future SDK accepts it, or renames the
  // diagnostic, this fails and the README and the bindings need another look. The graph is composed
  // once, so the second compile reuses it and a failure of the bound compile means the bindings
  // above no longer satisfy the SDK.
  const holoscan::ExecutionPlan unbound = holoscan::compile(graph);
  EXPECT_FALSE(unbound.ok());
  EXPECT_NE(unbound.json().find("IMPLICIT_DEVICE_ZERO_REJECTED"), std::string::npos);

  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{.operator_path = "echo",
                                            .output_port = "camera_out",
                                            .device = holoscan::DeviceId{bridge.pool_device()}});
  options.deployment.bind_imported_topic_device(
      {.endpoint_name = "cuda_in", .device = holoscan::DeviceId{bridge.pool_device()}});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  bridge.start(plan.endpoint_catalog());
  auto run_session = holoscan::run_async(plan);

  EnsureRclcppInit();
  auto external_node = std::make_shared<rclcpp::Node>("cuda_roundtrip_external", TestNodeOptions());
  auto publisher = external_node->create_publisher<sensor_msgs::msg::Image>("/cuda_roundtrip_in",
                                                                            rclcpp::QoS(10));
  std::mutex mutex;
  std::vector<ReceivedFrame> received;
  rclcpp::SubscriptionOptions subscription_options;
  subscription_options.acceptable_buffer_backends = "cuda";
  auto subscription = external_node->create_subscription<sensor_msgs::msg::Image>(
      "/cuda_roundtrip_out", rclcpp::QoS(10),
      [&](const sensor_msgs::msg::Image& msg) {
        ReceivedFrame frame;
        frame.backend = msg.data.get_backend_type();
        // to_vector() copies a CUDA buffer down and a host buffer as is.
        frame.pixels_match = msg.data.to_vector() == Pattern(msg.header.stamp.nanosec);
        std::scoped_lock lock(mutex);
        received.push_back(frame);
      },
      subscription_options);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(external_node);
  std::jthread spin([&] { executor.spin(); });

  // Keep publishing fresh frames until enough come back: DDS discovery takes a moment, and early
  // frames can be dropped before the bridge's endpoints match.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30) +
                        std::chrono::milliseconds(20 * frames);
  std::uint32_t published = 0;
  const auto delivered = [&] {
    std::scoped_lock lock(mutex);
    return static_cast<long>(received.size());
  };
  while (std::chrono::steady_clock::now() < deadline && delivered() < frames) {
    sensor_msgs::msg::Image msg;
    msg.width = kWidth;
    msg.height = kHeight;
    msg.encoding = "rgb8";
    msg.step = kWidth * 3U;
    msg.header.stamp.nanosec = published;  // carries the frame number
    const auto pattern = Pattern(published);
    if (HostPublisher()) {
      msg.data = rosidl::Buffer<std::uint8_t>(kBytes);
      std::memcpy(msg.data.data(), pattern.data(), kBytes);
    } else {
      msg.data = cb::allocate_buffer(kBytes);
      cudaStream_t stream = ThisThreadStream();
      auto write = cb::from_output_buffer(msg.data, stream);
      cudaMemcpyAsync(write.get_ptr(), pattern.data(), kBytes, cudaMemcpyHostToDevice, stream);
      cudaStreamSynchronize(stream);
    }
    publisher->publish(std::move(msg));
    ++published;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  executor.cancel();
  spin.join();
  run_session.request_stop();
  run_session.wait();
  // A CUDA-buffer subscription must not be released while a publisher in this process that feeds it
  // is still alive, or the ROS context fails to delete its participant. The test's publisher feeds
  // the bridge's subscription and the bridge's publisher feeds the test's, so release the test's
  // publisher, then stop the bridge, and let the test's subscription go last.
  publisher.reset();
  bridge.stop();

  std::scoped_lock lock(mutex);
  long cuda_frames = 0;
  long mismatched = 0;
  for (const ReceivedFrame& frame : received) {
    cuda_frames += frame.backend == "cuda" ? 1 : 0;
    mismatched += frame.pixels_match ? 0 : 1;
  }
  std::cerr << "ROS2_BRIDGE_CUDA published=" << published << " received=" << received.size()
            << " cuda=" << cuda_frames << " mismatched=" << mismatched
            << " graph_device_frames=" << counts->device.load()
            << " graph_host_frames=" << counts->host.load() << "\n";

  ASSERT_GE(static_cast<long>(received.size()), frames) << "too few frames came back";
  // The point of the test: every returned frame is a CUDA buffer, not just correct pixels.
  EXPECT_EQ(cuda_frames, static_cast<long>(received.size()))
      << "a frame arrived as host memory: the CUDA backend plugin is missing or the subscription "
         "did not opt in";
  EXPECT_EQ(mismatched, 0) << "pixels changed on the way through";
  EXPECT_GT(counts->device.load(), 0);
  EXPECT_EQ(counts->host.load(), 0) << "the graph saw a host tensor on a kCudaDevice topic";
}

}  // namespace
}  // namespace isaac::os::ros2_bridge
