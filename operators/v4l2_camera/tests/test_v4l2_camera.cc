/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "operators/v4l2_camera/v4l2_camera_op.h"

namespace isaac::os::ops {
namespace {

using namespace std::chrono_literals;

constexpr std::int64_t kWidth = 640;
constexpr std::int64_t kHeight = 480;
constexpr std::int64_t kFrameRate = 30;
constexpr std::size_t kExpectedFrames = 15U;

const holoscan::TensorInputSpec kDeviceImage{
    .representation =
        {
            .memory_kind = holoscan::MemoryKind::kCudaDevice,
            .dtype = DLDataType{kDLUInt, 8U, 1U},
            .rank = 3U,
        },
};

std::string camera_device() {
  const char* configured = std::getenv("ISAAC_V4L2_TEST_DEVICE");
  return configured != nullptr ? std::string{configured} : std::string{"/dev/video4"};
}
bool camera_offers_yuyv(const std::string& device) {
  const int descriptor = open(device.c_str(), O_RDWR);
  if (descriptor < 0) {
    return false;
  }
  bool found = false;
  v4l2_fmtdesc format{};
  format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  while (!found && ioctl(descriptor, VIDIOC_ENUM_FMT, &format) == 0) {
    found = format.pixelformat == V4L2_PIX_FMT_YUYV;
    ++format.index;
  }
  close(descriptor);
  return found;
}

struct CapturedFrame {
  std::vector<std::int64_t> shape;
  std::int64_t capture_time_ns{};
};

struct FrameCapture {
  static void reset() {
    const std::lock_guard lock(mutex);
    frames.clear();
  }

  static void record(CapturedFrame frame) {
    {
      const std::lock_guard lock(mutex);
      frames.push_back(std::move(frame));
    }
    ready.notify_all();
  }

  static std::vector<CapturedFrame> wait_for(std::size_t count) {
    std::unique_lock lock(mutex);
    ready.wait_for(lock, 10s, [count] { return frames.size() >= count; });
    return frames;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<CapturedFrame> frames;
};

class FrameSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(image, "in_image").queue_depth(4U).expects_tensor(kDeviceImage);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{image});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = image.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    CapturedFrame frame;
    frame.shape = received->data.shape();
    const holoscan::TimePoint capture_time = received->metadata.capture_time();
    frame.capture_time_ns = capture_time.valid() ? capture_time.timestamp_ns : 0;
    FrameCapture::record(std::move(frame));
    return {};
  }

  holoscan::Input<holoscan::Tensor> image;
};

TEST(V4l2Camera, PublishesColourFramesFromTheCamera) {
  const std::string device = camera_device();
  FrameCapture::reset();

  holoscan::Graph graph{"v4l2-camera-test"};
  auto camera = graph.op<V4l2CameraOp>("camera");
  auto sink = graph.op<FrameSink>("frame_sink");
  camera.set_params(V4l2CameraParams{
      .device = device,
      .width = kWidth,
      .height = kHeight,
      .frame_rate = kFrameRate,
      .buffer_count = 4,
  });
  graph.add_flow(camera->image, sink->image);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions compile_options;
  compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "camera",
      .output_port = "out_image",
      .device = holoscan::DeviceId{0},
  });
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(compile_options));
  ASSERT_TRUE(plan.ok()) << plan.json();

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto frames = FrameCapture::wait_for(kExpectedFrames);
  session.request_stop();
  session.wait();

  ASSERT_GE(frames.size(), kExpectedFrames)
      << "captured " << frames.size() << " frames from " << device;
  for (const CapturedFrame& frame : frames) {
    ASSERT_EQ(frame.shape.size(), 3U);
    EXPECT_EQ(frame.shape[0], kHeight);
    EXPECT_EQ(frame.shape[1], kWidth);
    EXPECT_EQ(frame.shape[2], 3) << "the operator publishes red, green, and blue";
    EXPECT_GT(frame.capture_time_ns, 0) << "every frame carries the instant it was taken";
  }
  for (std::size_t index = 1; index < frames.size(); ++index) {
    EXPECT_GT(frames[index].capture_time_ns, frames[index - 1].capture_time_ns)
        << "capture times advance with the frame order";
  }
  const std::int64_t span_ns = frames.back().capture_time_ns - frames.front().capture_time_ns;
  const std::int64_t period_ns = 1000000000LL / kFrameRate;
  EXPECT_GE(span_ns, period_ns * static_cast<std::int64_t>(frames.size() - 2U));
}

}  // namespace
}  // namespace isaac::os::ops

int main(int argc, char** argv) {
  const std::string device = isaac::os::ops::camera_device();
  if (!isaac::os::ops::camera_offers_yuyv(device)) {
    std::fprintf(stderr,
                 "No camera at %s offering YUYV. Set ISAAC_V4L2_TEST_DEVICE to select another "
                 "node.\n",
                 device.c_str());
    return ISAAC_CTEST_SKIP_RETURN_CODE;
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
