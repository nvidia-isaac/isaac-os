/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/run.hpp"
#include "holoscan/core/temporal_contract.hpp"
#include "holoscan/sensor_io/image_metadata.hpp"
#include "holoscan/time/realtime_clock.hpp"
#include <cuda.h>

#include "common/image_schema/image_schema.h"
#include "common/mcap_logger/frame_encoder.h"
#include "common/mcap_logger/mcap_logger.h"
#include "common/mcap_logger/tests/scoped_temp_directory.h"
#include "common/mcap_replayer/mcap_replayer.h"
#include "common/mcap_replayer/tests/replay_test_harness.h"
#include "common/nv_video_decoder/nv_video_decoder.hpp"

namespace isaac::os::mcap_replayer {
namespace {

using namespace std::chrono_literals;
using isaac::os::mcap_logger::FrameEncoder;
using isaac::os::mcap_logger::VideoCodec;
using test_support::ImageSinkOp;
using test_support::ReceivedImage;
using test_support::ReceivedLog;
using test_support::ReplayerTestGraph;

// 640x480, comfortably above the encoder's minimum frame size, as the logger's own video test uses.
constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 480;
// A solid colour survives lossy compression to within a few levels, which a pattern would not.
constexpr std::array<std::uint8_t, 3> kColour = {200, 60, 30};
constexpr int kTolerance = 6;
constexpr const char* kRgbTopic = "video_rgb";
constexpr const char* kRgbaTopic = "video_rgba";
constexpr const char* kH265Topic = "video_h265";

// The macro form is required: GTEST_SKIP() returns from the enclosing function.
#define SKIP_UNLESS_VIDEO_HARDWARE(codec_name, codec)                           \
  do {                                                                          \
    if (cuInit(0) != CUDA_SUCCESS) GTEST_SKIP() << "CUDA driver not available"; \
    if (!FrameEncoder::hardware_available(0, codec)) {                          \
      GTEST_SKIP() << "No " codec_name " encoder on GPU 0";                     \
    }                                                                           \
    if (!nv_video_decoder::NVVideoDecoder::is_available(0, codec_name)) {       \
      GTEST_SKIP() << "No " codec_name " decoder on GPU 0";                     \
    }                                                                           \
  } while (0)

// One solid frame per tick, in the given encoding. An RGBA frame carries alpha 7, to show that
// the replay supplies an opaque alpha and does not carry the recorded one.
template <holoscan::schema::ImageEncoding Encoding, std::size_t Channels>
class SolidOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 20ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    const std::size_t bytes = static_cast<std::size_t>(kWidth) * kHeight * Channels;
    auto pixels = std::make_unique<std::byte[]>(bytes);
    for (std::size_t pixel = 0; pixel < bytes; pixel += Channels) {
      for (std::size_t channel = 0; channel < 3; ++channel) {
        pixels[pixel + channel] = static_cast<std::byte>(kColour[channel]);
      }
      if constexpr (Channels == 4) {
        pixels[pixel + 3] = std::byte{7};
      }
    }
    auto tensor = detail::adopt_host_buffer({static_cast<std::int64_t>(bytes)},
                                            DLDataType{kDLUInt, 8U, 1U}, std::move(pixels));
    if (!tensor) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, std::string(tensor.error())});
    }
    holoscan::schema::ImageT value;
    value.width = static_cast<std::int32_t>(kWidth);
    value.height = static_cast<std::int32_t>(kHeight);
    value.encoding = Encoding;
    value.data = std::make_unique<holoscan::Tensor>(std::move(*tensor));
    return out.emit(value);
  }

  holoscan::Output<holoscan::schema::ImageT> out;
};

using SolidRgbOp = SolidOp<holoscan::schema::ImageEncoding_RGB8, 3>;
using SolidRgbaOp = SolidOp<holoscan::schema::ImageEncoding_RGBA8, 4>;

class RecordingGraph final : public holoscan::Graph {
 public:
  explicit RecordingGraph(isaac::os::mcap_logger::HsdkMcapLogger& logger)
      : Graph("compressed_video_recording"), logger_(logger) {}

 private:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    const auto encoded = [](VideoCodec codec) {
      return isaac::os::mcap_logger::TopicLogOptions{
          .encode = isaac::os::mcap_logger::VideoEncodeOptions{.codec = codec}};
    };
    const auto contract = [](holoscan::schema::ImageEncoding encoding) {
      return isaac::os::image_schema::ImageTensorFieldContract(kWidth, kHeight, encoding);
    };

    auto rgb = op<SolidRgbOp>("rgb_source");
    logger_.attach<holoscan::schema::ImageT>(*this, kRgbTopic, rgb->out,
                                             contract(holoscan::schema::ImageEncoding_RGB8),
                                             encoded(VideoCodec::kH264));
    auto rgba = op<SolidRgbaOp>("rgba_source");
    logger_.attach<holoscan::schema::ImageT>(*this, kRgbaTopic, rgba->out,
                                             contract(holoscan::schema::ImageEncoding_RGBA8),
                                             encoded(VideoCodec::kH264));
    // Only where the encoder supports it: otherwise logger.start() would throw inside the shared
    // fixture and every test in this file would fail instead of skipping.
    if (FrameEncoder::hardware_available(0, VideoCodec::kH265)) {
      auto h265 = op<SolidRgbOp>("h265_source");
      logger_.attach<holoscan::schema::ImageT>(*this, kH265Topic, h265->out,
                                               contract(holoscan::schema::ImageEncoding_RGB8),
                                               encoded(VideoCodec::kH265));
    }
  }

  isaac::os::mcap_logger::HsdkMcapLogger& logger_;
};

// Records the encoded topics once for the whole executable.
const std::filesystem::path& encoded_fixture() {
  static const std::filesystem::path path = [] {
    static const isaac::os::mcap_logger::ScopedTempDirectory scratch;
    isaac::os::mcap_logger::HsdkMcapLogger logger(isaac::os::mcap_logger::HsdkMcapLoggerOptions{
        .output_path = scratch.path() / "encoded.mcap", .timestamp_output_path = false});
    RecordingGraph graph(logger);
    const holoscan::ExecutionPlan plan = holoscan::compile(graph);
    EXPECT_TRUE(plan.ok());
    logger.start(plan.endpoint_catalog());
    holoscan::RunSession session = holoscan::run_async(plan);
    std::this_thread::sleep_for(800ms);
    session.request_stop();
    session.wait();
    logger.shutdown();
    return logger.output_path().value_or(std::filesystem::path{});
  }();
  return path;
}

struct Replayed {
  std::vector<ReceivedImage> frames;
  std::uint64_t recorded = 0;
};

// Replays one topic of the fixture to the end and returns every frame it delivered, with the
// number of messages the recording holds on it.
Replayed replay(const char* topic) {
  ReceivedLog::reset();
  const std::filesystem::path& path = encoded_fixture();
  std::optional<std::uint64_t> recorded;
  ReplayerTestGraph graph([&](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(
        path, ReplayOptions{.playback_speed = 10.0},
        Topic<holoscan::schema::ImageT>{
            topic, isaac::os::image_schema::ImageTensorFieldContract(8U * 1024U * 1024U)});
    recorded = replayer->message_count(topic);
    auto sink = owner.op<ImageSinkOp>("sink");
    replayer->attach<holoscan::schema::ImageT>(owner, topic, sink->in);
    return replayer;
  });
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  // Checked before the run starts, so a failure here leaves no session running.
  if (!plan.ok() || !recorded.has_value() || *recorded < 10U) {
    ADD_FAILURE() << "the fixture did not record enough frames of '" << topic << "'";
    return {};
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  graph.replayer().start(plan.endpoint_catalog());
  Replayed replayed;
  replayed.recorded = *recorded;
  replayed.frames = ReceivedLog::wait_for(static_cast<std::size_t>(*recorded), 30s);
  graph.replayer().stop();
  session.request_stop();
  session.wait();
  return replayed;
}

// Checks every pixel of a frame, not a sample of it: an uninitialised row or a plane read at the
// wrong offset shows up at the edges, not at the centre. Reports the first few offenders only.
// What is wrong with the pixel starting at offset, one description per offending channel.
std::vector<std::string> pixel_problems(const std::vector<std::uint8_t>& bytes, std::size_t offset,
                                        std::size_t channels) {
  std::vector<std::string> problems;
  const std::size_t pixel = offset / channels;
  for (std::size_t channel = 0; channel < 3; ++channel) {
    const int difference = std::abs(static_cast<int>(bytes[offset + channel]) - kColour[channel]);
    if (difference > kTolerance) {
      problems.push_back(std::format("pixel {} channel {} is {}, not the recorded {} within {}",
                                     pixel, channel, bytes[offset + channel], kColour[channel],
                                     kTolerance));
    }
  }
  if (channels == 4 && bytes[offset + 3] != 255U) {
    // The recorded alpha (7) never reaches the video; the replay supplies an opaque one.
    problems.push_back(std::format("pixel {} alpha is {}, not opaque", pixel, bytes[offset + 3]));
  }
  return problems;
}

// Every pixel is checked; only the first few offenders are reported.
void expect_every_pixel_close(const std::vector<std::uint8_t>& bytes, std::size_t channels) {
  constexpr std::size_t kMaxReported = 3;
  std::size_t reported = 0;
  for (std::size_t offset = 0; offset + channels <= bytes.size(); offset += channels) {
    for (const std::string& problem : pixel_problems(bytes, offset, channels)) {
      ADD_FAILURE() << problem;
      if (++reported == kMaxReported) {
        return;
      }
    }
  }
}

void expect_replays(const char* topic, holoscan::schema::ImageEncoding encoding,
                    std::size_t channels) {
  ASSERT_FALSE(encoded_fixture().empty());
  const Replayed replayed = replay(topic);
  ASSERT_GE(replayed.recorded, 10U);
  // Every recorded frame comes back out. A decoder that held each picture until the next arrived
  // would deliver one fewer, dropping the first and never emitting the last.
  EXPECT_EQ(replayed.frames.size(), replayed.recorded);
  for (const ReceivedImage& image : replayed.frames) {
    EXPECT_EQ(image.width, static_cast<std::int32_t>(kWidth));
    EXPECT_EQ(image.height, static_cast<std::int32_t>(kHeight));
    EXPECT_EQ(image.encoding, encoding);
    EXPECT_EQ(image.device_type, kDLCPU);
    ASSERT_EQ(image.bytes.size(), static_cast<std::size_t>(kWidth) * kHeight * channels);
    expect_every_pixel_close(image.bytes, channels);
  }
}

}  // namespace

TEST(CompressedVideoReplay, ReplaysAnRgbTopicAsHostRgbImages) {
  SKIP_UNLESS_VIDEO_HARDWARE("h264", VideoCodec::kH264);
  expect_replays(kRgbTopic, holoscan::schema::ImageEncoding_RGB8, 3);
}

TEST(CompressedVideoReplay, ReplaysAnRgbaTopicWithOpaqueAlpha) {
  SKIP_UNLESS_VIDEO_HARDWARE("h264", VideoCodec::kH264);
  expect_replays(kRgbaTopic, holoscan::schema::ImageEncoding_RGBA8, 4);
}

TEST(CompressedVideoReplay, ReplaysAnH265Topic) {
  SKIP_UNLESS_VIDEO_HARDWARE("h265", VideoCodec::kH265);
  expect_replays(kH265Topic, holoscan::schema::ImageEncoding_RGB8, 3);
}

}  // namespace isaac::os::mcap_replayer
