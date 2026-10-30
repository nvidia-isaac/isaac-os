/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/run.hpp"
#include "holoscan/core/temporal_contract.hpp"
#include "holoscan/logger/logger.hpp"
#include "holoscan/memory/tensor_storage.hpp"
#include "holoscan/schema/binary_schema.hpp"
#include "holoscan/schema/schema_package.hpp"
#include "holoscan/sensor_io/image_metadata.hpp"
#include "holoscan/sensor_io/schemas/image_generated.h"
#include "holoscan/sensor_io/sensor_schema_package.hpp"
#include "holoscan/time/realtime_clock.hpp"
#include <flatbuffers/flatbuffers.h>

#include "common/image_schema/image_schema.h"
#include "common/mcap_logger/mcap_logger.h"
#include "common/mcap_logger/tests/scoped_temp_directory.h"
#include "common/mcap_replayer/decoder.h"
#include "common/mcap_replayer/mcap_file.h"
#include "common/mcap_replayer/mcap_replayer.h"
#include "common/mcap_replayer/message.h"
#include "common/mcap_replayer/tests/mcap_test_support.h"
#include "common/mcap_replayer/tests/replay_test_harness.h"
#include "common/mcap_replayer/typed_port.h"
#include "pose_cpu_schema_traits.h"
#include "tensor_frame_generated.h"

namespace isaac::os::mcap_replayer {
namespace {

using namespace std::chrono_literals;
using isaac::os::mcap_replayer::test_support::read_message;
using test_support::ImageSinkOp;
using test_support::ReceivedImage;
using test_support::ReceivedLog;
using test_support::ReplayerTestGraph;
using test_support::tensor_bytes_of;

// The generated tensor envelope types, whose namespace comes from tensor_frame.fbs rather than
// from the C++ library namespace mcap_logger's own code lives in.
namespace wire = ::isaac::mcap_logger;

constexpr const char* kImageTopic = "image";

// What tells a tensor-schema topic from a plain one, at compile time: the first needs the contract
// its imported edge is built from, the second is just named.
static_assert(holoscan::detail::HasDescriptorTensorSchemaNativeTraits<holoscan::schema::ImageT>);
static_assert(!holoscan::detail::HasDescriptorTensorSchemaNativeTraits<isaac::PoseCpuT>);
static_assert(std::is_aggregate_v<Topic<isaac::PoseCpuT>>);
static_assert(!std::is_constructible_v<Topic<holoscan::schema::ImageT>, std::string_view>,
              "a tensor-schema topic must be given its contract");
static_assert(
    std::is_constructible_v<Topic<holoscan::schema::ImageT>, std::string_view,
                            holoscan::DescriptorTensorTopicContract<holoscan::schema::ImageT>>);

// The replayed image topic, with the contract it is imported under: frames up to 8 MiB.
Topic<holoscan::schema::ImageT> image_topic() {
  return Topic<holoscan::schema::ImageT>{
      kImageTopic, isaac::os::image_schema::ImageTensorFieldContract(8U * 1024U * 1024U)};
}
constexpr const char* kPoseTopic = "pose";
constexpr std::uint32_t kImageWidth = 4;
constexpr std::uint32_t kImageHeight = 4;
constexpr std::size_t kBytesPerPixel = 3;  // RGB8.
constexpr std::size_t kFrameBytes =
    static_cast<std::size_t>(kImageWidth) * kImageHeight * kBytesPerPixel;
constexpr std::uint64_t kLogTimeNs = 123456789ULL;

// A recognisable, position-dependent fill: every byte depends on both the frame it belongs to
// and its own offset in that frame, so a frame whose bytes were reordered, truncated or shifted
// by replay is distinguishable from one that arrived intact -- which a constant fill is not.
std::vector<std::uint8_t> frame_pattern(std::uint8_t seed, std::size_t count) {
  std::vector<std::uint8_t> bytes(count);
  for (std::size_t index = 0; index < count; ++index) {
    bytes[index] = static_cast<std::uint8_t>((seed + index) & 0xFF);
  }
  return bytes;
}

// True when bytes is the pattern frame_pattern() produces for whatever seed its first byte
// implies -- the seed itself is the recorded frame's sequence number, which a replay consumer
// has no independent way to know.
bool matches_frame_pattern(const std::vector<std::uint8_t>& bytes) {
  return !bytes.empty() && bytes == frame_pattern(bytes.front(), bytes.size());
}

// Encodes a holoscan.schema.Image table holding only the metadata fields a tensor-schema
// recording puts in the envelope's nested metadata: the image's own Tensor field travels
// out-of-band as a separate recorded tensor, never inline in this table.
std::vector<std::byte> encode_image_metadata(std::uint32_t width, std::uint32_t height,
                                             holoscan::schema::ImageEncoding encoding) {
  flatbuffers::FlatBufferBuilder builder;
  const auto frame_id_offset = builder.CreateString("camera");
  const auto header_offset = holoscan::schema::CreateHeader(
      builder, /*stamp=*/0, frame_id_offset, /*capture_timestamp_ns=*/0, /*device_sequence=*/0);
  const auto image_offset =
      holoscan::schema::CreateImage(builder, header_offset, static_cast<std::int32_t>(height),
                                    static_cast<std::int32_t>(width), encoding);
  holoscan::schema::FinishImageBuffer(builder, image_offset);
  const auto* data = reinterpret_cast<const std::byte*>(builder.GetBufferPointer());
  return std::vector<std::byte>(data, data + builder.GetSize());
}

// Builds a complete recorded tensor-stream payload by hand, carrying one tensor: the same shape
// the logger writes, but with every field under this test's control so a malformed recording can
// be expressed as easily as a well-formed one.
std::vector<std::byte> build_tensor_frame(std::span<const std::byte> metadata_bytes,
                                          const std::string& metadata_schema_name,
                                          std::span<const std::byte> metadata_binary_schema,
                                          const std::vector<std::uint32_t>& shape,
                                          wire::TensorDType dtype,
                                          const std::vector<std::uint8_t>& tensor_bytes) {
  flatbuffers::FlatBufferBuilder builder;

  const std::vector<std::uint8_t> metadata_u8(
      reinterpret_cast<const std::uint8_t*>(metadata_bytes.data()),
      reinterpret_cast<const std::uint8_t*>(metadata_bytes.data()) + metadata_bytes.size());
  const std::vector<std::uint8_t> binary_schema_u8(
      reinterpret_cast<const std::uint8_t*>(metadata_binary_schema.data()),
      reinterpret_cast<const std::uint8_t*>(metadata_binary_schema.data()) +
          metadata_binary_schema.size());

  const auto entry_offset = wire::CreateTensorEntryDirect(builder, &shape, dtype, &tensor_bytes);
  const std::vector<flatbuffers::Offset<wire::TensorEntry>> entries{entry_offset};
  const auto frame_offset = wire::CreateTensorFrameDirect(
      builder, &entries, &metadata_u8, metadata_schema_name.c_str(), &binary_schema_u8);
  wire::FinishTensorFrameBuffer(builder, frame_offset);

  const auto* data = reinterpret_cast<const std::byte*>(builder.GetBufferPointer());
  return std::vector<std::byte>(data, data + builder.GetSize());
}

// As build_tensor_frame(), with the metadata and schema identification a well-formed recording
// of a holoscan::schema::ImageT stream carries, so a test varies only the tensor it is about.
std::vector<std::byte> build_image_frame(const std::vector<std::uint32_t>& shape,
                                         wire::TensorDType dtype,
                                         const std::vector<std::uint8_t>& tensor_bytes) {
  const std::vector<std::byte> metadata =
      encode_image_metadata(kImageWidth, kImageHeight, holoscan::schema::ImageEncoding_RGB8);
  const auto& identity = holoscan::schema_package<holoscan::schema::ImageT>().identity();
  return build_tensor_frame(metadata, std::string(identity.canonical_name),
                            holoscan::binary_schema<holoscan::schema::ImageT>(), shape, dtype,
                            tensor_bytes);
}

// What one payload decoded to, if it decoded at all.
struct Decoded {
  bool succeeded = false;
  std::vector<std::byte> metadata;
  std::vector<holoscan::Tensor> tensors;
};

// Runs payload through the message builder a tensor topic is replayed with.
Decoded decode_tensor_payload(std::span<const std::byte> payload) {
  const MessageBuilder build = make_tensor_message_builder<holoscan::schema::ImageT>();
  const auto message = build(payload, {}, kLogTimeNs);
  if (!message) {
    return {};
  }
  EXPECT_EQ((*message)->log_time_ns, kLogTimeNs);
  auto* const typed = dynamic_cast<TensorMessage*>(message->get());
  EXPECT_NE(typed, nullptr);
  if (typed == nullptr) {
    return {};
  }
  return Decoded{.succeeded = true,
                 .metadata = std::move(typed->metadata),
                 .tensors = std::move(typed->tensors)};
}

// Emits one small RGB8 frame per tick, each filled with the position-dependent pattern above so
// a replayed frame can be checked byte for byte.
class FixtureImageOp final : public holoscan::Operator<> {
 public:
  static constexpr auto kPeriod = 10ms;

  void setup(holoscan::OperatorSpec& spec) override { spec.output(out_image, "out_image"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kPeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    const std::vector<std::uint8_t> pattern =
        frame_pattern(static_cast<std::uint8_t>(sequence_), kFrameBytes);
    ++sequence_;

    auto pixels = std::make_unique<std::uint8_t[]>(kFrameBytes);
    std::memcpy(pixels.get(), pattern.data(), kFrameBytes);

    const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(kFrameBytes)};
    auto tensor = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                                        holoscan::MemoryStorageType::kSystem, pixels.get(),
                                        [](void* ptr) -> holoscan::expected<void, holoscan::Error> {
                                          delete[] static_cast<std::uint8_t*>(ptr);
                                          return {};
                                        });
    if (!tensor) {
      return holoscan::make_unexpected(tensor.error());
    }
    pixels.release();  // make_tensor() took ownership; its release function above frees this now.

    holoscan::schema::ImageT value;
    value.height = static_cast<std::int32_t>(kImageHeight);
    value.width = static_cast<std::int32_t>(kImageWidth);
    value.encoding = holoscan::schema::ImageEncoding_RGB8;
    value.data = std::make_unique<holoscan::Tensor>(std::move(*tensor));
    return out_image.emit(value);
  }

  holoscan::Output<holoscan::schema::ImageT> out_image;

 private:
  int sequence_ = 0;
};

// Emits a pose at a period sharing no common factor with the image's, so a recording holds a flat
// topic interleaved unevenly with a tensor one. Its sequence number rides in the x coordinate.
class FixturePoseOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out_pose, "out_pose"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 7ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    isaac::PoseCpuT value;
    value.position = holoscan::schema::Vector3d(static_cast<double>(sequence_), 0.0, 0.0);
    value.orientation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);
    ++sequence_;
    return out_pose.emit(value);
  }

  holoscan::Output<isaac::PoseCpuT> out_pose;

 private:
  int sequence_ = 0;
};

class FixtureGraph final : public holoscan::Graph {
 public:
  explicit FixtureGraph(isaac::os::mcap_logger::HsdkMcapLogger& logger)
      : Graph("tensor_decoder_fixture"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto image_source = op<FixtureImageOp>("image_source");
    logger_.attach<holoscan::schema::ImageT>(
        *this, kImageTopic, image_source->out_image,
        isaac::os::image_schema::ImageTensorFieldContract(kImageWidth, kImageHeight,
                                                          holoscan::schema::ImageEncoding_RGB8));
    auto pose_source = op<FixturePoseOp>("pose_source");
    logger_.attach<isaac::PoseCpuT>(*this, kPoseTopic, pose_source->out_pose);
  }

 private:
  isaac::os::mcap_logger::HsdkMcapLogger& logger_;
};

// Records a real tensor-stream recording once for the whole executable: a hand-built payload
// proves the decoder, but only bytes a real recorder wrote prove replay of one.
const std::filesystem::path& tensor_fixture() {
  static const std::filesystem::path path = [] {
    using isaac::os::mcap_logger::HsdkMcapLogger;
    using isaac::os::mcap_logger::HsdkMcapLoggerOptions;

    static const isaac::os::mcap_logger::ScopedTempDirectory scratch;
    const std::filesystem::path directory = scratch.path();

    HsdkMcapLogger logger(HsdkMcapLoggerOptions{.output_path = directory / "fixture.mcap",
                                                .timestamp_output_path = false});
    FixtureGraph graph(logger);
    const holoscan::ExecutionPlan plan = holoscan::compile(graph);
    EXPECT_TRUE(plan.ok());
    logger.start(plan.endpoint_catalog());
    holoscan::RunSession session = holoscan::run_async(plan);
    std::this_thread::sleep_for(2s);
    session.request_stop();
    session.wait();
    logger.shutdown();
    const std::optional<std::filesystem::path>& written = logger.output_path();
    EXPECT_TRUE(written.has_value()) << "the fixture logger produced no file to read back";
    return written.value_or(std::filesystem::path{});
  }();
  return path;
}

// What the recording actually holds on one topic, read back from the file rather than assumed
// from what the fixture meant to write: a recorder under load can drop a message, and replay can
// only be held to what was recorded.
struct RecordedTopic {
  std::vector<std::uint64_t> log_times;
  std::vector<std::vector<std::byte>> payloads;
};

std::map<std::string, RecordedTopic> read_recording(const std::filesystem::path& path) {
  McapFile opened = McapFile::open(path);
  std::map<std::string, RecordedTopic> recording;
  RecordedMessage message;
  while (read_message(opened, message)) {
    RecordedTopic& topic = recording[opened.topic_of(message.channel_id).value()];
    topic.log_times.push_back(message.log_time_ns);
    topic.payloads.emplace_back(message.payload.begin(), message.payload.end());
  }
  return recording;
}

// The pattern seed of every frame the recording holds on the image topic, in recorded order.
std::vector<std::uint8_t> recorded_frame_seeds(const std::filesystem::path& path) {
  std::vector<std::uint8_t> seeds;
  const std::map<std::string, RecordedTopic> recording = read_recording(path);
  const auto images = recording.find(kImageTopic);
  EXPECT_NE(images, recording.end());
  if (images == recording.end()) {
    return seeds;
  }
  for (const std::vector<std::byte>& payload : images->second.payloads) {
    const Decoded decoded = decode_tensor_payload(payload);
    EXPECT_TRUE(decoded.succeeded && !decoded.tensors.empty());
    if (!decoded.succeeded || decoded.tensors.empty()) {
      return seeds;
    }
    const std::vector<std::uint8_t> bytes = tensor_bytes_of(decoded.tensors.front());
    EXPECT_FALSE(bytes.empty());
    seeds.push_back(bytes.empty() ? 0U : bytes.front());
  }
  return seeds;
}

// Counts the poses replayed to it, for the test that replays them beside the image stream.
class PoseSinkOp final : public holoscan::Operator<> {
 public:
  static void reset() { received_ = 0; }
  [[nodiscard]] static std::size_t received() { return received_; }

  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(256); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{in});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = in.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    ++received_;
    return {};
  }

  holoscan::Input<isaac::PoseCpuT> in;

 private:
  static inline std::atomic<std::size_t> received_{0};
};

}  // namespace

TEST(TensorDecoder, ReconstructsTheMetadataAndPixelsOfARecordedTensor) {
  const std::vector<std::uint8_t> pixels = frame_pattern(0, kFrameBytes);
  const std::vector<std::byte> metadata =
      encode_image_metadata(kImageWidth, kImageHeight, holoscan::schema::ImageEncoding_RGB8);
  const std::vector<std::byte> payload =
      build_image_frame({static_cast<std::uint32_t>(kFrameBytes)}, wire::TensorDType_UInt8, pixels);

  const Decoded decoded = decode_tensor_payload(payload);

  ASSERT_TRUE(decoded.succeeded);
  // The recorded metadata bytes must pass through unchanged, not be re-derived from a
  // materialized ImageT -- this is the property the direct-publish redesign depends on.
  EXPECT_EQ(decoded.metadata, metadata);
  ASSERT_EQ(decoded.tensors.size(), 1U);
  const holoscan::Tensor& tensor = decoded.tensors.front();
  const auto shape = tensor.shape();
  ASSERT_EQ(shape.size(), 1U);
  EXPECT_EQ(shape[0], static_cast<std::int64_t>(kFrameBytes));
  EXPECT_EQ(tensor.device().device_type, kDLCPU);
  EXPECT_EQ(tensor_bytes_of(tensor), pixels);
}

// Deliberately a well-formed message of a different kind, not random bytes: random bytes are
// rejected on size and offset grounds before the marker that identifies a recorded tensor message
// is ever looked at, so they would pass this test with that marker check deleted. A structurally
// valid message carrying the wrong marker is what proves the marker is what rejects it.
TEST(TensorDecoder, RejectsAWellFormedPayloadThatIsNotARecordedTensor) {
  const std::vector<std::byte> other_kind_of_message =
      encode_image_metadata(kImageWidth, kImageHeight, holoscan::schema::ImageEncoding_RGB8);

  EXPECT_FALSE(decode_tensor_payload(other_kind_of_message).succeeded);
}

TEST(TensorDecoder, RejectsAPayloadThatIsNotAMessageAtAll) {
  const std::vector<std::byte> garbage(16, std::byte{0xFF});

  EXPECT_FALSE(decode_tensor_payload(garbage).succeeded);
}

// A recording carrying no tensor where one was recorded must fail outright rather than decode
// into a value whose tensor field is empty: reporting success for a message that could not be
// rebuilt pushes the check onto every consumer downstream.
TEST(TensorDecoder, RejectsARecordingCarryingFewerTensorsThanTheTypeHolds) {
  const std::vector<std::byte> metadata =
      encode_image_metadata(kImageWidth, kImageHeight, holoscan::schema::ImageEncoding_RGB8);
  const auto& identity = holoscan::schema_package<holoscan::schema::ImageT>().identity();

  flatbuffers::FlatBufferBuilder builder;
  const std::vector<std::uint8_t> metadata_u8(
      reinterpret_cast<const std::uint8_t*>(metadata.data()),
      reinterpret_cast<const std::uint8_t*>(metadata.data()) + metadata.size());
  const std::span<const std::byte> binary_schema =
      holoscan::binary_schema<holoscan::schema::ImageT>();
  const std::vector<std::uint8_t> binary_schema_u8(
      reinterpret_cast<const std::uint8_t*>(binary_schema.data()),
      reinterpret_cast<const std::uint8_t*>(binary_schema.data()) + binary_schema.size());
  const std::vector<flatbuffers::Offset<wire::TensorEntry>> no_entries;
  const auto frame_offset = wire::CreateTensorFrameDirect(
      builder, &no_entries, &metadata_u8, std::string(identity.canonical_name).c_str(),
      &binary_schema_u8);
  wire::FinishTensorFrameBuffer(builder, frame_offset);
  const auto* data = reinterpret_cast<const std::byte*>(builder.GetBufferPointer());
  const std::vector<std::byte> payload(data, data + builder.GetSize());

  EXPECT_FALSE(decode_tensor_payload(payload).succeeded);
}

// Nothing in the recorded format couples a tensor's declared shape to the size of the data it
// actually carries, so a malformed or hostile recording can claim any shape it likes. Accepting
// one would hand downstream code a tensor whose reported size reaches past its allocation.
TEST(TensorDecoder, RejectsARecordedTensorWhoseShapeDoesNotMatchItsData) {
  const std::vector<std::uint8_t> undersized(8, 0);
  const std::vector<std::byte> payload =
      build_image_frame({1U << 20}, wire::TensorDType_UInt8, undersized);

  EXPECT_FALSE(decode_tensor_payload(payload).succeeded);
}

// The same hazard by a different route: a shape whose dimensions multiply out past what a byte
// count can hold must be rejected outright, not wrapped around into a small, plausible one that
// then appears to agree with the data actually carried.
TEST(TensorDecoder, RejectsARecordedTensorWhoseShapeOverflowsAByteCount) {
  constexpr std::uint32_t kHuge = std::numeric_limits<std::uint32_t>::max();
  const std::vector<std::uint8_t> data(8, 0);
  const std::vector<std::byte> payload =
      build_image_frame({kHuge, kHuge, kHuge}, wire::TensorDType_Float64, data);

  EXPECT_FALSE(decode_tensor_payload(payload).succeeded);
}

// A decoder test cannot show that a reconstructed tensor survives publication: the port is where
// a tensor-carrying message is handled differently from any other, so replay of a real recording
// through a real graph is the only thing that proves the pixels arrive.
TEST(TensorDecoder, ReplaysARecordedTensorStreamThroughAGraphWithItsPixelsIntact) {
  const std::filesystem::path path = tensor_fixture();
  ReceivedLog::reset();

  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer =
        std::make_unique<McapReplayer>(path, ReplayOptions{.playback_speed = 10.0}, image_topic());
    auto sink = owner.op<ImageSinkOp>("sink");
    replayer->attach<holoscan::schema::ImageT>(owner, kImageTopic, sink->in);
    return replayer;
  });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);
  graph.replayer().start(plan.endpoint_catalog());
  const std::vector<ReceivedImage> received = ReceivedLog::wait_for(3, 30s);
  graph.replayer().stop();
  session.request_stop();
  session.wait();

  ASSERT_GE(received.size(), 3U);
  for (const ReceivedImage& image : received) {
    EXPECT_EQ(image.width, static_cast<std::int32_t>(kImageWidth));
    EXPECT_EQ(image.height, static_cast<std::int32_t>(kImageHeight));
    EXPECT_EQ(image.encoding, holoscan::schema::ImageEncoding_RGB8);
    ASSERT_TRUE(image.has_tensor);
    ASSERT_EQ(image.shape.size(), 1U);
    EXPECT_EQ(image.shape[0], static_cast<std::int64_t>(kFrameBytes));
    EXPECT_EQ(image.device_type, kDLCPU);
    ASSERT_EQ(image.bytes.size(), kFrameBytes);
    EXPECT_TRUE(matches_frame_pattern(image.bytes))
        << "the replayed pixels are not the ones that were recorded";
  }
  // Each frame's pattern starts at its own sequence number, so arrivals carrying the recording's
  // own seeds in its own order show the frames came through in recorded order with none lost or
  // repeated -- intact pixels alone would not rule out a frame delivered in another's place.
  // Replay starts at the recording's first message and this run does not loop, so what arrived
  // is a prefix of what was recorded.
  const std::vector<std::uint8_t> recorded = recorded_frame_seeds(path);
  ASSERT_GE(recorded.size(), received.size());
  for (std::size_t index = 0; index < received.size(); ++index) {
    EXPECT_EQ(received[index].bytes.front(), recorded[index])
        << "frame " << index << " is not the one the recording holds in that place";
  }
}

// Tensor and flat topics are published through different kinds of port, a tensor message takes
// longer to put out, and every topic of one replayer shares a single pacing thread. So only a run
// carrying both at once shows that neither disturbs the other's delivery or timing.
TEST(TensorDecoder, ReplaysATensorTopicAndAFlatTopicTogetherEachOnItsOwnSchedule) {
  const std::filesystem::path path = tensor_fixture();
  const std::map<std::string, RecordedTopic> recording = read_recording(path);
  ASSERT_GE(recording.count(kImageTopic), 1U);
  ASSERT_GE(recording.count(kPoseTopic), 1U);
  const RecordedTopic& images = recording.at(kImageTopic);
  const RecordedTopic& poses = recording.at(kPoseTopic);
  ASSERT_GE(images.log_times.size(), 10U);
  ASSERT_GE(poses.log_times.size(), 10U);

  ReceivedLog::reset();
  PoseSinkOp::reset();
  // The recording's own speed. The fixture spans about 2 s, long enough that a stall of a few tens
  // of milliseconds as the run starts stays a small share of the span the rate is measured over.
  constexpr double kSpeed = 1.0;
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer =
        std::make_unique<McapReplayer>(path, ReplayOptions{.playback_speed = kSpeed}, image_topic(),
                                       Topic<isaac::PoseCpuT>{kPoseTopic});
    auto image_sink = owner.op<ImageSinkOp>("image_sink");
    replayer->attach<holoscan::schema::ImageT>(owner, kImageTopic, image_sink->in);
    auto pose_sink = owner.op<PoseSinkOp>("pose_sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, pose_sink->in);
    return replayer;
  });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);
  graph.replayer().start(plan.endpoint_catalog());
  const ReplayStats& stats = graph.replayer().stats();
  const auto deadline = std::chrono::steady_clock::now() + 30s;
  while ((stats.emitted(kImageTopic) + stats.refused(kImageTopic) < images.log_times.size() ||
          stats.emitted(kPoseTopic) + stats.refused(kPoseTopic) < poses.log_times.size()) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(2ms);
  }
  const std::vector<ReceivedImage> received =
      ReceivedLog::wait_for(stats.emitted(kImageTopic), 10s);
  const auto pose_deadline = std::chrono::steady_clock::now() + 10s;
  while (PoseSinkOp::received() < stats.emitted(kPoseTopic) &&
         std::chrono::steady_clock::now() < pose_deadline) {
    std::this_thread::sleep_for(2ms);
  }
  graph.replayer().stop();
  session.request_stop();
  session.wait();

  EXPECT_EQ(stats.emitted(kImageTopic), images.log_times.size());
  EXPECT_EQ(stats.emitted(kPoseTopic), poses.log_times.size());
  EXPECT_EQ(stats.refused(kImageTopic), 0U);
  EXPECT_EQ(stats.refused(kPoseTopic), 0U);
  EXPECT_EQ(received.size(), images.log_times.size());
  EXPECT_EQ(PoseSinkOp::received(), poses.log_times.size());
  for (const ReceivedImage& image : received) {
    ASSERT_TRUE(image.has_tensor);
    ASSERT_EQ(image.bytes.size(), kFrameBytes);
    EXPECT_TRUE(matches_frame_pattern(image.bytes))
        << "the replayed pixels are not the ones that were recorded";
  }
  // Both topics share the pacing thread, and a tensor message takes longer to publish than a flat
  // one, so what this has to show is that neither topic is held back as the run goes on. A stall
  // as the run starts delays a few messages once, a few per cent of the recording at most; a
  // thread kept too busy to keep up falls further behind with every message, so most of what it
  // sends goes out late. The share below sits well between the two. A thread that is busy but
  // keeps up, with every message under the late threshold, is not what this looks for.
  constexpr double kMostLateShare = 0.10;
  for (const char* topic : {kImageTopic, kPoseTopic}) {
    EXPECT_LE(static_cast<double>(stats.late_arrivals(topic)),
              kMostLateShare * static_cast<double>(stats.emitted(topic)))
        << "topic '" << topic << "' went out more than " << ReplayStats::kLateThreshold.count()
        << " ms late on " << stats.late_arrivals(topic) << " of " << stats.emitted(topic)
        << " messages";
  }
}

TEST(TensorDecoder, RejectsATensorTopicAskedForAsATypeItWasNotRecordedAs) {
  // Construction throws before it reaches the one graph-authoring call it makes, so this does
  // not have to run from inside compose() -- and running it outside means the failure surfaces
  // as a construction error rather than being swallowed and re-reported as a composition
  // failure. verify_topic() logs the specific reason via HOLOSCAN_LOG_WARN rather than building
  // it into the exception, so this only checks that construction is rejected at all.
  holoscan::Graph graph("tensor_replay_rejection_test");
  EXPECT_THROW(McapReplayer(tensor_fixture(), Topic<isaac::PoseCpuT>{kImageTopic}),
               std::invalid_argument);
}

}  // namespace isaac::os::mcap_replayer
