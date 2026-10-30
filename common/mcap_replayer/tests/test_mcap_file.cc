/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/run.hpp"
#include "holoscan/core/temporal_contract.hpp"
#include "holoscan/memory/tensor_storage.hpp"
#include "holoscan/sensor_io/image_metadata.hpp"
#include "holoscan/time/realtime_clock.hpp"

#include "common/image_schema/image_schema.h"
#include "common/mcap_logger/mcap_logger.h"
#include "common/mcap_logger/tests/scoped_temp_directory.h"
#include "common/mcap_replayer/mcap_file.h"
#include "common/mcap_replayer/tests/mcap_test_support.h"
#include "mcap/reader.hpp"
#include "mcap/writer.hpp"
#include "pose_cpu_schema_traits.h"
#include "tensor_frame_generated.h"

namespace {

using isaac::os::mcap_replayer::McapFile;
using isaac::os::mcap_replayer::ReadResult;
using isaac::os::mcap_replayer::RecordedMessage;
using isaac::os::mcap_replayer::test_support::FixtureMessage;
using isaac::os::mcap_replayer::test_support::read_message;
using isaac::os::mcap_replayer::test_support::video_message_in;
using isaac::os::mcap_replayer::test_support::write_paired_video_file;

// Emits a fixed number of poses whose x coordinate is the sequence number, so a reader can
// assert both ordering and payload without depending on timing.
class FixturePoseOp final : public holoscan::Operator<> {
 public:
  static constexpr auto kPeriod = std::chrono::milliseconds(10);

  void setup(holoscan::OperatorSpec& spec) override { spec.output(out_pose, "out_pose"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kPeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    isaac::PoseCpuT value;
    value.position = holoscan::schema::Vector3d(static_cast<double>(sequence_), 0.0, 0.0);
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
      : Graph("mcap_file_fixture"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto source = op<FixturePoseOp>("source");
    logger_.attach<isaac::PoseCpuT>(*this, "pose", source->out_pose);
  }

 private:
  isaac::os::mcap_logger::HsdkMcapLogger& logger_;
};

std::filesystem::path write_fixture(const std::filesystem::path& directory) {
  using isaac::os::mcap_logger::HsdkMcapLogger;
  using isaac::os::mcap_logger::HsdkMcapLoggerOptions;

  HsdkMcapLogger logger(HsdkMcapLoggerOptions{.output_path = directory / "fixture.mcap",
                                              .timestamp_output_path = false});
  FixtureGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  EXPECT_TRUE(plan.ok());
  logger.start(plan.endpoint_catalog());
  holoscan::RunSession session = holoscan::run_async(plan);
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  session.request_stop();
  session.wait();
  logger.shutdown();
  // The logger writes a file only when it is configured with one; every fixture here is,
  // so an empty result means the fixture itself was mis-set up rather than the replayer.
  const std::optional<std::filesystem::path>& written = logger.output_path();
  EXPECT_TRUE(written.has_value()) << "the fixture logger produced no file to read back";
  return written.value_or(std::filesystem::path{});
}

constexpr std::uint32_t kTensorFixtureImageWidth = 4;
constexpr std::uint32_t kTensorFixtureImageHeight = 4;
constexpr std::size_t kTensorFixtureImageBytesPerPixel = 3;  // RGB8.
constexpr std::size_t kTensorFixtureImageFrameBytes =
    static_cast<std::size_t>(kTensorFixtureImageWidth) * kTensorFixtureImageHeight *
    kTensorFixtureImageBytesPerPixel;

// Emits a small RGB8 holoscan::schema::ImageT frame every tick, via the same host-buffer
// ownership pattern as applications/mcap_logger_reference/image_counter_op.h and
// test_mcap_logger_tensor_schema_integration.cc's ImageGeneratorOp -- content doesn't matter
// here, only that a real tensor-schema topic exists alongside the flat "pose" one so the
// TensorFrame envelope/peek path in McapFile exercises for real.
class FixtureImageOp final : public holoscan::Operator<> {
 public:
  static constexpr auto kPeriod = std::chrono::milliseconds(10);

  void setup(holoscan::OperatorSpec& spec) override { spec.output(out_image, "out_image"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kPeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto pixels = std::make_unique<std::uint8_t[]>(kTensorFixtureImageFrameBytes);
    std::memset(pixels.get(), sequence_, kTensorFixtureImageFrameBytes);

    const std::array<std::int64_t, 1> shape{
        static_cast<std::int64_t>(kTensorFixtureImageFrameBytes)};
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
    value.height = static_cast<std::int32_t>(kTensorFixtureImageHeight);
    value.width = static_cast<std::int32_t>(kTensorFixtureImageWidth);
    value.encoding = holoscan::schema::ImageEncoding_RGB8;
    value.data = std::make_unique<holoscan::Tensor>(std::move(*tensor));

    ++sequence_;
    return out_image.emit(value);
  }

  holoscan::Output<holoscan::schema::ImageT> out_image;

 private:
  int sequence_ = 0;
};

// Logs both a flat topic ("pose") and a tensor-schema topic ("image") into one file, so
// McapFile exercises the TensorFrame envelope match, the per-channel filtered peek, and nested
// schema-name/binary-schema extraction -- none of which the pose-only fixture above ever
// touches.
class TensorFixtureGraph final : public holoscan::Graph {
 public:
  explicit TensorFixtureGraph(isaac::os::mcap_logger::HsdkMcapLogger& logger)
      : Graph("mcap_file_tensor_fixture"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));

    auto pose_source = op<FixturePoseOp>("pose_source");
    logger_.attach<isaac::PoseCpuT>(*this, "pose", pose_source->out_pose);

    auto image_source = op<FixtureImageOp>("image_source");
    logger_.attach<holoscan::schema::ImageT>(
        *this, "image", image_source->out_image,
        isaac::os::image_schema::ImageTensorFieldContract(kTensorFixtureImageWidth,
                                                          kTensorFixtureImageHeight,
                                                          holoscan::schema::ImageEncoding_RGB8));
  }

 private:
  isaac::os::mcap_logger::HsdkMcapLogger& logger_;
};

std::filesystem::path write_tensor_fixture(const std::filesystem::path& directory) {
  using isaac::os::mcap_logger::HsdkMcapLogger;
  using isaac::os::mcap_logger::HsdkMcapLoggerOptions;

  HsdkMcapLogger logger(HsdkMcapLoggerOptions{.output_path = directory / "tensor_fixture.mcap",
                                              .timestamp_output_path = false});
  TensorFixtureGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  EXPECT_TRUE(plan.ok());
  logger.start(plan.endpoint_catalog());
  holoscan::RunSession session = holoscan::run_async(plan);
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  session.request_stop();
  session.wait();
  logger.shutdown();
  // The logger writes a file only when it is configured with one; every fixture here is,
  // so an empty result means the fixture itself was mis-set up rather than the replayer.
  const std::optional<std::filesystem::path>& written = logger.output_path();
  EXPECT_TRUE(written.has_value()) << "the fixture logger produced no file to read back";
  return written.value_or(std::filesystem::path{});
}

// Ground truth independent of McapFile's own reading logic: mcap's Statistics, read via a
// freshly opened raw mcap::McapReader rather than anything McapFile computed.
std::unordered_map<mcap::ChannelId, std::uint64_t> read_channel_message_counts(
    const std::filesystem::path& path) {
  mcap::McapReader reader;
  const mcap::Status open_status = reader.open(path.string());
  EXPECT_TRUE(open_status.ok()) << open_status.message;
  const mcap::Status summary_status =
      reader.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);
  EXPECT_TRUE(summary_status.ok()) << summary_status.message;
  const std::optional<mcap::Statistics>& stats = reader.statistics();
  EXPECT_TRUE(stats.has_value());
  return stats.value_or(mcap::Statistics{}).channelMessageCounts;
}

}  // namespace

TEST(McapFile, LooksUpTopicByChannelId) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const auto path = write_fixture(directory);

  McapFile file = McapFile::open(path);

  const std::vector<mcap::ChannelId> ids = file.channel_ids();
  ASSERT_EQ(ids.size(), 1U);
  const mcap::ChannelId pose_id = ids.front();

  EXPECT_EQ(file.topic_of(pose_id), "pose");
  EXPECT_TRUE(file.verify_topic<isaac::PoseCpuT>("pose"));

  // A channel id that was never assigned by the file is unknown to the getter.
  const mcap::ChannelId unknown_id = pose_id + 1000;
  EXPECT_FALSE(file.topic_of(unknown_id).has_value());
}

TEST(McapFile, IteratesMessagesInLogTimeOrder) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const auto path = write_fixture(directory);

  McapFile file = McapFile::open(path);

  const mcap::ChannelId pose_id = file.channel_ids().front();

  RecordedMessage message;
  std::uint64_t previous = 0;
  std::size_t count = 0;
  // Bounded at one file's worth rather than draining until next() returns false, so this test
  // does not depend on should_loop's default.
  const std::uint64_t total = read_channel_message_counts(path).at(pose_id);
  while (count < total && read_message(file, message)) {
    EXPECT_EQ(message.channel_id, pose_id);
    EXPECT_GE(message.log_time_ns, previous);
    EXPECT_FALSE(message.payload.empty());
    previous = message.log_time_ns;
    ++count;
  }
  EXPECT_EQ(count, total);
}

// The step past each message waits for the following call, so the end of the file is only reached
// by the call after the last message, and stays reached.
TEST(McapFile, ReadsEveryMessageOnceThenKeepsReportingTheEnd) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const auto path = write_fixture(directory);

  McapFile file = McapFile::open(path, /*should_loop=*/false);
  const std::vector<mcap::ChannelId> ids = file.channel_ids();
  ASSERT_EQ(ids.size(), 1U);
  const std::uint64_t total = read_channel_message_counts(path).at(ids.front());

  RecordedMessage message;
  std::uint64_t read = 0;
  while (read_message(file, message)) {
    ++read;
  }
  EXPECT_EQ(read, total);
  EXPECT_FALSE(read_message(file, message));
  EXPECT_FALSE(read_message(file, message));
}

TEST(McapFile, LoopsPastEndOfFileWithoutTheCallerObservingTheWrap) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const auto path = write_tensor_fixture(directory);

  const std::unordered_map<mcap::ChannelId, std::uint64_t> ground_truth =
      read_channel_message_counts(path);
  ASSERT_EQ(ground_truth.size(), 2U);

  McapFile file = McapFile::open(path, /*should_loop=*/true);

  std::uint64_t total_per_pass = 0;
  for (const auto& [channel_id, count] : ground_truth) {
    total_per_pass += count;
  }
  ASSERT_GT(total_per_pass, 0U);

  // Drain several times past what one pass over the file holds, so next() must wrap at least
  // twice, then reconcile per-channel tallies against the file's own statistics: if the wrap
  // were visible to the caller -- a short read, a duplicated or skipped message, a stall -- the
  // exact multiple below would not hold.
  constexpr std::uint64_t kPasses = 3;
  const std::uint64_t total_to_read = total_per_pass * kPasses;

  std::unordered_map<mcap::ChannelId, std::uint64_t> counted;
  RecordedMessage message;
  for (std::uint64_t i = 0; i < total_to_read; ++i) {
    ASSERT_TRUE(read_message(file, message)) << "next() returned false before draining "
                                             << total_to_read << " messages (at " << i << ")";
    ++counted[message.channel_id];
  }

  for (const auto& [channel_id, count] : ground_truth) {
    EXPECT_EQ(counted[channel_id], count * kPasses) << "channel " << channel_id;
  }
}

TEST(McapFile, ReportsHowManyMessagesEachChannelHoldsFromTheRecordingsStatistics) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const auto path = write_fixture(directory);

  McapFile file = McapFile::open(path);

  const std::vector<mcap::ChannelId> ids = file.channel_ids();
  ASSERT_EQ(ids.size(), 1U);
  EXPECT_EQ(file.message_count(ids.front()), read_channel_message_counts(path).at(ids.front()));
  // A channel the file never had holds nothing.
  EXPECT_EQ(file.message_count(ids.front() + 1000), 0U);
}

// A recording with one channel under the tensor envelope's schema, holding the given payload as
// its only message.
std::filesystem::path write_tensor_envelope_with(const std::filesystem::path& directory,
                                                 std::span<const std::byte> payload) {
  const auto path = directory / "envelope.mcap";
  mcap::McapWriter writer;
  const mcap::Status opened = writer.open(path.string(), mcap::McapWriterOptions(""));
  EXPECT_TRUE(opened.ok()) << opened.message;
  mcap::Schema schema(std::string(isaac::os::mcap_replayer::kTensorFrameSchemaName), "flatbuffer",
                      mcap::ByteArray{});
  writer.addSchema(schema);
  mcap::Channel channel("image", "flatbuffer", schema.id);
  writer.addChannel(channel);
  mcap::Message message{};
  message.channelId = channel.id;
  message.logTime = 1;
  message.publishTime = 1;
  message.data = payload.data();
  message.dataSize = payload.size();
  EXPECT_TRUE(writer.write(message).ok());
  writer.close();
  return path;
}

// The envelope's nested schema fields are optional in the format, so a frame without them is
// well-formed and verifies, yet cannot say what it carries.
TEST(McapFile, RejectsATensorEnvelopeThatDoesNotSayWhatTypeItCarries) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();

  flatbuffers::FlatBufferBuilder builder;
  isaac::mcap_logger::FinishTensorFrameBuffer(builder,
                                              isaac::mcap_logger::CreateTensorFrame(builder));
  const auto path = write_tensor_envelope_with(
      directory, std::as_bytes(std::span(builder.GetBufferPointer(), builder.GetSize())));

  McapFile file = McapFile::open(path);

  EXPECT_FALSE(file.verify_topic<holoscan::schema::ImageT>("image"));
}

// Too short to hold even the buffer's root offset, so nothing may be read from it before it is
// verified.
TEST(McapFile, RejectsATensorEnvelopeTooShortToBeAFrame) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const std::array<std::byte, 2> tiny{std::byte{1}, std::byte{2}};
  const auto path = write_tensor_envelope_with(directory, tiny);

  McapFile file = McapFile::open(path);

  EXPECT_FALSE(file.verify_topic<holoscan::schema::ImageT>("image"));
}

// Damage in the middle of the data ends iteration like the end of the file does, and the reason
// is kept. Wrapping around it would replay the readable start of the file over and over.
TEST(McapFile, ReportsAReadFailurePartwayThroughAndDoesNotWrapOverIt) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const auto path = directory / "damaged.mcap";

  constexpr std::uint64_t kMessages = 300;
  {
    mcap::McapWriterOptions options("");
    options.chunkSize = 1024;
    options.compression = mcap::Compression::None;
    mcap::McapWriter writer;
    ASSERT_TRUE(writer.open(path.string(), options).ok());
    mcap::Schema schema("isaac.PoseCpu", "flatbuffer", mcap::ByteArray{});
    writer.addSchema(schema);
    mcap::Channel channel("pose", "flatbuffer", schema.id);
    writer.addChannel(channel);
    const std::array<std::byte, 64> payload{};
    for (std::uint64_t index = 0; index < kMessages; ++index) {
      mcap::Message message{};
      message.channelId = channel.id;
      message.logTime = index + 1;
      message.publishTime = index + 1;
      message.data = payload.data();
      message.dataSize = payload.size();
      ASSERT_TRUE(writer.write(message).ok());
    }
    writer.close();
  }
  {
    // Overwrite the length of the first message record inside a chunk in the middle of the data, so
    // the reader finds a record that runs past the end of the chunk. The summary after the data is
    // untouched, so the file still opens. A chunk record is an opcode (1 byte), its length (8), the
    // chunk's start and end times (8 each), its uncompressed size (8) and CRC (4), the compression
    // name (4 for its length, none for no compression) and the records' size (8); the records
    // follow, each as an opcode and a length.
    mcap::McapReader raw;
    ASSERT_TRUE(raw.open(path.string()).ok());
    ASSERT_TRUE(raw.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan).ok());
    const auto& chunks = raw.chunkIndexes();
    ASSERT_GT(chunks.size(), 4U) << "the fixture should span several chunks";
    const auto damage_at = chunks[chunks.size() / 2].chunkStartOffset;
    raw.close();

    std::fstream damaged(path, std::ios::in | std::ios::out | std::ios::binary);
    constexpr std::streamoff kFirstRecordLength = 1 + 8 + 8 + 8 + 8 + 4 + 4 + 8 + 1;
    damaged.seekp(static_cast<std::streamoff>(damage_at) + kFirstRecordLength);
    const std::string garbage(8, static_cast<char>(0xFF));
    damaged.write(garbage.data(), static_cast<std::streamsize>(garbage.size()));
  }

  McapFile file = McapFile::open(path, /*should_loop=*/true);
  RecordedMessage message;
  std::uint64_t read = 0;
  while (read_message(file, message) && read < 10 * kMessages) {
    ++read;
  }

  EXPECT_LT(read, kMessages) << "the damaged chunk should have ended the pass early";
  EXPECT_FALSE(file.read_problem().empty());
  EXPECT_FALSE(read_message(file, message)) << "a failed read is final even when looping";
}

// Damage that makes the reader throw instead of reporting a problem (a chunk claiming a size it
// cannot allocate) while messages are being read is the same failure, and must not escape as an
// exception.
TEST(McapFile, TreatsAReaderThatThrowsOnDamagedDataAsAReadFailure) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const auto path = directory / "throwing.mcap";
  {
    mcap::McapWriterOptions options("");
    options.chunkSize = 1024;
    options.compression = mcap::Compression::None;
    mcap::McapWriter writer;
    ASSERT_TRUE(writer.open(path.string(), options).ok());
    mcap::Schema schema("isaac.PoseCpu", "flatbuffer", mcap::ByteArray{});
    writer.addSchema(schema);
    mcap::Channel channel("pose", "flatbuffer", schema.id);
    writer.addChannel(channel);
    const std::array<std::byte, 64> payload{};
    for (std::uint64_t index = 0; index < 300; ++index) {
      mcap::Message message{};
      message.channelId = channel.id;
      message.logTime = index + 1;
      message.publishTime = index + 1;
      message.data = payload.data();
      message.dataSize = payload.size();
      ASSERT_TRUE(writer.write(message).ok());
    }
    writer.close();
  }
  {
    // A chunk record is an opcode (1 byte), its length (8) and its start and end times (8 each),
    // and then its uncompressed size, which is what is overwritten here.
    mcap::McapReader raw;
    ASSERT_TRUE(raw.open(path.string()).ok());
    ASSERT_TRUE(raw.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan).ok());
    const auto& chunks = raw.chunkIndexes();
    ASSERT_GT(chunks.size(), 4U);
    const auto damage_at = chunks[chunks.size() / 2].chunkStartOffset;
    raw.close();

    std::fstream damaged(path, std::ios::in | std::ios::out | std::ios::binary);
    damaged.seekp(static_cast<std::streamoff>(damage_at + 1 + 8 + 8 + 8));
    const std::string garbage(8, static_cast<char>(0xFF));
    damaged.write(garbage.data(), static_cast<std::streamsize>(garbage.size()));
  }

  McapFile file = McapFile::open(path, /*should_loop=*/true);
  RecordedMessage message;
  std::uint64_t read = 0;
  while (read_message(file, message) && read < 3000) {
    ++read;
  }

  EXPECT_LT(read, 300U) << "the damaged chunk should have ended the pass early";
  EXPECT_FALSE(file.read_problem().empty());
  EXPECT_FALSE(read_message(file, message)) << "a failed read is final even when looping";
}

// With one topic on two tensor-envelope channels, the type is vouched for by the channel replay
// uses (the lowest id), not by whichever message the reader meets first.
TEST(McapFile, ValidatesATensorTopicFromTheChannelReplayUsesNotAnotherUnderTheSameTopic) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const auto path = scratch.path("two_channels.mcap");

  // A frame that really carries an ImageT, and one that does not say what it carries.
  const std::span<const std::byte> image_schema =
      holoscan::binary_schema<holoscan::schema::ImageT>();
  const std::vector<std::uint8_t> image_schema_bytes(
      reinterpret_cast<const std::uint8_t*>(image_schema.data()),
      reinterpret_cast<const std::uint8_t*>(image_schema.data()) + image_schema.size());
  const std::string image_type_name(
      holoscan::schema_package<holoscan::schema::ImageT>().identity().canonical_name);
  flatbuffers::FlatBufferBuilder valid;
  isaac::mcap_logger::FinishTensorFrameBuffer(
      valid, isaac::mcap_logger::CreateTensorFrameDirect(
                 valid, nullptr, nullptr, image_type_name.c_str(), &image_schema_bytes));
  flatbuffers::FlatBufferBuilder invalid;
  isaac::mcap_logger::FinishTensorFrameBuffer(invalid,
                                              isaac::mcap_logger::CreateTensorFrame(invalid));

  {
    mcap::McapWriter writer;
    ASSERT_TRUE(writer.open(path.string(), mcap::McapWriterOptions("")).ok());
    mcap::Schema schema(std::string(isaac::os::mcap_replayer::kTensorFrameSchemaName), "flatbuffer",
                        mcap::ByteArray{});
    writer.addSchema(schema);
    mcap::Channel selected("image", "flatbuffer", schema.id);
    mcap::Channel other("image", "flatbuffer", schema.id);
    writer.addChannel(selected);
    writer.addChannel(other);
    ASSERT_LT(selected.id, other.id);
    // The other channel's invalid message is earlier in time, so it is the one met first.
    const auto write = [&](mcap::ChannelId channel_id, std::uint64_t log_time,
                           const flatbuffers::FlatBufferBuilder& frame) {
      mcap::Message message{};
      message.channelId = channel_id;
      message.logTime = log_time;
      message.publishTime = log_time;
      message.data = reinterpret_cast<const std::byte*>(frame.GetBufferPointer());
      message.dataSize = frame.GetSize();
      ASSERT_TRUE(writer.write(message).ok());
    };
    write(other.id, 1, invalid);
    write(selected.id, 2, valid);
    writer.close();
  }

  McapFile file = McapFile::open(path);

  ASSERT_EQ(file.channel_of("image"), std::optional<mcap::ChannelId>(1));
  EXPECT_TRUE(file.verify_topic<holoscan::schema::ImageT>("image"))
      << "the replayed channel carries an ImageT, and the other channel's earlier message must "
         "not decide the answer";
}

TEST(McapFile, FindsTheChannelOfATopicAndNoneForAnUnknownTopic) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const auto path = write_fixture(directory);

  McapFile file = McapFile::open(path);

  const std::vector<mcap::ChannelId> ids = file.channel_ids();
  ASSERT_EQ(ids.size(), 1U);
  EXPECT_EQ(file.channel_of("pose"), ids.front());
  EXPECT_FALSE(file.channel_of("not_recorded").has_value());
}

// Two channels under one topic name (say, recorded under two schema versions) cannot both be
// replayed as one topic. The lowest channel id is used whatever order the reader lists them in,
// and the open says so for the others.
TEST(McapFile, ReplaysTheLowestChannelOfATopicRecordedOnSeveralAndSaysSo) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const auto path = directory / "shared_topic.mcap";

  mcap::McapWriter writer;
  const mcap::Status opened_for_writing = writer.open(path.string(), mcap::McapWriterOptions(""));
  ASSERT_TRUE(opened_for_writing.ok()) << opened_for_writing.message;
  mcap::Schema schema("isaac.PoseCpu", "flatbuffer", mcap::ByteArray{});
  writer.addSchema(schema);
  mcap::Channel first("pose", "flatbuffer", schema.id);
  mcap::Channel second("pose", "flatbuffer", schema.id);
  writer.addChannel(first);
  writer.addChannel(second);
  const std::array<std::byte, 4> payload{};
  for (const mcap::ChannelId channel_id : {first.id, second.id}) {
    mcap::Message message{};
    message.channelId = channel_id;
    message.logTime = 1;
    message.publishTime = 1;
    message.data = payload.data();
    message.dataSize = payload.size();
    ASSERT_TRUE(writer.write(message).ok());
  }
  writer.close();
  ASSERT_NE(first.id, second.id);

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  McapFile file = McapFile::open(path);
  const std::string output =
      testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();

  EXPECT_EQ(file.channel_of("pose"), std::min(first.id, second.id));
  EXPECT_NE(output.find("recorded on 2 channels"), std::string::npos) << output;
}

// A recording with no message indexes is a perfectly valid file that this reader cannot read in
// recorded order. The reader reports that by ending iteration, so without the failure being
// surfaced the file looks exactly like one holding no messages at all -- and the caller is told
// there is nothing to replay, which is untrue and points at the wrong thing to fix.
TEST(McapFile, ReportsWhyARecordingItCannotReadInRecordedOrderCannotBeOpened) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const auto path = directory / "no_index.mcap";

  mcap::McapWriterOptions options("");
  options.noChunking = true;
  options.noMessageIndex = true;
  mcap::McapWriter writer;
  const mcap::Status opened_for_writing = writer.open(path.string(), options);
  ASSERT_TRUE(opened_for_writing.ok()) << opened_for_writing.message;
  mcap::Schema schema("isaac.PoseCpu", "flatbuffer", mcap::ByteArray{});
  writer.addSchema(schema);
  mcap::Channel channel("pose", "flatbuffer", schema.id);
  writer.addChannel(channel);
  const std::array<std::byte, 4> payload{};
  mcap::Message message{};
  message.channelId = channel.id;
  message.logTime = 1;
  message.publishTime = 1;
  message.data = payload.data();
  message.dataSize = payload.size();
  const mcap::Status written = writer.write(message);
  ASSERT_TRUE(written.ok()) << written.message;
  writer.close();

  try {
    static_cast<void>(McapFile::open(path));
    FAIL() << "a recording whose messages cannot be read must not open as though it could";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("no_index.mcap"), std::string::npos) << error.what();
    EXPECT_NE(std::string(error.what()).find("index"), std::string::npos)
        << "the error must carry the reader's own reason rather than a guess: " << error.what();
  }
}

TEST(McapFile, PeekingATensorSchemaMidIterationDoesNotDisturbNext) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const auto path = write_tensor_fixture(directory);

  const std::unordered_map<mcap::ChannelId, std::uint64_t> ground_truth =
      read_channel_message_counts(path);
  ASSERT_EQ(ground_truth.size(), 2U);
  std::uint64_t total_per_pass = 0;
  for (const auto& [channel_id, count] : ground_truth) {
    total_per_pass += count;
  }
  ASSERT_GT(total_per_pass, 0U);

  McapFile file = McapFile::open(path, /*should_loop=*/true);

  std::unordered_map<mcap::ChannelId, std::uint64_t> counted;
  RecordedMessage message;

  // Advance the real iterator partway through the first pass before ever peeking, so the peek
  // below runs with next()'s own LinearMessageView already mid-iteration -- the case the deleted
  // peek-during-open() design never had to handle, since its peek always ran before that view
  // existed.
  constexpr std::uint64_t kAdvanceBeforePeek = 3;
  ASSERT_LT(kAdvanceBeforePeek, total_per_pass);
  for (std::uint64_t i = 0; i < kAdvanceBeforePeek; ++i) {
    ASSERT_TRUE(read_message(file, message));
    ++counted[message.channel_id];
  }

  // verify_topic() opens its own, separate LinearMessageView on the same reader to peek the
  // tensor channel's first message -- this must not perturb the position next() is already at.
  EXPECT_TRUE(file.verify_topic<holoscan::schema::ImageT>("image"));

  // Keep draining -- several passes past one file's worth -- and reconcile against the file's
  // own statistics exactly as in the wrap test, so a peek that had nudged next()'s position
  // would show up as a short or duplicated count here.
  constexpr std::uint64_t kPasses = 3;
  const std::uint64_t total_to_read = total_per_pass * kPasses;
  for (std::uint64_t i = kAdvanceBeforePeek; i < total_to_read; ++i) {
    ASSERT_TRUE(read_message(file, message)) << "next() returned false before draining "
                                             << total_to_read << " messages (at " << i << ")";
    ++counted[message.channel_id];
  }

  for (const auto& [channel_id, count] : ground_truth) {
    EXPECT_EQ(counted[channel_id], count * kPasses) << "channel " << channel_id;
  }
}

TEST(McapFile, ReportsAnErrorNamingThePathForAMissingFile) {
  try {
    static_cast<void>(McapFile::open("/nonexistent/none.mcap"));
    FAIL() << "expected a missing recording to be reported";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("none.mcap"), std::string::npos) << error.what();
  }
}

TEST(McapFile, TensorAndFlatTopicsBothVerifyAgainstTheirRealTypesInTheSameRecording) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const std::filesystem::path directory = scratch.path();
  const auto path = write_tensor_fixture(directory);

  McapFile file = McapFile::open(path);

  // The flat topic's schema is its own, unaffected by the tensor path.
  EXPECT_TRUE(file.verify_topic<isaac::PoseCpuT>("pose"));

  // The tensor channel's own recorded schema is the TensorFrame envelope, but verify_topic()
  // checks against the nested payload's real type instead -- the same "holoscan.schema.Image"
  // literal test_mcap_logger_tensor_schema_integration.cc asserts the writer emits -- because
  // that is the type a caller building a typed port actually needs to check against, not the
  // envelope every tensor stream shares.
  EXPECT_TRUE(file.verify_topic<holoscan::schema::ImageT>("image"));
}

namespace {

// Everything one next() call hands back, copied out because the views die on the next call.
struct Seen {
  mcap::ChannelId channel_id = 0;
  std::uint64_t log_time_ns = 0;
  std::string payload;
  std::string sidecar;
};

std::vector<Seen> read_all(McapFile& file) {
  std::vector<Seen> seen;
  RecordedMessage message;
  while (read_message(file, message)) {
    seen.push_back(Seen{
        message.channel_id, message.log_time_ns,
        std::string(reinterpret_cast<const char*>(message.payload.data()), message.payload.size()),
        std::string(reinterpret_cast<const char*>(message.sidecar.data()),
                    message.sidecar.size())});
  }
  return seen;
}

// The messages a non-looping read of a recording written from `messages` returns.
std::vector<Seen> read_recording(const std::vector<FixtureMessage>& messages) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  McapFile file = McapFile::open(write_paired_video_file(scratch.path(), messages));
  return read_all(file);
}

}  // namespace

TEST(McapFilePairing, AVideoTopicIsPairedAndIsTheTypeOfItsMetadata) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  McapFile file = McapFile::open(write_paired_video_file(
      scratch.path(), {{"video", 10, video_message_in("h264")}, {"video/metadata", 10, "S1"}}));

  EXPECT_TRUE(file.is_paired_video("video"));
  EXPECT_FALSE(file.is_paired_video("video/metadata"));
  EXPECT_FALSE(file.is_paired_video("other"));
  EXPECT_FALSE(file.is_paired_video("missing"));
  EXPECT_TRUE(file.verify_topic<holoscan::schema::ImageT>("video"));
  EXPECT_FALSE(file.verify_topic<isaac::PoseCpuT>("video"));
  EXPECT_TRUE(file.is_tensor_channel("video"));
}

TEST(McapFilePairing, ReportsTheCodecAVideoTopicWasRecordedInOnceItIsVerified) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  McapFile file = McapFile::open(
      write_paired_video_file(scratch.path(), {{"video", 10, video_message_in("h265")},
                                               {"video/metadata", 10, "S1"},
                                               {"video", 20, video_message_in("h264")},
                                               {"video/metadata", 20, "S2"}}));

  EXPECT_FALSE(file.video_codec_of("video")) << "the codec is read when the topic is verified";
  ASSERT_TRUE(file.verify_topic<holoscan::schema::ImageT>("video"));
  EXPECT_EQ(file.video_codec_of("video"), "h265") << "the first message names the codec";
  EXPECT_TRUE(file.verify_topic<holoscan::schema::ImageT>("video")) << "verifying again is fine";
  EXPECT_EQ(file.video_codec_of("video"), "h265");
}

TEST(McapFilePairing, ReadingTheCodecDoesNotMoveThePositionOfTheNextRead) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  McapFile file = McapFile::open(write_paired_video_file(
      scratch.path(), {{"video", 10, video_message_in("h264")}, {"video/metadata", 10, "S1"}}));

  ASSERT_TRUE(file.verify_topic<holoscan::schema::ImageT>("video"));

  RecordedMessage message;
  ASSERT_TRUE(read_message(file, message));
  EXPECT_EQ(message.log_time_ns, 10U);
}

TEST(McapFilePairing, HasNoCodecForATopicThatIsNotEncodedVideo) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  McapFile file = McapFile::open(write_paired_video_file(
      scratch.path(),
      {{"other", 5, "O1"}, {"video", 10, video_message_in("h264")}, {"video/metadata", 10, "S1"}}));
  ASSERT_TRUE(file.verify_topic<holoscan::schema::ImageT>("video"));

  EXPECT_FALSE(file.video_codec_of("video/metadata")) << "the metadata is not a video topic";
  EXPECT_FALSE(file.video_codec_of("other")) << "an unpaired topic is not encoded video";
  EXPECT_FALSE(file.video_codec_of("missing"));
}

// A video topic whose codec cannot be read cannot be decoded, so it is refused when it is
// verified, with the reason logged, rather than dropping every frame during replay.
TEST(McapFilePairing, RefusesAVideoTopicWhoseCodecCannotBeRead) {
  const auto verify_with_reason = [](const std::vector<FixtureMessage>& messages,
                                     std::string& output) {
    const isaac::os::mcap_logger::ScopedTempDirectory scratch;
    McapFile file = McapFile::open(write_paired_video_file(scratch.path(), messages));
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const bool verified = file.verify_topic<holoscan::schema::ImageT>("video");
    output = testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();
    EXPECT_FALSE(file.video_codec_of("video"));
    return verified;
  };
  std::string output;

  EXPECT_FALSE(verify_with_reason({{"video/metadata", 10, "S1"}}, output));
  EXPECT_NE(output.find("no recorded video messages"), std::string::npos) << output;

  EXPECT_FALSE(verify_with_reason(
      {{"video", 10, "not a protobuf message"}, {"video/metadata", 10, "S1"}}, output));
  EXPECT_NE(output.find("is damaged"), std::string::npos) << output;

  EXPECT_FALSE(verify_with_reason(
      {{"video", 10, video_message_in("")}, {"video/metadata", 10, "S1"}}, output));
  EXPECT_NE(output.find("does not say which codec"), std::string::npos) << output;
}

TEST(McapFilePairing, AVideoTopicWhoseMetadataSchemaBytesDifferIsNotTheType) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  McapFile file = McapFile::open(
      write_paired_video_file(scratch.path(), {{"video", 10, "V1"}, {"video/metadata", 10, "S1"}},
                              /*include_sidecar_channel=*/true, "not the image schema"));

  EXPECT_TRUE(file.is_paired_video("video"));
  EXPECT_FALSE(file.verify_topic<holoscan::schema::ImageT>("video"));
}

TEST(McapFilePairing, TheMetadataTopicOfAVideoIsNotAnImageTopicOfItsOwn) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  McapFile file = McapFile::open(
      write_paired_video_file(scratch.path(), {{"video", 10, "V1"}, {"video/metadata", 10, "S1"}}));

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const bool verified = file.verify_topic<holoscan::schema::ImageT>("video/metadata");
  const std::string output =
      testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();

  EXPECT_FALSE(verified);
  EXPECT_NE(output.find("metadata of encoded video topic 'video'; name 'video' instead"),
            std::string::npos)
      << output;
}

TEST(McapFilePairing, AVideoTopicWithoutItsMetadataChannelIsNotPaired) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  McapFile file = McapFile::open(write_paired_video_file(scratch.path(), {{"video", 10, "V1"}},
                                                         /*include_sidecar_channel=*/false));

  EXPECT_FALSE(file.is_paired_video("video"));
  EXPECT_FALSE(file.verify_topic<holoscan::schema::ImageT>("video"));
}

TEST(McapFilePairing, JoinsEachVideoMessageWithItsMetadataWhicheverArrivesFirst) {
  // The reader's order for equal log times is its own business, so the file puts the metadata
  // first for one pair and last for the other; either order must pair. A message on an unrelated
  // channel passes through untouched.
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  McapFile file =
      McapFile::open(write_paired_video_file(scratch.path(), {{"other", 5, "O1"},
                                                              {"video/metadata", 10, "S1"},
                                                              {"video", 10, "V1"},
                                                              {"video", 20, "V2"},
                                                              {"video/metadata", 20, "S2"}}));
  const std::vector<Seen> seen = read_all(file);

  ASSERT_EQ(seen.size(), 3U);
  EXPECT_EQ(seen[0].payload, "O1");
  EXPECT_TRUE(seen[0].sidecar.empty());
  EXPECT_EQ(seen[1].log_time_ns, 10U);
  EXPECT_EQ(seen[1].channel_id, file.channel_of("video"));
  EXPECT_EQ(seen[2].channel_id, file.channel_of("video"));
  EXPECT_EQ(seen[1].payload, "V1");
  EXPECT_EQ(seen[1].sidecar, "S1");
  EXPECT_EQ(seen[2].payload, "V2");
  EXPECT_EQ(seen[2].sidecar, "S2");
}

TEST(McapFilePairing, DropsAHalfWhosePartnerNeverArrivesAtTheSameLogTime) {
  // The partner is recorded at another time.
  std::vector<Seen> seen =
      read_recording({{"video", 10, "V1"}, {"video/metadata", 20, "S2"}, {"video", 20, "V2"}});
  ASSERT_EQ(seen.size(), 1U);
  EXPECT_EQ(seen[0].payload, "V2");
  EXPECT_EQ(seen[0].sidecar, "S2");

  // Two video messages in a row: the later one is kept.
  seen = read_recording({{"video", 10, "V1"}, {"video", 11, "V2"}, {"video/metadata", 11, "S2"}});
  ASSERT_EQ(seen.size(), 1U);
  EXPECT_EQ(seen[0].payload, "V2");

  // A video message left over at the end of the file.
  seen = read_recording({{"video", 10, "V1"}, {"video/metadata", 10, "S1"}, {"video", 20, "V2"}});
  ASSERT_EQ(seen.size(), 1U);
  EXPECT_EQ(seen[0].payload, "V1");
}

TEST(McapFilePairing, LoopingPairsAgainOnEveryPass) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  McapFile file = McapFile::open(
      write_paired_video_file(scratch.path(), {{"video", 10, "V1"}, {"video/metadata", 10, "S1"}}),
      /*should_loop=*/true);

  RecordedMessage message;
  for (int pass = 0; pass < 3; ++pass) {
    ASSERT_TRUE(read_message(file, message));
    EXPECT_EQ(
        std::string(reinterpret_cast<const char*>(message.payload.data()), message.payload.size()),
        "V1");
    EXPECT_EQ(message.sidecar.size(), 2U);
  }
}

// A looping file in which no pair ever completes ends the read: a pass that can return nothing
// would otherwise repeat forever. The first call hands control back at the end of the pass and the
// next one reports the cause.
TEST(McapFilePairing, ALoopingFileWithNoCompletePairEndsWithAReason) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  McapFile file = McapFile::open(write_paired_video_file(scratch.path(), {{"video", 10, "V1"}}),
                                 /*should_loop=*/true);

  RecordedMessage message;
  EXPECT_EQ(file.next(message), ReadResult::kNothingReady);
  EXPECT_EQ(file.next(message), ReadResult::kEnd);
  EXPECT_NE(file.read_problem().find("matching metadata message"), std::string::npos)
      << file.read_problem();
  EXPECT_EQ(file.next(message), ReadResult::kEnd);
}

TEST(McapFilePairing, ALoopingFileWithOneCompletePairAndAStrayHalfKeepsLooping) {
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  McapFile file = McapFile::open(
      write_paired_video_file(
          scratch.path(), {{"video", 10, "V1"}, {"video/metadata", 10, "S1"}, {"video", 20, "V2"}}),
      /*should_loop=*/true);

  RecordedMessage message;
  for (int pass = 0; pass < 4; ++pass) {
    ASSERT_TRUE(read_message(file, message));
    EXPECT_EQ(message.log_time_ns, 10U);
  }
  EXPECT_TRUE(file.read_problem().empty());
  EXPECT_EQ(file.dropped_halves(), 3U);
}

TEST(McapFilePairing, CountsTheHalvesDroppedWithoutAPartner) {
  const auto dropped = [](const std::vector<FixtureMessage>& messages) {
    const isaac::os::mcap_logger::ScopedTempDirectory scratch;
    McapFile file = McapFile::open(write_paired_video_file(scratch.path(), messages));
    read_all(file);
    return file.dropped_halves();
  };

  EXPECT_EQ(dropped({{"video", 10, "V1"}, {"video/metadata", 10, "S1"}}), 0U);
  EXPECT_EQ(dropped({{"video", 10, "V1"}, {"video/metadata", 20, "S2"}, {"video", 20, "V2"}}), 1U);
  EXPECT_EQ(dropped({{"video", 10, "V1"}, {"video", 11, "V2"}, {"video/metadata", 11, "S2"}}), 1U);
  EXPECT_EQ(dropped({{"video", 10, "V1"}, {"video/metadata", 10, "S1"}, {"video", 20, "V2"}}), 1U);
}
