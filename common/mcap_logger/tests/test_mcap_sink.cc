/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "foxglove/channel.hpp"
#include "foxglove/context.hpp"
#include "foxglove/mcap.hpp"
#include "gtest/gtest.h"

#include "common/mcap_logger/mcap_sink.h"
#include "common/mcap_logger/tests/chunk_compressions.h"
#include "common/mcap_logger/tests/read_in_log_time_order.h"
#include "common/mcap_logger/tests/scoped_temp_directory.h"
#include "mcap/reader.hpp"

namespace isaac::os::mcap_logger {
namespace {

// Number of messages in a finished .mcap file, for the cross-sink isolation tests below.
int message_count(const std::filesystem::path& path) {
  mcap::McapReader reader;
  const auto status = reader.open(path.string());
  EXPECT_TRUE(status.ok()) << status.message;
  int count = 0;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    static_cast<void>(view);
    ++count;
  }
  return count;
}

TEST(McapSinkTest, WritesOneMessageThatCanBeReadBack) {
  ScopedTempDirectory temp_directory;
  const auto path = temp_directory.path("mcap_sink_test_writes_one_message.mcap");

  constexpr std::array<std::byte, 4> kSchemaBytes{std::byte{1}, std::byte{2}, std::byte{3},
                                                  std::byte{4}};
  constexpr std::array<std::byte, 3> kPayload{std::byte{9}, std::byte{8}, std::byte{7}};

  {
    McapSink sink(McapSinkOptions{.output_path = path});
    const bool ok =
        sink.write_flatbuffer_schema("joint_command", "isaac.JointCommandCpu", kSchemaBytes,
                                     /*log_time_ns=*/1000, kPayload);
    EXPECT_TRUE(ok);
  }  // McapSink's destructor must close/finalize the file.

  mcap::McapReader reader;
  const auto status = reader.open(path.string());
  ASSERT_TRUE(status.ok()) << status.message;
  ASSERT_TRUE(reader.header().has_value());
  EXPECT_EQ(reader.header()->profile, "isaac_mcap_logger");

  int message_count = 0;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    ++message_count;
    EXPECT_EQ(view.channel->topic, "joint_command");
    ASSERT_NE(view.schema, nullptr);
    EXPECT_EQ(view.schema->name, "isaac.JointCommandCpu");
    EXPECT_EQ(view.schema->encoding, "flatbuffer");
    // A plain write_flatbuffer_schema() carries one FlatBuffers message, so it keeps the
    // standard encoding that generic MCAP tooling knows how to decode.
    EXPECT_EQ(view.channel->messageEncoding, "flatbuffer");
    ASSERT_EQ(view.schema->data.size(), kSchemaBytes.size());
    for (std::size_t i = 0; i < kSchemaBytes.size(); ++i) {
      EXPECT_EQ(view.schema->data[i], kSchemaBytes[i]);
    }
    EXPECT_EQ(view.message.logTime, 1000U);
    ASSERT_EQ(view.message.dataSize, kPayload.size());
    for (std::size_t i = 0; i < kPayload.size(); ++i) {
      EXPECT_EQ(view.message.data[i], static_cast<std::byte>(kPayload[i]));
    }
  }
  EXPECT_EQ(message_count, 1);
}

// A payload that compresses well, so a chunk written with LZ4 or Zstd is measurably smaller than
// the same chunk written with none.
std::vector<ChunkCompression> WriteCompressiblePayloadAndReadChunks(
    const std::filesystem::path& path, McapSinkOptions options) {
  constexpr std::array<std::byte, 4> kSchemaBytes{std::byte{1}, std::byte{2}, std::byte{3},
                                                  std::byte{4}};
  const std::vector<std::byte> payload(64 * 1024, std::byte{0});
  options.output_path = path;
  {
    McapSink sink(options);
    EXPECT_TRUE(sink.write_flatbuffer_schema("frames", "isaac.Frame", kSchemaBytes,
                                             /*log_time_ns=*/1000, payload));
  }
  return ReadChunkCompressions(path);
}

TEST(McapSinkTest, WritesChunksWithTheRequestedCompression) {
  struct Case {
    foxglove::McapCompression compression;
    const char* declared;  // What each chunk of the file says about itself.
    bool shrinks;          // Whether a chunk of this payload gets smaller.
  };
  const std::array<Case, 3> cases{{
      {foxglove::McapCompression::None, "", false},
      {foxglove::McapCompression::Lz4, "lz4", true},
      {foxglove::McapCompression::Zstd, "zstd", true},
  }};
  for (const Case& test_case : cases) {
    SCOPED_TRACE(std::string("compression '") + test_case.declared + "'");
    ScopedTempDirectory temp_directory;
    const auto chunks = WriteCompressiblePayloadAndReadChunks(
        temp_directory.path("mcap_sink_test_compression.mcap"),
        McapSinkOptions{.compression = test_case.compression});
    ASSERT_FALSE(chunks.empty());
    for (const ChunkCompression& chunk : chunks) {
      EXPECT_EQ(chunk.algorithm, test_case.declared);
      if (test_case.shrinks) {
        EXPECT_LT(chunk.compressed_bytes, chunk.uncompressed_bytes);
      } else {
        EXPECT_EQ(chunk.compressed_bytes, chunk.uncompressed_bytes);
      }
    }
  }
}

TEST(McapSinkTest, CompressesChunksWithLz4UnlessToldOtherwise) {
  ScopedTempDirectory temp_directory;
  const auto chunks = WriteCompressiblePayloadAndReadChunks(
      temp_directory.path("mcap_sink_test_default_compression.mcap"), McapSinkOptions{});
  ASSERT_FALSE(chunks.empty());
  for (const ChunkCompression& chunk : chunks) {
    EXPECT_EQ(chunk.algorithm, "lz4");
  }
}

TEST(McapSinkTest, ReusesTheSameChannelForRepeatedUniqueId) {
  ScopedTempDirectory temp_directory;
  const auto path = temp_directory.path("mcap_sink_test_reuses_channel.mcap");

  constexpr std::array<std::byte, 4> kSchemaBytes{std::byte{1}, std::byte{2}, std::byte{3},
                                                  std::byte{4}};
  constexpr std::array<std::byte, 1> kPayload{std::byte{1}};

  {
    McapSink sink(McapSinkOptions{.output_path = path});
    EXPECT_TRUE(sink.write_flatbuffer_schema("joint_command", "isaac.JointCommandCpu", kSchemaBytes,
                                             1000, kPayload));
    EXPECT_TRUE(sink.write_flatbuffer_schema("joint_command", "isaac.JointCommandCpu", kSchemaBytes,
                                             2000, kPayload));
  }

  mcap::McapReader reader;
  const auto status = reader.open(path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  std::vector<mcap::ChannelId> channel_ids;
  std::vector<std::uint64_t> log_times;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    channel_ids.push_back(view.channel->id);
    log_times.push_back(view.message.logTime);
  }
  ASSERT_EQ(channel_ids.size(), 2U);
  EXPECT_EQ(channel_ids[0], channel_ids[1]);  // Same channel reused, not recreated.
  EXPECT_EQ(log_times[0], 1000U);
  EXPECT_EQ(log_times[1], 2000U);
}

TEST(McapSinkTest, DistinctUniqueIdsGetDistinctChannels) {
  ScopedTempDirectory temp_directory;
  const auto path = temp_directory.path("mcap_sink_test_distinct_channels.mcap");

  constexpr std::array<std::byte, 4> kSchemaBytes{std::byte{1}, std::byte{2}, std::byte{3},
                                                  std::byte{4}};
  constexpr std::array<std::byte, 1> kPayload{std::byte{1}};

  {
    McapSink sink(McapSinkOptions{.output_path = path});
    EXPECT_TRUE(sink.write_flatbuffer_schema("joint_command", "isaac.JointCommandCpu", kSchemaBytes,
                                             1000, kPayload));
    EXPECT_TRUE(sink.write_flatbuffer_schema("joint_state", "isaac.JointStateCpu", kSchemaBytes,
                                             1000, kPayload));
  }

  mcap::McapReader reader;
  const auto status = reader.open(path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  std::vector<mcap::ChannelId> channel_ids;
  std::vector<std::string> topics;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    channel_ids.push_back(view.channel->id);
    topics.push_back(view.channel->topic);
  }
  ASSERT_EQ(channel_ids.size(), 2U);
  EXPECT_NE(channel_ids[0], channel_ids[1]);  // Distinct unique_ids get distinct channels.
  ASSERT_EQ(topics.size(), 2U);
  EXPECT_NE(topics[0], topics[1]);
}

TEST(McapSinkTest, ConstructorThrowsWhenTheOutputDirectoryCannotBeCreated) {
  // A regular file sitting where a parent directory needs to be created makes
  // create_directories() fail deterministically and portably (no reliance on running as a
  // non-root user or on a read-only filesystem being available in every test environment).
  ScopedTempDirectory temp_directory;
  const auto blocker_path = temp_directory.path("blocker");
  {
    std::ofstream blocker(blocker_path);
    ASSERT_TRUE(blocker.good());
  }

  const auto output_path = blocker_path / "subdir" / "out.mcap";
  EXPECT_THROW(McapSink sink(McapSinkOptions{.output_path = output_path}), std::runtime_error);
}

TEST(McapSinkTest, ConstructorThrowsWhenNeitherFileNorStreamIsEnabled) {
  // output_path is nullopt and enable_stream defaults to false -- no usable sink at all.
  EXPECT_THROW(McapSink sink(McapSinkOptions{}), std::runtime_error);
}

TEST(McapSinkTest, StreamOnlyModeWritesNoFileButStillLogsSuccessfully) {
  ScopedTempDirectory temp_directory;
  const auto unused_path = temp_directory.path("should_not_be_created.mcap");

  McapSinkOptions options;
  options.enable_stream = true;
  options.websocket_port = 0;  // Let the OS assign a free ephemeral port.

  constexpr std::array<std::byte, 4> kSchemaBytes{std::byte{1}, std::byte{2}, std::byte{3},
                                                  std::byte{4}};
  constexpr std::array<std::byte, 3> kPayload{std::byte{9}, std::byte{8}, std::byte{7}};

  McapSink sink(options);
  EXPECT_TRUE(sink.write_flatbuffer_schema("joint_command", "isaac.JointCommandCpu", kSchemaBytes,
                                           1000, kPayload));
  ASSERT_TRUE(sink.websocket_port().has_value());
  EXPECT_NE(*sink.websocket_port(), 0U);               // The OS assigned a real, nonzero port.
  EXPECT_FALSE(std::filesystem::exists(unused_path));  // No output_path was given -- no file.
}

TEST(McapSinkTest, ConstructorThrowsWhenTheWebsocketPortIsAlreadyBound) {
  ScopedTempDirectory temp_directory;

  McapSinkOptions first_options;
  first_options.output_path = temp_directory.path("port_conflict_first.mcap");
  first_options.enable_stream = true;
  first_options.websocket_port = 0;  // Let the OS assign a free ephemeral port.
  McapSink first_sink(first_options);
  ASSERT_TRUE(first_sink.websocket_port().has_value());

  McapSinkOptions second_options;
  second_options.output_path = temp_directory.path("port_conflict_second.mcap");
  second_options.enable_stream = true;
  second_options.websocket_port = *first_sink.websocket_port();  // Already bound by first_sink.
  EXPECT_THROW(McapSink second_sink(second_options), std::runtime_error);
}

// Characterizes the Foxglove SDK behavior that McapSink's per-instance context exists to avoid:
// a default-constructed foxglove::Context is the SDK's shared global one, and every sink attached
// to a context receives every channel logged on it. Built from raw SDK primitives rather than
// McapSink, since McapSink deliberately never uses the default context -- if this test ever goes
// red, the premise behind that choice has changed and McapSink's own context comment is stale.
TEST(McapSinkTest, DefaultFoxgloveContextFansOneChannelOutToEveryWriterOnIt) {
  ScopedTempDirectory temp_directory;
  const auto path_a = temp_directory.path("shared_context_a.mcap");
  const auto path_b = temp_directory.path("shared_context_b.mcap");

  constexpr std::array<std::byte, 4> kSchemaBytes{std::byte{1}, std::byte{2}, std::byte{3},
                                                  std::byte{4}};
  constexpr std::array<std::byte, 1> kPayload{std::byte{1}};

  {
    const foxglove::Context default_context;  // Default-constructed: the SDK's global context.

    // McapWriterOptions::path is non-owning, so both strings must outlive their create() call.
    const std::string path_a_str = path_a.string();
    const std::string path_b_str = path_b.string();

    foxglove::McapWriterOptions options_a;
    options_a.context = default_context;
    options_a.path = path_a_str;
    auto writer_a = foxglove::McapWriter::create(options_a);
    ASSERT_TRUE(writer_a.has_value());

    foxglove::McapWriterOptions options_b;
    options_b.context = default_context;
    options_b.path = path_b_str;
    auto writer_b = foxglove::McapWriter::create(options_b);
    ASSERT_TRUE(writer_b.has_value());

    foxglove::Schema schema;
    schema.name = "isaac.JointCommandCpu";
    schema.encoding = McapSink::kFlatBufferMessageEncoding;
    schema.data = kSchemaBytes.data();
    schema.data_len = kSchemaBytes.size();

    // One channel, logged exactly once, on the same default context both writers are attached to.
    auto channel = foxglove::RawChannel::create(
        "joint_command", McapSink::kFlatBufferMessageEncoding, schema, default_context);
    ASSERT_TRUE(channel.has_value());
    ASSERT_EQ(channel->log(kPayload.data(), kPayload.size(), 1000), foxglove::FoxgloveError::Ok);

    EXPECT_EQ(writer_a->close(), foxglove::FoxgloveError::Ok);
    EXPECT_EQ(writer_b->close(), foxglove::FoxgloveError::Ok);
  }

  // The single log() call landed in both files -- exactly the cross-talk McapSink must not have.
  EXPECT_EQ(message_count(path_a), 1);
  EXPECT_EQ(message_count(path_b), 1);
}

// The guard for the behavior characterized above: two live McapSink instances must not see each
// other's channels, which holds only because each owns a separately created context. Reverting
// McapSink's context_ to a default-constructed one turns this test red.
TEST(McapSinkTest, SeparateSinksDoNotShareAContext) {
  ScopedTempDirectory temp_directory;
  const auto path_a = temp_directory.path("sink_isolation_a.mcap");
  const auto path_b = temp_directory.path("sink_isolation_b.mcap");

  constexpr std::array<std::byte, 4> kSchemaBytes{std::byte{1}, std::byte{2}, std::byte{3},
                                                  std::byte{4}};
  constexpr std::array<std::byte, 1> kPayload{std::byte{1}};

  {
    McapSink sink_a(McapSinkOptions{.output_path = path_a});
    McapSink sink_b(McapSinkOptions{.output_path = path_b});
    // Written through sink_a only, while sink_b is alive alongside it.
    EXPECT_TRUE(sink_a.write_flatbuffer_schema("joint_command", "isaac.JointCommandCpu",
                                               kSchemaBytes, 1000, kPayload));
  }

  EXPECT_EQ(message_count(path_a), 1);
  EXPECT_EQ(message_count(path_b), 0);
}

TEST(McapSinkTest, WritesCompressedVideoUnderItsOwnSchemaAndEncoding) {
  ScopedTempDirectory temp_directory;
  const auto path = temp_directory.path("mcap_sink_test_compressed_video.mcap");
  const std::vector<std::byte> kEncoded{std::byte{1}, std::byte{2}, std::byte{3}};

  {
    McapSink sink(McapSinkOptions{.output_path = path});
    // Deliberately unlike the topic name, so the frame_id assertion below can tell the two
    // apart: writing the topic name into that field would be invisible if they matched.
    ASSERT_TRUE(sink.write_compressed_video("camera", "optical_frame", 1000U, kEncoded, "h264"));
    ASSERT_TRUE(sink.write_compressed_video("camera", "optical_frame", 2000U, kEncoded, "h264"));
  }

  mcap::McapReader reader;
  const auto status = reader.open(path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  std::vector<mcap::ChannelId> channel_ids;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    // The two markers a replayer branches on, both channel-level so they are readable before
    // any message is parsed.
    EXPECT_EQ(view.channel->messageEncoding, "protobuf");
    ASSERT_NE(view.schema, nullptr);
    EXPECT_EQ(view.schema->name, "foxglove.CompressedVideo");
    EXPECT_EQ(view.schema->encoding, "protobuf");

    // foxglove.CompressedVideo.frame_id carries the coordinate frame the camera declared, which
    // is not the topic it is recorded on -- a reader resolving this video against a transform
    // needs the former. Checked by scanning the encoded message rather than by linking a
    // protobuf reader into this test: the field is a bare length-prefixed string, and the two
    // candidate values share no bytes, so their presence and absence is unambiguous.
    const std::string_view payload(reinterpret_cast<const char*>(view.message.data),
                                   view.message.dataSize);
    EXPECT_NE(payload.find("optical_frame"), std::string_view::npos)
        << "the source image's frame_id is missing from the message";
    EXPECT_EQ(payload.find("camera"), std::string_view::npos)
        << "the topic name was written into frame_id instead of the source image's";

    channel_ids.push_back(view.channel->id);
  }
  ASSERT_EQ(channel_ids.size(), 2U);
  EXPECT_EQ(channel_ids[0], channel_ids[1]);  // Channel reused, not recreated per frame.
}

TEST(McapSinkTest, WriteFlatbufferSchemaFailsOnATopicAlreadyWritingCompressedVideo) {
  ScopedTempDirectory temp_directory;
  const auto path = temp_directory.path("mcap_sink_test_mixed_schema_video_first.mcap");

  const std::vector<std::byte> kEncoded{std::byte{1}, std::byte{2}, std::byte{3}};
  constexpr std::array<std::byte, 4> kSchemaBytes{std::byte{1}, std::byte{2}, std::byte{3},
                                                  std::byte{4}};
  constexpr std::array<std::byte, 1> kPayload{std::byte{1}};

  McapSink sink(McapSinkOptions{.output_path = path});
  ASSERT_TRUE(sink.write_compressed_video("camera", "optical_frame", 1000U, kEncoded, "h264"));
  // An MCAP channel carries exactly one schema for its whole life -- a topic that started as
  // CompressedVideo can't also become a flatbuffer channel.
  EXPECT_FALSE(sink.write_flatbuffer_schema("camera", "isaac.JointCommandCpu", kSchemaBytes, 2000,
                                            kPayload));
}

TEST(McapSinkTest, WriteCompressedVideoFailsOnATopicAlreadyWritingFlatbufferSchema) {
  ScopedTempDirectory temp_directory;
  const auto path = temp_directory.path("mcap_sink_test_mixed_schema_flatbuffer_first.mcap");

  constexpr std::array<std::byte, 4> kSchemaBytes{std::byte{1}, std::byte{2}, std::byte{3},
                                                  std::byte{4}};
  constexpr std::array<std::byte, 1> kPayload{std::byte{1}};
  const std::vector<std::byte> kEncoded{std::byte{1}, std::byte{2}, std::byte{3}};

  McapSink sink(McapSinkOptions{.output_path = path});
  ASSERT_TRUE(sink.write_flatbuffer_schema("camera", "isaac.JointCommandCpu", kSchemaBytes, 1000,
                                           kPayload));
  // The mirrored direction: a topic that started as flatbuffer can't also become CompressedVideo.
  EXPECT_FALSE(sink.write_compressed_video("camera", "optical_frame", 2000U, kEncoded, "h264"));
}

}  // namespace
}  // namespace isaac::os::mcap_logger
