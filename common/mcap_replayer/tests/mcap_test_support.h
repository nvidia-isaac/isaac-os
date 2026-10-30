/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "holoscan/schema/binary_schema.hpp"
#include "holoscan/schema/schema_package.hpp"

#include "common/mcap_replayer/mcap_file.h"
#include "mcap/writer.hpp"

namespace isaac::os::mcap_replayer::test_support {

struct FixtureMessage {
  std::string topic;  // "video", "video/metadata" or "other"
  std::uint64_t log_time_ns = 0;
  std::string payload;
};

// A foxglove.CompressedVideo message holding only its format field (number 4), which is all
// McapFile::video_codec_of() and the replayer's codec check read.
inline std::string video_message_in(std::string_view format) {
  std::string message;
  message.push_back(static_cast<char>((4U << 3U) | 2U));
  message.push_back(static_cast<char>(format.size()));
  message.append(format);
  return message;
}

// Writes a small recording with the channels a logger-encoded topic leaves behind: "video"
// (foxglove.CompressedVideo, protobuf) and "video/metadata" (holoscan.schema.Image, flatbuffer),
// plus an unrelated "other" channel. Payload bytes are opaque to McapFile, so tests pass short
// strings. include_sidecar_channel = false leaves the metadata channel out of the file.
// sidecar_schema_data replaces the metadata channel's recorded binary schema bytes.
inline std::filesystem::path write_paired_video_file(
    const std::filesystem::path& directory, const std::vector<FixtureMessage>& messages,
    bool include_sidecar_channel = true,
    std::optional<std::string> sidecar_schema_data = std::nullopt) {
  const std::filesystem::path path = directory / "paired.mcap";
  mcap::McapWriter writer;
  const mcap::Status opened = writer.open(path.string(), mcap::McapWriterOptions(""));
  EXPECT_TRUE(opened.ok()) << opened.message;

  mcap::Schema video_schema("foxglove.CompressedVideo", "protobuf", "");
  writer.addSchema(video_schema);
  // The metadata channel's schema carries the real binary schema, as the logger records it.
  const auto& image_binary_schema = holoscan::binary_schema<holoscan::schema::Image>();
  mcap::Schema image_schema(
      "holoscan.schema.Image", "flatbuffer",
      sidecar_schema_data.value_or(std::string(
          reinterpret_cast<const char*>(image_binary_schema.data()), image_binary_schema.size())));
  writer.addSchema(image_schema);
  mcap::Schema other_schema("other.Thing", "jsonschema", "{}");
  writer.addSchema(other_schema);

  std::unordered_map<std::string, mcap::ChannelId> channels;
  mcap::Channel video("video", "protobuf", video_schema.id);
  writer.addChannel(video);
  channels["video"] = video.id;
  if (include_sidecar_channel) {
    mcap::Channel sidecar("video/metadata", "flatbuffer", image_schema.id);
    writer.addChannel(sidecar);
    channels["video/metadata"] = sidecar.id;
  }
  mcap::Channel other("other", "json", other_schema.id);
  writer.addChannel(other);
  channels["other"] = other.id;

  std::uint32_t sequence = 0;
  for (const FixtureMessage& fixture : messages) {
    mcap::Message message;
    message.channelId = channels.at(fixture.topic);
    message.sequence = sequence++;
    message.logTime = fixture.log_time_ns;
    message.publishTime = fixture.log_time_ns;
    message.data = reinterpret_cast<const std::byte*>(fixture.payload.data());
    message.dataSize = fixture.payload.size();
    const mcap::Status written = writer.write(message);
    EXPECT_TRUE(written.ok()) << written.message;
  }
  writer.close();
  return path;
}

// next() as a bool, for tests that only want the messages of a recording: skips the "nothing
// ready" results a paired topic produces between messages, and is false only at the end.
inline bool read_message(McapFile& file, RecordedMessage& message) {
  using enum ReadResult;
  while (true) {
    switch (file.next(message)) {
      case kMessage:
        return true;
      case kEnd:
        return false;
      case kNothingReady:
        break;
    }
  }
}

}  // namespace isaac::os::mcap_replayer::test_support
