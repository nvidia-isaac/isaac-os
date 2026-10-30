/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

#include "common/mcap_logger/compressed_video_proto.h"

namespace isaac::os::mcap_logger {
namespace {

using Bytes = std::vector<std::uint8_t>;

void append_varint(Bytes& out, std::uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<std::uint8_t>(value | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<std::uint8_t>(value));
}

void append_key(Bytes& out, std::uint64_t field, std::uint64_t wire_type) {
  append_varint(out, (field << 3) | wire_type);
}

void append_length_delimited(Bytes& out, std::uint64_t field, const Bytes& content) {
  append_key(out, field, 2);
  append_varint(out, content.size());
  out.insert(out.end(), content.begin(), content.end());
}

Bytes text(std::string_view value) { return Bytes(value.begin(), value.end()); }

TEST(ReadCompressedVideo, ReadsDataAndFormatAndSkipsTheRest) {
  Bytes message;
  Bytes timestamp;  // A google.protobuf.Timestamp of 5 seconds and 7 nanoseconds.
  append_key(timestamp, 1, 0);
  append_varint(timestamp, 5);
  append_key(timestamp, 2, 0);
  append_varint(timestamp, 7);
  append_length_delimited(message, 1, timestamp);
  append_length_delimited(message, 2, text("camera"));
  append_length_delimited(message, 3, Bytes{0, 0, 0, 1, 0x65});
  append_length_delimited(message, 4, text("h264"));

  const auto parsed = ReadCompressedVideo(message);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(Bytes(parsed->data.begin(), parsed->data.end()), (Bytes{0, 0, 0, 1, 0x65}));
  EXPECT_EQ(parsed->format, "h264");
}

TEST(ReadCompressedVideo, SkipsUnknownVarintAndFixedWidthFields) {
  Bytes message;
  append_key(message, 9, 0);
  append_varint(message, 300);
  append_key(message, 10, 1);
  message.insert(message.end(), 8, 0xAA);
  append_key(message, 11, 5);
  message.insert(message.end(), 4, 0xBB);
  append_length_delimited(message, 4, text("h265"));

  const auto parsed = ReadCompressedVideo(message);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->format, "h265");
  EXPECT_TRUE(parsed->data.empty());
}

TEST(ReadCompressedVideo, AnEmptyMessageHasEmptyFields) {
  const auto parsed = ReadCompressedVideo({});
  ASSERT_TRUE(parsed.has_value());
  EXPECT_TRUE(parsed->data.empty());
  EXPECT_TRUE(parsed->format.empty());
}

TEST(ReadCompressedVideo, RejectsMalformedMessages) {
  Bytes truncated_length;  // claims 10 bytes, holds 3
  append_key(truncated_length, 3, 2);
  append_varint(truncated_length, 10);
  truncated_length.insert(truncated_length.end(), 3, 0x01);

  Bytes length_too_large;
  append_key(length_too_large, 3, 2);
  append_varint(length_too_large, UINT64_MAX);

  Bytes truncated_fixed32;
  append_key(truncated_fixed32, 11, 5);
  truncated_fixed32.insert(truncated_fixed32.end(), 3, 0);

  Bytes truncated_fixed64;
  append_key(truncated_fixed64, 10, 1);
  truncated_fixed64.insert(truncated_fixed64.end(), 7, 0);

  Bytes unsupported_wire_type;  // start-group: not used by this message
  append_key(unsupported_wire_type, 3, 3);

  const std::vector<Bytes> malformed = {truncated_length,
                                        length_too_large,
                                        Bytes{0x80} /* truncated key */,
                                        truncated_fixed32,
                                        truncated_fixed64,
                                        unsupported_wire_type,
                                        Bytes{0x00, 0x01} /* field 0, varint */,
                                        Bytes{0x02, 0x00} /* field 0, length-delimited */,
                                        Bytes{0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                                              0x02} /* varint overflowing 64 bits */};
  for (const Bytes& message : malformed) {
    EXPECT_FALSE(ReadCompressedVideo(message).has_value());
  }
}

}  // namespace
}  // namespace isaac::os::mcap_logger
