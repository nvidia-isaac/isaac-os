/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace isaac::os::mcap_logger {

// The two fields of a foxglove.CompressedVideo message a reader needs, as views into the bytes it
// was parsed from.
struct CompressedVideoFields {
  std::span<const std::uint8_t> data;  // field 3: the Annex-B bitstream
  std::string_view format;             // field 4: "h264" or "h265"
};

namespace detail {

// Reads one base-128 varint at `position` and advances past it. Nullopt if the buffer ends first
// or the varint does not fit in 64 bits.
[[nodiscard]] inline std::optional<std::uint64_t> ReadVarint(std::span<const std::uint8_t> bytes,
                                                             std::size_t& position) {
  std::uint64_t value = 0;
  for (int shift = 0; shift <= 63; shift += 7) {
    if (position >= bytes.size()) {
      return std::nullopt;
    }
    const std::uint8_t byte = bytes[position++];
    if (shift == 63 && (byte & 0x7F) > 1) {
      return std::nullopt;  // only bit 0 of the tenth byte fits in 64 bits
    }
    value |= static_cast<std::uint64_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) {
      return value;
    }
  }
  return std::nullopt;
}

// Reads the length-delimited field `field` at `position` and keeps it if it is one the message
// reader wants. Returns false if the length is cut off or overruns the buffer.
[[nodiscard]] inline bool ReadLengthDelimited(std::span<const std::uint8_t> bytes,
                                              std::size_t& position, std::uint64_t field,
                                              CompressedVideoFields& fields) {
  const std::optional<std::uint64_t> length = ReadVarint(bytes, position);
  if (!length.has_value() || *length > bytes.size() - position) {
    return false;
  }
  const auto content = bytes.subspan(position, static_cast<std::size_t>(*length));
  if (field == 3) {
    fields.data = content;
  } else if (field == 4) {
    fields.format = std::string_view(reinterpret_cast<const char*>(content.data()), content.size());
  }
  position += static_cast<std::size_t>(*length);
  return true;
}

// Reads the value that follows a key with the given wire type, keeping or skipping it. Returns
// false for a truncated value or a wire type this message never uses.
[[nodiscard]] inline bool ReadFieldValue(std::span<const std::uint8_t> bytes, std::size_t& position,
                                         std::uint64_t field, std::uint64_t wire_type,
                                         CompressedVideoFields& fields) {
  switch (wire_type) {
    case 0:  // varint
      return ReadVarint(bytes, position).has_value();
    case 1:    // fixed64
    case 5: {  // fixed32
      const std::size_t width = wire_type == 1 ? 8 : 4;
      if (bytes.size() - position < width) {
        return false;
      }
      position += width;
      return true;
    }
    case 2:  // length-delimited
      return ReadLengthDelimited(bytes, position, field, fields);
    default:
      return false;
  }
}

}  // namespace detail

// Reads foxglove.CompressedVideo (field 1 timestamp, 2 frame_id, 3 data, 4 format). The Foxglove
// C++ SDK serializes this message but cannot parse it, so this is the one reader the logger's
// own tests and the replayer share. Fields it does not read are skipped; a field a message
// repeats keeps its last value, as protobuf specifies. Returns nullopt for bytes that are not a
// well-formed message: a truncated field, a length that overruns the buffer, a field number of
// zero, or a wire type this message never uses.
[[nodiscard]] inline std::optional<CompressedVideoFields> ReadCompressedVideo(
    std::span<const std::uint8_t> bytes) {
  CompressedVideoFields fields;
  std::size_t position = 0;
  while (position < bytes.size()) {
    const std::optional<std::uint64_t> key = detail::ReadVarint(bytes, position);
    if (!key.has_value()) {
      return std::nullopt;
    }
    const std::uint64_t field = *key >> 3;
    if (field == 0) {  // protobuf field numbers start at 1
      return std::nullopt;
    }
    if (!detail::ReadFieldValue(bytes, position, field, *key & 0x7U, fields)) {
      return std::nullopt;
    }
  }
  return fields;
}

}  // namespace isaac::os::mcap_logger
