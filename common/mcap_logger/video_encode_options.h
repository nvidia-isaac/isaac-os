/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace isaac::os::mcap_logger {

// The video codecs this repo can both encode and read back: NVVideoEncoder's CODEC_GUIDS and
// NVVideoDecoder's CODEC_IDS cover exactly H.264 and HEVC. HEVC and H.265 are two names for one
// codec, so they share a single enumerator. Naming the codec as a type rather than a string means
// a codec neither side supports cannot be spelled at all.
enum class VideoCodec : std::uint8_t { kH264, kH265 };

// The lowercase spelling foxglove.CompressedVideo's `format` field expects -- writing "H264"
// verbatim yields a file Foxglove Studio may refuse to render. Distinct from NvencCodecName():
// this spelling is for the recorded file's readers, not for the encoder.
[[nodiscard]] constexpr std::string_view FoxgloveCodecFormat(VideoCodec codec) {
  switch (codec) {
    case VideoCodec::kH264:
      return "h264";
    case VideoCodec::kH265:
      return "h265";
  }
  // Only reachable for a value cast in from outside the enumerators above. Throwing rather than
  // falling back to H.264 keeps a codec that was added to the enum but not to this mapping from
  // recording a file whose declared format disagrees with its own bitstream, which nothing would
  // notice until someone tried to play it back.
  throw std::logic_error(
      "Unable to record video: the requested video codec is not one this build "
      "can encode.");
}

// The uppercase spelling NVVideoEncoder's CODEC_GUIDS map keys on. Distinct from
// FoxgloveCodecFormat(): this spelling is for the encoder, not for the recorded file's readers.
// Returned by value because NVVideoEncoder takes its codec as a const std::string&.
[[nodiscard]] inline std::string NvencCodecName(VideoCodec codec) {
  switch (codec) {
    case VideoCodec::kH264:
      return "H264";
    case VideoCodec::kH265:
      return "H265";
  }
  // Only reachable for a value cast in from outside the enumerators above; see
  // FoxgloveCodecFormat() for why this refuses rather than falling back to H.264.
  throw std::logic_error(
      "Unable to record video: the requested video codec is not one this build "
      "can encode.");
}

// The companion channel an encoded topic records its holoscan.schema.Image metadata on.
// foxglove.CompressedVideo has room for only the timestamp, frame id, codec and payload, so an
// encoded topic writes two channels instead of one. Both the code that writes that channel and
// the attach-time check that keeps any other topic from claiming its name derive the name here,
// so the two cannot drift apart.
[[nodiscard]] inline std::string VideoMetadataChannelName(std::string_view topic) {
  return std::string(topic) + "/metadata";
}

// Per-topic NVENC settings. No width/height: they come from the first frame's own
// holoscan.schema.Image metadata, so they cannot drift from the DescriptorTensorTopicContract
// that already fixes the tensor's bounds.
struct VideoEncodeOptions {
  VideoCodec codec = VideoCodec::kH264;
  std::uint32_t bitrate = 10'000'000;
  std::uint32_t frame_rate = 30;
  std::uint32_t cuda_device = 0;
};

// Throws std::invalid_argument for a configuration that would record unreadable video. Called
// during attach(), so a bad rate fails at compose time rather than after a long run has already
// written a useless file. The codec needs no check: VideoCodec has no unsupported value.
//
// The source ImageEncoding is deliberately NOT checked here: ImageTensorFieldContract consumes
// `encoding` only to size max_tensor_bytes and then discards it, and
// DescriptorTensorTopicContract<T> stores none, so no encoding is in scope at attach() time.
// FrameEncoder checks it against the first frame's own holoscan.schema.Image metadata instead.
inline void ValidateVideoEncodeOptions(const VideoEncodeOptions& options) {
  if (options.bitrate == 0 || options.frame_rate == 0) {
    throw std::invalid_argument("Video bitrate and frame rate must both be greater than zero.");
  }
}

}  // namespace isaac::os::mcap_logger
