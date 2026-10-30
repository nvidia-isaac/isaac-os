/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "common/mcap_replayer/decoder.h"

namespace isaac::os::mcap_replayer {

// The builder for a holoscan::schema::ImageT topic the logger recorded as encoded video: decodes
// the video half on `cuda_device` and publishes it as the same host image tensor a raw recording
// of the topic would have produced, under the sidecar's metadata bytes unchanged. The sidecar's
// `encoding` picks the pixel layout: RGB8 decodes to three channels, RGBA8 to four with alpha
// 255; anything else the logger cannot have recorded as video is rejected (dropped and counted). A
// frame size with an odd width or height is rejected too, since 4:2:0 video has no whole chroma
// sample for it. The sidecar is validated
// before any decoding, so a damaged one is rejected the same way with or without a video decoder.
//
// The returned builder owns one decoder, created on its first message, so it must only ever be
// called from one thread at a time (the prefetch thread does).
[[nodiscard]] MessageBuilder make_compressed_video_builder(std::uint32_t cuda_device);

// Why `cuda_device` cannot decode `codec` ("h264" or "h265", the codecs the logger records), or no
// value if it can. The reason names the cause: a build without the NVIDIA Video Codec SDK, or a GPU
// with no hardware decoder for the codec.
[[nodiscard]] std::optional<std::string> video_decode_unavailable_reason(std::uint32_t cuda_device,
                                                                         std::string_view codec);

}  // namespace isaac::os::mcap_replayer
