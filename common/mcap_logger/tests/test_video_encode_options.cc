/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>

#include "gtest/gtest.h"

#include "common/mcap_logger/video_encode_options.h"

namespace isaac::os::mcap_logger {
namespace {

// The two spellings are for different consumers -- the recorded file's readers and NVENC's own
// codec table -- and mixing them up yields either a file Foxglove Studio refuses to render or an
// encoder that cannot start, so both are pinned here.
TEST(FoxgloveCodecFormat, IsLowercaseForEveryCodec) {
  EXPECT_EQ(FoxgloveCodecFormat(VideoCodec::kH264), "h264");
  EXPECT_EQ(FoxgloveCodecFormat(VideoCodec::kH265), "h265");
}

// Keys of NVVideoEncoder's CODEC_GUIDS. HEVC and H.265 are one codec, and "H265" is the spelling
// that map carries for it.
TEST(NvencCodecName, IsUppercaseForEveryCodec) {
  EXPECT_EQ(NvencCodecName(VideoCodec::kH264), "H264");
  EXPECT_EQ(NvencCodecName(VideoCodec::kH265), "H265");
}

TEST(ValidateVideoEncodeOptions, AcceptsTheDefaults) {
  EXPECT_NO_THROW(ValidateVideoEncodeOptions(VideoEncodeOptions{}));
}

TEST(ValidateVideoEncodeOptions, RejectsZeroedRateFields) {
  EXPECT_THROW(ValidateVideoEncodeOptions(VideoEncodeOptions{.bitrate = 0}), std::invalid_argument);
  EXPECT_THROW(ValidateVideoEncodeOptions(VideoEncodeOptions{.frame_rate = 0}),
               std::invalid_argument);
  EXPECT_THROW(ValidateVideoEncodeOptions(VideoEncodeOptions{.bitrate = 0, .frame_rate = 0}),
               std::invalid_argument);
}

}  // namespace
}  // namespace isaac::os::mcap_logger
