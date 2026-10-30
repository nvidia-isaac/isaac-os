/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "holoscan/sensor_io/schemas/image_generated.h"
#include <flatbuffers/flatbuffers.h>

#include "common/image_schema/image_schema.h"
#include "common/mcap_logger/tests/scoped_temp_directory.h"
#include "common/mcap_replayer/compressed_video_builder.h"
#include "common/mcap_replayer/mcap_replayer.h"
#include "common/mcap_replayer/tests/mcap_test_support.h"

namespace isaac::os::mcap_replayer {
namespace {

std::vector<std::byte> image_metadata(std::int32_t width, std::int32_t height,
                                      holoscan::schema::ImageEncoding encoding) {
  flatbuffers::FlatBufferBuilder builder;
  const auto header =
      holoscan::schema::CreateHeader(builder, 0, builder.CreateString("camera"), 0, 0);
  holoscan::schema::FinishImageBuffer(
      builder, holoscan::schema::CreateImage(builder, header, height, width, encoding));
  const auto* data = reinterpret_cast<const std::byte*>(builder.GetBufferPointer());
  return std::vector<std::byte>(data, data + builder.GetSize());
}

// The video bytes are never reached by these tests: the sidecar is validated before any decoding,
// so they need neither a GPU nor a real bitstream.
const std::vector<std::byte> kNotVideo(8, std::byte{0});

std::string error_of(const std::vector<std::byte>& sidecar) {
  const MessageBuilder build = make_compressed_video_builder(/*cuda_device=*/0);
  const auto message = build(kNotVideo, sidecar, 1234);
  EXPECT_FALSE(message.has_value());
  return message ? std::string{} : message.error();
}

TEST(CompressedVideoBuilder, RejectsASidecarThatIsNotAnImage) {
  EXPECT_NE(error_of(std::vector<std::byte>(16, std::byte{0x7F})).find("metadata"),
            std::string::npos);
}

TEST(CompressedVideoBuilder, RejectsAnEncodingTheLoggerCannotHaveRecordedAsVideo) {
  const std::string error =
      error_of(image_metadata(64, 64, holoscan::schema::ImageEncoding_CUSTOM));
  EXPECT_NE(error.find("RGB8"), std::string::npos) << error;
}

TEST(CompressedVideoBuilder, RejectsAnOddFrameSizeBeforeDecoding) {
  for (const auto& [width, height] : {std::pair{641, 480}, std::pair{640, 481}}) {
    const std::string error =
        error_of(image_metadata(width, height, holoscan::schema::ImageEncoding_RGB8));
    EXPECT_NE(error.find("only even frame sizes"), std::string::npos) << error;
    EXPECT_NE(error.find(std::to_string(width) + "x" + std::to_string(height)), std::string::npos)
        << error;
  }
}

TEST(CompressedVideoBuilder, RejectsAnEmptyFrameSize) {
  const std::string error = error_of(image_metadata(0, 0, holoscan::schema::ImageEncoding_RGB8));
  EXPECT_FALSE(error.empty());
}

TEST(CompressedVideoBuilder, ConstructionNamesTheMissingSdkWhenTheBuildCannotDecode) {
  const auto reason = video_decode_unavailable_reason(/*cuda_device=*/0, "h264");
  if (!reason) {
    GTEST_SKIP() << "This build can decode video on GPU 0, so there is no failure to observe.";
  }
  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const auto path = test_support::write_paired_video_file(
      scratch.path(),
      {{"video", 10, test_support::video_message_in("h264")}, {"video/metadata", 10, "S1"}});
  try {
    McapReplayer replayer(
        path, Topic<holoscan::schema::ImageT>{
                  "video", isaac::os::image_schema::ImageTensorFieldContract(8U * 1024U * 1024U)});
    FAIL() << "Construction should have failed for a topic that cannot be decoded.";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find(*reason), std::string::npos) << error.what();
    EXPECT_NE(std::string(error.what()).find("hardware video decoder"), std::string::npos)
        << error.what();
  }
}

TEST(CompressedVideoBuilder, ConstructionChecksTheCodecTheTopicWasRecordedIn) {
  // No GPU decodes this codec, so construction must refuse it whichever way the build and host
  // can decode H.264: the check follows the recording, not the logger's default codec.
  const auto reason = video_decode_unavailable_reason(/*cuda_device=*/0, "not-a-codec");
  ASSERT_TRUE(reason.has_value());

  const isaac::os::mcap_logger::ScopedTempDirectory scratch;
  const auto path = test_support::write_paired_video_file(
      scratch.path(),
      {{"video", 10, test_support::video_message_in("not-a-codec")}, {"video/metadata", 10, "S1"}});
  try {
    McapReplayer replayer(
        path, Topic<holoscan::schema::ImageT>{
                  "video", isaac::os::image_schema::ImageTensorFieldContract(8U * 1024U * 1024U)});
    FAIL() << "Construction should have failed for a codec that cannot be decoded.";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("Topic 'video'"), std::string::npos) << error.what();
    EXPECT_NE(std::string(error.what()).find(*reason), std::string::npos) << error.what();
  }
}

}  // namespace
}  // namespace isaac::os::mcap_replayer
