/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/memory/tensor_storage.hpp"
#include "holoscan/sensor_io/image_metadata.hpp"
#include <cuda_runtime.h>

#include "common/mcap_logger/frame_encoder.h"

namespace isaac::os::mcap_logger {
namespace {

constexpr std::uint32_t kWidth = 320;
constexpr std::uint32_t kHeight = 240;

// A rank-1 contiguous uint8 host tensor -- the exact shape ImageTensorFieldContract pins for
// holoscan.schema.Image's `data` field, which is why the pixel geometry never comes from the
// tensor itself. The tensor owns a copy of the bytes through its release function, so the
// caller's vector need not outlive it.
std::shared_ptr<holoscan::Tensor> MakeFlatRgbTensor(const std::vector<std::uint8_t>& bytes) {
  // The release function owns the copy, so nothing here is allocated or freed by hand: the bytes
  // live exactly as long as that function object does, whether or not the tensor ever calls it.
  auto storage = std::make_shared<std::vector<std::uint8_t>>(bytes);
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(storage->size())};
  auto tensor = holoscan::make_tensor(
      shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt, holoscan::MemoryStorageType::kSystem,
      storage->data(),
      [storage](void*) -> holoscan::expected<void, holoscan::Error> { return {}; });
  if (!tensor) {
    return nullptr;
  }
  return std::make_shared<holoscan::Tensor>(std::move(*tensor));
}

// The same flat frame, but resident in CUDA device memory. A Holoscan camera pipeline delivers
// its frames here rather than on the host, so this is the staging path essentially every real
// frame takes.
std::shared_ptr<holoscan::Tensor> MakeDeviceRgbTensor(const std::vector<std::uint8_t>& bytes) {
  void* device_pixels = nullptr;
  if (cudaMalloc(&device_pixels, bytes.size()) != cudaSuccess) {
    return nullptr;
  }
  if (cudaMemcpy(device_pixels, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    cudaFree(device_pixels);
    return nullptr;
  }
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(bytes.size())};
  auto tensor = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                                      holoscan::MemoryStorageType::kDevice, device_pixels,
                                      [](void* ptr) -> holoscan::expected<void, holoscan::Error> {
                                        cudaFree(ptr);
                                        return {};
                                      });
  if (!tensor) {
    cudaFree(device_pixels);
    return nullptr;
  }
  return std::make_shared<holoscan::Tensor>(std::move(*tensor));
}

int CountOccurrences(const std::string& haystack, const std::string& needle) {
  int count = 0;
  for (std::size_t at = haystack.find(needle); at != std::string::npos;
       at = haystack.find(needle, at + needle.size())) {
    ++count;
  }
  return count;
}

TEST(FrameEncoder, ReportsTheLowercaseCodecFoxgloveExpects) {
  // Asked for H.265 specifically: a GPU that only encodes H.264 passes the default check, and then
  // refuses to construct this encoder.
  if (!FrameEncoder::hardware_available(0, VideoCodec::kH265)) {
    GTEST_SKIP() << "No H.265 NVENC encoder on this host.";
  }
  const auto encoder =
      std::make_shared<FrameEncoder>(VideoEncodeOptions{.codec = VideoCodec::kH265});
  EXPECT_EQ(encoder->foxglove_codec_format(), "h265");
}

TEST(FrameEncoder, EncodesAnRgbFrameToANonEmptyBitstream) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  const std::vector<std::uint8_t> bytes(static_cast<std::size_t>(kWidth) * kHeight * 3, 128);
  const auto tensor = MakeFlatRgbTensor(bytes);
  ASSERT_NE(tensor, nullptr);
  const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});
  const auto encoded =
      encoder->encode(*tensor, kWidth, kHeight, holoscan::schema::ImageEncoding_RGB8);
  ASSERT_TRUE(encoded.has_value());
  EXPECT_FALSE(encoded->empty());
}

// A session's staging buffers are made for its first frame and reused by every later one, because
// the session's frame size is fixed. A three-channel host frame needs two -- the ARGB output and a
// landing buffer for the copy -- and a four-channel frame needs only the ARGB output, so neither
// count depends on how many frames the session encodes. Allocating and freeing them per frame would
// show as a count that grows with the frame number.
TEST(FrameEncoder, ReusesItsStagingBuffersAcrossFrames) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  constexpr int kFrames = 20;

  const std::vector<std::uint8_t> rgb(static_cast<std::size_t>(kWidth) * kHeight * 3, 128);
  const auto rgb_tensor = MakeFlatRgbTensor(rgb);
  ASSERT_NE(rgb_tensor, nullptr);
  {
    const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});
    const std::uint64_t before = FrameEncoder::staging_allocations();
    for (int frame = 0; frame < kFrames; ++frame) {
      ASSERT_TRUE(
          encoder->encode(*rgb_tensor, kWidth, kHeight, holoscan::schema::ImageEncoding_RGB8)
              .has_value())
          << "frame " << frame;
    }
    EXPECT_EQ(FrameEncoder::staging_allocations() - before, 2U);
  }

  const std::vector<std::uint8_t> rgba(static_cast<std::size_t>(kWidth) * kHeight * 4, 128);
  const auto rgba_tensor = MakeFlatRgbTensor(rgba);
  ASSERT_NE(rgba_tensor, nullptr);
  {
    const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});
    const std::uint64_t before = FrameEncoder::staging_allocations();
    for (int frame = 0; frame < kFrames; ++frame) {
      ASSERT_TRUE(
          encoder->encode(*rgba_tensor, kWidth, kHeight, holoscan::schema::ImageEncoding_RGBA8)
              .has_value())
          << "frame " << frame;
    }
    EXPECT_EQ(FrameEncoder::staging_allocations() - before, 1U);
  }
}

// The session's CUDA context is pushed before the copy is issued, while the frame's memory was
// allocated in whatever context the pipeline was using. Unified addressing on a single device
// makes the two reachable from one another; this test is what proves it, and would fail on a
// configuration where they are not.
TEST(FrameEncoder, EncodesAnRgbFrameThatAlreadyLivesOnTheGpu) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  const std::vector<std::uint8_t> bytes(static_cast<std::size_t>(kWidth) * kHeight * 3, 128);
  const auto tensor = MakeDeviceRgbTensor(bytes);
  ASSERT_NE(tensor, nullptr);
  // Guards the point of the test: a host-resident tensor would take the other staging branch and
  // leave the device-to-device copy uncovered again.
  ASSERT_EQ(tensor->device().device_type, kDLCUDA);
  const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});
  const auto encoded =
      encoder->encode(*tensor, kWidth, kHeight, holoscan::schema::ImageEncoding_RGB8);
  ASSERT_TRUE(encoded.has_value());
  EXPECT_FALSE(encoded->empty());
}

// The Annex-B start code prefix, as H.264 defines it. A NAL unit may be preceded by this alone,
// or by a leading zero_byte making the sequence four bytes -- the longer form is required before
// a sequence or picture parameter set and before the first NAL of an access unit, and optional
// elsewhere. Both spellings are legal, so a stream carrying either is well-formed.
inline constexpr std::array<std::uint8_t, 3> kAnnexBStartCodePrefix{0x00U, 0x00U, 0x01U};

// True when `data` opens with an Annex-B start code in either legal spelling. Deliberately not a
// fixed four-byte comparison: which spelling an encoder emits is its own choice, so pinning one
// would fail this test against a bitstream that is perfectly valid and decodes fine.
[[nodiscard]] bool StartsWithAnnexBStartCode(std::span<const std::byte> data) {
  const auto matches_prefix_at = [data](std::size_t offset) {
    if (data.size() < offset + kAnnexBStartCodePrefix.size()) {
      return false;
    }
    for (std::size_t i = 0; i < kAnnexBStartCodePrefix.size(); ++i) {
      if (static_cast<std::uint8_t>(data[offset + i]) != kAnnexBStartCodePrefix[i]) {
        return false;
      }
    }
    return true;
  };
  // Offset 1 covers the four-byte form, whose leading zero_byte must itself be zero.
  return matches_prefix_at(0) ||
         (!data.empty() && static_cast<std::uint8_t>(data[0]) == 0x00U && matches_prefix_at(1));
}

// Every frame is an IDR with SPS/PPS (NVVideoEncoder forces FORCEINTRA|FORCEIDR|OUTPUT_SPSPPS),
// so each frame must start with an Annex-B start code -- that independence is what lets a
// replayer decode any frame without seeking back to a keyframe.
TEST(FrameEncoder, EmitsAnnexBStartCodeOnEveryFrame) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  const std::vector<std::uint8_t> bytes(static_cast<std::size_t>(kWidth) * kHeight * 3, 64);
  const auto tensor = MakeFlatRgbTensor(bytes);
  ASSERT_NE(tensor, nullptr);
  const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});
  for (int i = 0; i < 3; ++i) {
    const auto encoded =
        encoder->encode(*tensor, kWidth, kHeight, holoscan::schema::ImageEncoding_RGB8);
    ASSERT_TRUE(encoded.has_value()) << "frame " << i;
    EXPECT_TRUE(StartsWithAnnexBStartCode(*encoded)) << "frame " << i;
  }
}

// The four-channel device path is its own code: it hands the tensor straight to the
// out-of-place channel swap instead of copying into the staging buffer and swapping in place,
// so the host test above does not reach it.
TEST(FrameEncoder, EncodesAFourChannelFrameThatAlreadyLivesOnTheGpu) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  const std::vector<std::uint8_t> bytes(static_cast<std::size_t>(kWidth) * kHeight * 4, 200);
  const auto tensor = MakeDeviceRgbTensor(bytes);
  ASSERT_NE(tensor, nullptr);
  // Guards the point of the test: a host-resident tensor would take the in-place branch and
  // leave the out-of-place conversion uncovered again.
  ASSERT_EQ(tensor->device().device_type, kDLCUDA);
  const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});
  const auto encoded =
      encoder->encode(*tensor, kWidth, kHeight, holoscan::schema::ImageEncoding_RGBA8);
  ASSERT_TRUE(encoded.has_value());
  EXPECT_FALSE(encoded->empty());
}

TEST(FrameEncoder, EncodesAFourChannelFrame) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  const std::vector<std::uint8_t> bytes(static_cast<std::size_t>(kWidth) * kHeight * 4, 200);
  const auto tensor = MakeFlatRgbTensor(bytes);
  ASSERT_NE(tensor, nullptr);
  const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});
  const auto encoded =
      encoder->encode(*tensor, kWidth, kHeight, holoscan::schema::ImageEncoding_RGBA8);
  ASSERT_TRUE(encoded.has_value());
  EXPECT_FALSE(encoded->empty());
}

// A source encoding the encoder does not accept is refused rather than encoded as something
// else. MONO8 stands in for the whole unsupported set, which is every encoding but RGB8 and
// RGBA8 -- including ones that read as RGB-family, such as BGR8. Only those two are accepted
// because the encoder consumes ARGB and the matching decoder returns NV12, so only they survive
// a record-then-replay round trip. The check lives here, per frame, rather than at attach()
// time, because the tensor contract discards the image encoding and none is in scope there.
TEST(FrameEncoder, RejectsAnUnsupportedSourceEncoding) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  const std::vector<std::uint8_t> bytes(static_cast<std::size_t>(kWidth) * kHeight, 128);
  const auto tensor = MakeFlatRgbTensor(bytes);
  ASSERT_NE(tensor, nullptr);
  const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});
  EXPECT_FALSE(
      encoder->encode(*tensor, kWidth, kHeight, holoscan::schema::ImageEncoding_MONO8).has_value());
}

// A frame shorter than width * height * channels would make the staging copy read past the end
// of the tensor's buffer, so it is refused before any device memory is touched.
TEST(FrameEncoder, RejectsAFrameShorterThanItsDeclaredGeometry) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  const std::vector<std::uint8_t> bytes(static_cast<std::size_t>(kWidth) * kHeight * 3 - 1, 128);
  const auto tensor = MakeFlatRgbTensor(bytes);
  ASSERT_NE(tensor, nullptr);
  const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});
  EXPECT_FALSE(
      encoder->encode(*tensor, kWidth, kHeight, holoscan::schema::ImageEncoding_RGB8).has_value());
}

TEST(FrameEncoder, RejectsAFrameWhoseDimensionsChangedMidStream) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  const std::vector<std::uint8_t> small(static_cast<std::size_t>(kWidth) * kHeight * 3, 10);
  const std::vector<std::uint8_t> large(640U * 480U * 3, 10);
  const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});
  ASSERT_TRUE(
      encoder
          ->encode(*MakeFlatRgbTensor(small), kWidth, kHeight, holoscan::schema::ImageEncoding_RGB8)
          .has_value());
  // The session is sized on the first frame; a different size must be refused, not silently
  // encoded against mismatched dimensions.
  EXPECT_FALSE(
      encoder->encode(*MakeFlatRgbTensor(large), 640, 480, holoscan::schema::ImageEncoding_RGB8)
          .has_value());
}

// A frame too small for the hardware encoder is one of the failures the availability probe
// structurally cannot foresee: the probe opens its own fixed-size session, while the real one is
// sized from the first frame, which does not exist yet when the probe runs. NVENC refuses to
// initialize an encoder at 4x2, so the constructor hands back a perfectly good encoder and the
// session then fails to open on the first frame. That failure is permanent -- the stream's size
// is fixed for its whole life, so no later frame can succeed where the first one failed -- so it
// has to be reported exactly once, in full, rather than swallowed by the dropped-frame budget or
// repeated at frame rate while reopening a session every time.
TEST(FrameEncoder, ReportsAnEncoderThatCannotStartExactlyOnce) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  constexpr std::uint32_t kTinyWidth = 4;
  constexpr std::uint32_t kTinyHeight = 2;
  const std::vector<std::uint8_t> bytes(static_cast<std::size_t>(kTinyWidth) * kTinyHeight * 3,
                                        128);
  const auto tensor = MakeFlatRgbTensor(bytes);
  ASSERT_NE(tensor, nullptr);
  const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  // More frames than the dropped-frame log budget, so a report that went through that budget
  // instead would still be visible here more than once.
  constexpr int kFrames = 5;
  for (int i = 0; i < kFrames; ++i) {
    EXPECT_FALSE(
        encoder->encode(*tensor, kTinyWidth, kTinyHeight, holoscan::schema::ImageEncoding_RGB8)
            .has_value())
        << "frame " << i;
  }
  const std::string output =
      testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();

  EXPECT_EQ(CountOccurrences(output, "will record nothing for the rest of the run"), 1)
      << "A topic whose encoder can never start must say so once, and must not repeat it (or "
         "reopen an encoding session) for every frame of the run:\n"
      << output;
}

// The availability probe has to answer for the device the caller configured, not for whichever
// device the process happens to see first: a topic aimed at an ordinal this host does not have
// must be told so here, while it can still choose to record raw frames, rather than at its first
// frame, when that choice is already gone.
TEST(FrameEncoder, ReportsADeviceThisHostDoesNotHaveAsUnavailable) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  // The first ordinal past the last real device, so it is absent whatever this host has.
  const auto missing_device = static_cast<std::uint32_t>(device_count);
  EXPECT_FALSE(FrameEncoder::hardware_available(missing_device));
  // Asked again, because the answer is cached per device: a cache keyed on nothing would hand
  // this call the answer it computed for the missing device above, or vice versa.
  EXPECT_TRUE(FrameEncoder::hardware_available(0));
  EXPECT_FALSE(FrameEncoder::hardware_available(missing_device));
}

// The constructor is the path that cannot probe one device and encode on another: it is handed the
// options, and the device it probes is the one those options name. It has to throw rather than
// hand back an encoder that can never encode, and say which device and what to change.
TEST(FrameEncoder, ThrowsForADeviceThisHostDoesNotHave) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  // The first ordinal past the last real device, so it is absent whatever this host has.
  const auto missing_device = static_cast<std::uint32_t>(device_count);
  try {
    FrameEncoder encoder(VideoEncodeOptions{.cuda_device = missing_device}, "camera");
    FAIL() << "An encoder was built for a device this host does not have.";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("topic 'camera'"), std::string::npos) << message;
    EXPECT_NE(message.find("GPU " + std::to_string(missing_device)), std::string::npos) << message;
  }
}

TEST(FrameEncoder, BuildsAUsableEncoderForAnAvailableDevice) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{.cuda_device = 0});
  const std::vector<std::uint8_t> bytes(static_cast<std::size_t>(kWidth) * kHeight * 3, 128);
  const auto tensor = MakeFlatRgbTensor(bytes);
  ASSERT_NE(tensor, nullptr);
  const auto encoded =
      encoder->encode(*tensor, kWidth, kHeight, holoscan::schema::ImageEncoding_RGB8);
  ASSERT_TRUE(encoded.has_value());
  EXPECT_FALSE(encoded->empty());
}

// A zero bitrate is wrong on every machine, so it must throw std::invalid_argument rather than the
// std::runtime_error that means "this host has no encoder" -- and it must do so even for a device
// that would have failed the probe, which is why the check runs first.
TEST(FrameEncoder, ThrowsInvalidArgumentForAZeroBitrateRatherThanAnAbsentEncoderError) {
  EXPECT_THROW(std::make_shared<FrameEncoder>(VideoEncodeOptions{.bitrate = 0}),
               std::invalid_argument);
  EXPECT_THROW(std::make_shared<FrameEncoder>(VideoEncodeOptions{.frame_rate = 0}),
               std::invalid_argument);
  // An ordinal far past anything this host could have, so the probe would certainly refuse it:
  // the zero bitrate still has to surface as invalid_argument rather than as that refusal.
  constexpr std::uint32_t kAbsentDevice = 1024;
  EXPECT_THROW(std::make_shared<FrameEncoder>(
                   VideoEncodeOptions{.bitrate = 0, .cuda_device = kAbsentDevice}),
               std::invalid_argument);
}

}  // namespace
}  // namespace isaac::os::mcap_logger
