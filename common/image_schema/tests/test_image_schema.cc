/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>

#include "gtest/gtest.h"

#include "common/image_schema/image_schema.h"

namespace isaac::os::image_schema {
namespace {

TEST(BytesPerPixelTest, Rgb8IsThree) {
  auto result = bytes_per_pixel(holoscan::schema::ImageEncoding_RGB8);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 3U);
}

TEST(BytesPerPixelTest, Bgr8IsThree) {
  auto result = bytes_per_pixel(holoscan::schema::ImageEncoding_BGR8);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 3U);
}

TEST(BytesPerPixelTest, Rgba8IsFour) {
  auto result = bytes_per_pixel(holoscan::schema::ImageEncoding_RGBA8);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 4U);
}

TEST(BytesPerPixelTest, Bgra8IsFour) {
  auto result = bytes_per_pixel(holoscan::schema::ImageEncoding_BGRA8);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 4U);
}

TEST(BytesPerPixelTest, Mono8IsOne) {
  auto result = bytes_per_pixel(holoscan::schema::ImageEncoding_MONO8);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 1U);
}

TEST(BytesPerPixelTest, UnsupportedEncodingReturnsAnError) {
  // Cast an out-of-range value rather than naming one of ImageEncoding's real-but-unsupported
  // members (e.g. a multi-planar/float/Bayer encoding) -- this guarantees hitting the default:
  // case regardless of exactly which enumerators this SDK version defines, without this test
  // needing to know their names.
  const auto unsupported = static_cast<holoscan::schema::ImageEncoding>(0xFFFF);
  auto result = bytes_per_pixel(unsupported);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// ImageTensorFieldContract() itself returns an opaque
// holoscan::DescriptorTensorTopicContract<ImageT> with no public accessor for the bound it was
// given (by design -- Graph::export_topic is the only consumer), so these tests confirm
// construction succeeds (or throws, for the unsupported-encoding case) rather than inspecting the
// computed contract directly.
TEST(ImageTensorFieldContractTest, BuildsForAGivenByteBound) {
  EXPECT_NO_THROW({ auto contract = ImageTensorFieldContract(1920U * 1080U * 3U); });
}

TEST(ImageTensorFieldContractTest, BuildsForRgb8) {
  EXPECT_NO_THROW({
    auto contract = ImageTensorFieldContract(1920, 1080, holoscan::schema::ImageEncoding_RGB8);
  });
}

TEST(ImageTensorFieldContractTest, BuildsForRgba8) {
  EXPECT_NO_THROW({
    auto contract = ImageTensorFieldContract(640, 480, holoscan::schema::ImageEncoding_RGBA8);
  });
}

TEST(ImageTensorFieldContractTest, BuildsForMono8) {
  EXPECT_NO_THROW({
    auto contract = ImageTensorFieldContract(320, 240, holoscan::schema::ImageEncoding_MONO8);
  });
}

TEST(ImageTensorFieldContractTest, UnsupportedEncodingThrows) {
  const auto unsupported = static_cast<holoscan::schema::ImageEncoding>(0xFFFF);
  EXPECT_THROW(ImageTensorFieldContract(1920, 1080, unsupported), std::invalid_argument);
}

TEST(ImageTensorFieldContractTest, OverflowingDimensionsThrow) {
  // width * height * bytes_per_pixel(RGB8) overflows a 64-bit byte count for these dimensions,
  // and must throw rather than silently wrap around to a small, wrong bound.
  EXPECT_THROW(
      ImageTensorFieldContract(0xFFFFFFFFU, 0xFFFFFFFFU, holoscan::schema::ImageEncoding_RGB8),
      std::invalid_argument);
}

TEST(ImageTensorFieldContractTest, MemoryKindDefaultsToHost) {
  // No accessor to inspect the built representation's memory_kind directly (see the class
  // comment above); this just confirms the default argument is accepted and builds cleanly, the
  // same way the default-argument call sites elsewhere in this codebase rely on it.
  EXPECT_NO_THROW({ auto contract = ImageTensorFieldContract(1920U * 1080U * 3U); });
}

TEST(ImageTensorFieldContractTest, BuildsForCudaDeviceMemoryKind) {
  EXPECT_NO_THROW({
    auto contract = ImageTensorFieldContract(1920U * 1080U * 3U, holoscan::MemoryKind::kCudaDevice);
  });
}

TEST(ImageTensorFieldContractTest, WidthHeightOverloadAcceptsMemoryKind) {
  EXPECT_NO_THROW({
    auto contract = ImageTensorFieldContract(1920, 1080, holoscan::schema::ImageEncoding_RGB8,
                                             holoscan::MemoryKind::kCudaDevice);
  });
}

}  // namespace
}  // namespace isaac::os::image_schema
