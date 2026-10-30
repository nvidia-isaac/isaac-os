/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>
#include <thread>

#include "cuda.h"
#include "gtest/gtest.h"

#include "common/nv_video_decoder/nv_video_decoder.hpp"

namespace {

// Mirrors SKIP_IF_NVENC_UNAVAILABLE in test_nv_video_encoder.cpp. Macro form
// is required: GTEST_SKIP() expands to a `return` from the enclosing
// function, so a helper would only return from the helper and let the test
// body keep running into a hard FAIL.
#define SKIP_IF_NVDEC_UNAVAILABLE()                                             \
  do {                                                                          \
    if (cuInit(0) != CUDA_SUCCESS) GTEST_SKIP() << "CUDA driver not available"; \
    if (!::nv_video_decoder::NVVideoDecoder::is_available()) {                  \
      GTEST_SKIP() << "No NVDEC engine on this GPU";                            \
    }                                                                           \
  } while (0)

class NVVideoDecoderCodecNameTest : public ::testing::TestWithParam<std::string> {};

TEST_P(NVVideoDecoderCodecNameTest, InitializeAcceptsName) {
  SKIP_IF_NVDEC_UNAVAILABLE();
  nv_video_decoder::NVVideoDecoder decoder(GetParam(), /*cuda_device=*/0);
  EXPECT_TRUE(decoder.initialize());
}

INSTANTIATE_TEST_SUITE_P(SupportedCodecs, NVVideoDecoderCodecNameTest,
                         ::testing::Values("h264", "H264", "hevc", "HEVC", "h265", "H265"));

TEST(NVVideoDecoderCudaStackTest, TwoDecodersLeaveStackUnchanged) {
  SKIP_IF_NVDEC_UNAVAILABLE();

  CUdevice dev;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) GTEST_SKIP() << "No CUDA device 0";

  CUcontext initial = nullptr;
  cuCtxGetCurrent(&initial);

  {
    nv_video_decoder::NVVideoDecoder d1("h264", /*cuda_device=*/0);
    ASSERT_TRUE(d1.initialize());
    nv_video_decoder::NVVideoDecoder d2("h264", /*cuda_device=*/0);
    ASSERT_TRUE(d2.initialize());
  }

  CUcontext after = nullptr;
  cuCtxGetCurrent(&after);
  EXPECT_EQ(after, initial);
}

TEST(NVVideoDecoderCudaStackTest, CreatingAndDestroyingOnDifferentThreadsLeavesBothStacksAlone) {
  SKIP_IF_NVDEC_UNAVAILABLE();

  CUdevice device = 0;
  ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);

  std::unique_ptr<nv_video_decoder::NVVideoDecoder> decoder;
  CUcontext left_current = reinterpret_cast<CUcontext>(1);
  std::jthread creator([&] {
    decoder = std::make_unique<nv_video_decoder::NVVideoDecoder>("h264", /*cuda_device=*/0);
    EXPECT_TRUE(decoder->initialize());
    cuCtxGetCurrent(&left_current);
  });
  creator.join();
  EXPECT_EQ(left_current, nullptr)
      << "initialize() left its context current on the creating thread";

  CUcontext during = nullptr;
  CUcontext after = nullptr;
  std::jthread destroyer([&] {
    CUcontext mine = nullptr;
    ASSERT_EQ(cuDevicePrimaryCtxRetain(&mine, device), CUDA_SUCCESS);
    ASSERT_EQ(cuCtxPushCurrent(mine), CUDA_SUCCESS);
    during = mine;
    decoder.reset();
    cuCtxGetCurrent(&after);
    cuCtxPopCurrent(nullptr);
    cuDevicePrimaryCtxRelease(device);
  });
  destroyer.join();
  EXPECT_EQ(after, during) << "destroying the decoder popped a context it never pushed";
}

TEST(NVVideoDecoderAvailabilityTest, IsFalseForAnUnknownCodecOrDevice) {
  EXPECT_FALSE(nv_video_decoder::NVVideoDecoder::is_available(0, "vp9"));
  EXPECT_FALSE(nv_video_decoder::NVVideoDecoder::is_available(9999, "h264"));
}

}  // namespace
