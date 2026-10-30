/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <exception>
#include <memory>
#include <string>
#include <thread>

#include "cuda.h"
#include "gtest/gtest.h"

#include "common/nv_video_encoder/nv_video_encoder.hpp"

namespace {

constexpr uint32_t kWidth = 640;
constexpr uint32_t kHeight = 480;
constexpr uint32_t kBitrate = 1'000'000;
constexpr uint32_t kFrameRate = 30;
constexpr uint32_t kCudaDevice = 0;

// Centralizes the runtime preconditions every test in this file shares: a
// usable CUDA driver, a device 0, and an actual NVENC engine on that device.
// A100/H100 silicon has 0 NVENC engines, so without the is_available() probe
// these tests would compile and run on those nodes only to fail at session
// open with NV_ENC_ERR_UNSUPPORTED_DEVICE - reported as a hard FAIL rather
// than a SKIP, which is the wrong signal in CI output.
//
// Implemented as a macro rather than a function: GTEST_SKIP() expands to a
// `return` from the enclosing function. Inside a helper function it would
// only return from the helper, leaving the test body to run and trip
// EXPECT_TRUE failures that mask the skip with FAILs in the CI report.
#define SKIP_IF_NVENC_UNAVAILABLE()                                                 \
  do {                                                                              \
    if (cuInit(0) != CUDA_SUCCESS) GTEST_SKIP() << "CUDA driver not available";     \
    if (!::nv_video_encoder::NVVideoEncoder::is_available(kCudaDevice)) {           \
      GTEST_SKIP() << "No NVENC engine on this GPU (e.g. A100/H100 ship NVDEC but " \
                      "no NVENC)";                                                  \
    }                                                                               \
  } while (0)

class NVVideoEncoderCodecNameTest : public ::testing::TestWithParam<std::string> {};

TEST_P(NVVideoEncoderCodecNameTest, InitializeAcceptsName) {
  SKIP_IF_NVENC_UNAVAILABLE();
  // The macro above asks about H.264. A GPU that encodes only that would run this for the HEVC
  // spellings and fail, though initialization is right to refuse a codec the device lacks.
  if (!::nv_video_encoder::NVVideoEncoder::is_available(kCudaDevice, GetParam())) {
    GTEST_SKIP() << GetParam() << " is not supported by the NVENC encoder on this GPU";
  }
  nv_video_encoder::NVVideoEncoder encoder(kWidth, kHeight, GetParam(), kBitrate, kFrameRate,
                                           kCudaDevice);
  EXPECT_TRUE(encoder.initialize());
}

INSTANTIATE_TEST_SUITE_P(SupportedCodecs, NVVideoEncoderCodecNameTest,
                         ::testing::Values("h264", "H264", "hevc", "HEVC", "h265", "H265"));

// The probe exists to predict initialize(), so the two have to agree for every codec. A probe that
// only opened a session would call H.265 available on an H.264-only part, and the encoder would
// then fail on the first frame instead of the topic being refused at start. On a GPU that offers a
// codec both say yes and on one that does not both say no, so this holds wherever it runs.
TEST_P(NVVideoEncoderCodecNameTest, ProbeAgreesWithInitialize) {
  SKIP_IF_NVENC_UNAVAILABLE();
  bool initialized = false;
  try {
    nv_video_encoder::NVVideoEncoder encoder(kWidth, kHeight, GetParam(), kBitrate, kFrameRate,
                                             kCudaDevice);
    initialized = encoder.initialize();
  } catch (const std::exception&) {
    initialized = false;
  }
  EXPECT_EQ(::nv_video_encoder::NVVideoEncoder::is_available(kCudaDevice, GetParam()), initialized)
      << "codec " << GetParam();
}

// A codec spelled two ways is one codec, and so one answer -- and one cache entry.
TEST(NVVideoEncoderAvailabilityTest, H265AndHevcShareAnAnswer) {
  const bool hevc = ::nv_video_encoder::NVVideoEncoder::is_available(kCudaDevice, "HEVC");
  EXPECT_EQ(::nv_video_encoder::NVVideoEncoder::is_available(kCudaDevice, "h265"), hevc);
  EXPECT_EQ(::nv_video_encoder::NVVideoEncoder::is_available(kCudaDevice, "H265"), hevc);
  EXPECT_EQ(::nv_video_encoder::NVVideoEncoder::is_available(kCudaDevice, "hevc"), hevc);
}

// No device offers a codec this encoder has no GUID for, so it is refused without probing anything.
TEST(NVVideoEncoderAvailabilityTest, AnUnknownCodecIsUnavailable) {
  EXPECT_FALSE(::nv_video_encoder::NVVideoEncoder::is_available(kCudaDevice, "av1"));
  EXPECT_FALSE(::nv_video_encoder::NVVideoEncoder::is_available(kCudaDevice, ""));
}

TEST(NVVideoEncoderCudaStackTest, TwoEncodersLeaveStackUnchanged) {
  SKIP_IF_NVENC_UNAVAILABLE();

  CUdevice dev;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) GTEST_SKIP() << "No CUDA device 0";

  CUcontext initial = nullptr;
  cuCtxGetCurrent(&initial);

  {
    nv_video_encoder::NVVideoEncoder e1(kWidth, kHeight, "h264", kBitrate, kFrameRate, kCudaDevice);
    ASSERT_TRUE(e1.initialize());
    nv_video_encoder::NVVideoEncoder e2(kWidth, kHeight, "h264", kBitrate, kFrameRate, kCudaDevice);
    ASSERT_TRUE(e2.initialize());
  }

  CUcontext after = nullptr;
  cuCtxGetCurrent(&after);
  EXPECT_EQ(after, initial);
}

// The test above cannot catch an unbalanced push. A push in initialize() and a pop in the
// destructor cancel out whenever both run against the same stack, which is all a
// single-threaded test can arrange. CUDA context stacks are per-thread, so that cancellation
// is exactly the thing that stops holding once the two calls land on different threads -- and
// a logger that opens an encoder lazily on a worker thread and destroys it from a shutdown
// path on another does precisely that. The pop then applies to a stack that never saw the
// push, silently removing whatever that thread had current.
TEST(NVVideoEncoderCudaStackTest, SurvivesInitializeAndDestroyOnDifferentThreads) {
  SKIP_IF_NVENC_UNAVAILABLE();

  CUdevice dev;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) GTEST_SKIP() << "No CUDA device 0";

  // Stands in for the context the application has current on the thread that tears the logger
  // down -- the one an unbalanced pop would consume.
  CUcontext sentinel = nullptr;
  ASSERT_EQ(cuDevicePrimaryCtxRetain(&sentinel, dev), CUDA_SUCCESS);
  ASSERT_EQ(cuCtxPushCurrent(sentinel), CUDA_SUCCESS);

  std::unique_ptr<nv_video_encoder::NVVideoEncoder> encoder;
  CUcontext worker_current = nullptr;
  CUresult worker_current_result = CUDA_ERROR_UNKNOWN;
  bool initialized = false;
  // initialize() throws when setup fails, and an exception leaving a std::thread body calls
  // std::terminate, which would take the whole test executable down rather than fail this test.
  // So the worker hands it back to be rethrown here, where gtest reports it as a failure.
  std::exception_ptr worker_failure;
  std::thread worker([&] {
    try {
      encoder = std::make_unique<nv_video_encoder::NVVideoEncoder>(
          kWidth, kHeight, "h264", kBitrate, kFrameRate, kCudaDevice);
      initialized = encoder->initialize();
      worker_current_result = cuCtxGetCurrent(&worker_current);
    } catch (...) {
      worker_failure = std::current_exception();
    }
  });
  worker.join();
  if (worker_failure || !initialized) {
    // Put this thread's stack back before failing, so the next test does not inherit the sentinel.
    cuCtxPopCurrent(nullptr);
    cuDevicePrimaryCtxRelease(dev);
    if (worker_failure) {
      std::rethrow_exception(worker_failure);
    }
    FAIL() << "initialize() returned false on the worker thread";
  }

  // Opening the session must not leave the worker holding a context it never asked for: this
  // thread goes on to run unrelated work, and on a multi-GPU box that context is not even
  // necessarily the right device's.
  // The query has to have worked for the null below to mean anything: a failed query leaves its
  // output null too.
  ASSERT_EQ(worker_current_result, CUDA_SUCCESS);
  EXPECT_EQ(worker_current, nullptr) << "initialize() left a context current on its own thread";

  // Destroying on a different thread must leave this thread's stack exactly as it was.
  encoder.reset();
  CUcontext after = nullptr;
  ASSERT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
  EXPECT_EQ(after, sentinel) << "the destructor popped a context this thread still needed";

  cuCtxPopCurrent(nullptr);
  cuDevicePrimaryCtxRelease(dev);
}

// The test above shows initialize() leaves an initially empty worker stack empty, which a version
// that popped too much, or nothing, could still satisfy. A worker that already has a context
// current
// -- an application thread that uses CUDA itself -- is the stronger case: initialize() must put
// back exactly that context, neither dropping it nor leaving the encoder's on top of it.
//
// The worker's context is one of its own, not the device's primary context: the encoder pushes the
// primary context, so a worker holding the primary one could not tell its own context from a push
// that was never popped.
TEST(NVVideoEncoderCudaStackTest, InitializeRestoresAContextTheWorkerAlreadyHad) {
  SKIP_IF_NVENC_UNAVAILABLE();

  CUdevice dev;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) GTEST_SKIP() << "No CUDA device 0";

  CUcontext workers_context = nullptr;
  bool created = false;
  bool initialized = false;
  CUcontext current_after = nullptr;
  CUresult current_result = CUDA_ERROR_UNKNOWN;
  std::exception_ptr worker_failure;
  std::thread worker([&] {
    try {
      // cuCtxCreate leaves the new context current on this thread.
      created = cuCtxCreate(&workers_context, nullptr, 0, dev) == CUDA_SUCCESS;
      if (created) {
        nv_video_encoder::NVVideoEncoder encoder(kWidth, kHeight, "h264", kBitrate, kFrameRate,
                                                 kCudaDevice);
        initialized = encoder.initialize();
        current_result = cuCtxGetCurrent(&current_after);
      }
    } catch (...) {
      worker_failure = std::current_exception();
    }
    if (created) {
      cuCtxPopCurrent(nullptr);
      cuCtxDestroy(workers_context);
    }
  });
  worker.join();

  if (worker_failure) {
    std::rethrow_exception(worker_failure);
  }
  ASSERT_TRUE(created) << "the worker could not create a context of its own";
  ASSERT_TRUE(initialized);
  ASSERT_EQ(current_result, CUDA_SUCCESS);
  EXPECT_EQ(current_after, workers_context)
      << "initialize() did not leave the worker's own context current";
}

}  // namespace
