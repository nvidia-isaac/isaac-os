/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nv_video_decoder.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "NvDecoder/NvDecoder.h"
#include "cuda.h"
#include "cuviddec.h"
#include "holoscan/logger/logger.hpp"
#include "nvcuvid.h"

namespace nv_video_decoder {

static const std::map<std::string, cudaVideoCodec> CODEC_IDS = {
    {"H264", cudaVideoCodec_H264},
    {"H265", cudaVideoCodec_HEVC},
    {"HEVC", cudaVideoCodec_HEVC},
};

namespace {

bool probe_nvdec(uint32_t cuda_device, cudaVideoCodec codec) {
  if (cuInit(0) != CUDA_SUCCESS) {
    return false;
  }
  CUdevice dev = 0;
  if (cuDeviceGet(&dev, static_cast<int>(cuda_device)) != CUDA_SUCCESS) {
    return false;
  }
  CUcontext probe_ctx = nullptr;
  if (cuDevicePrimaryCtxRetain(&probe_ctx, dev) != CUDA_SUCCESS) {
    return false;
  }
  if (cuCtxPushCurrent(probe_ctx) != CUDA_SUCCESS) {
    cuDevicePrimaryCtxRelease(dev);
    return false;
  }
  CUVIDDECODECAPS caps = {};
  caps.eCodecType = codec;
  caps.eChromaFormat = cudaVideoChromaFormat_420;
  caps.nBitDepthMinus8 = 0;
  bool ok = false;
  if (cuvidGetDecoderCaps(&caps) == CUDA_SUCCESS) {
    ok = (caps.bIsSupported != 0) && (caps.nNumNVDECs > 0);
  }
  cuCtxPopCurrent(nullptr);
  cuDevicePrimaryCtxRelease(dev);
  return ok;
}

// Makes a context current for one scope. NvDecoder's constructor creates its lock and CUDA stream
// on whatever context is current, unlike the rest of its entry points, which push their own.
class ScopedContextPush {
 public:
  explicit ScopedContextPush(CUcontext context)
      : pushed_(cuCtxPushCurrent(context) == CUDA_SUCCESS) {}
  ~ScopedContextPush() {
    if (pushed_) {
      cuCtxPopCurrent(nullptr);
    }
  }
  ScopedContextPush(const ScopedContextPush&) = delete;
  ScopedContextPush& operator=(const ScopedContextPush&) = delete;
  [[nodiscard]] bool ok() const { return pushed_; }

 private:
  bool pushed_;
};

}  // namespace

bool NVVideoDecoder::is_available(uint32_t cuda_device, const std::string& codec) {
  std::string upper_codec = codec;
  std::ranges::transform(upper_codec, upper_codec.begin(), ::toupper);
  const auto codec_it = CODEC_IDS.find(upper_codec);
  if (codec_it == CODEC_IDS.end()) {
    return false;
  }
  // A property of the host's silicon, not of any call: one probe per (device, codec) suffices.
  static std::mutex mutex;
  static std::map<std::pair<uint32_t, cudaVideoCodec>, bool> cache;
  const std::scoped_lock lock(mutex);
  const auto key = std::make_pair(cuda_device, codec_it->second);
  const auto cached = cache.find(key);
  if (cached != cache.end()) {
    return cached->second;
  }
  return cache.emplace(key, probe_nvdec(cuda_device, codec_it->second)).first->second;
}

NVVideoDecoder::NVVideoDecoder(const std::string& codec, uint32_t cuda_device)
    : codec_(codec), cuda_device_(cuda_device) {}

NVVideoDecoder::~NVVideoDecoder() { cleanup(); }

bool NVVideoDecoder::initialize() {
  if (initialized_) {
    return true;
  }

  try {
    if (!setup_cuda()) {
      HOLOSCAN_LOG_ERROR("Failed to setup CUDA");
      throw std::runtime_error("Failed to setup CUDA");
    }

    if (!configure_decoder()) {
      HOLOSCAN_LOG_ERROR("Failed to configure decoder");
      throw std::runtime_error("Failed to configure decoder");
    }

    initialized_ = true;
    return true;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Initialization failed: {}", e.what());
    cleanup();
    throw;
  }
}

bool NVVideoDecoder::setup_cuda() {
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to initialize CUDA");
    return false;
  }

  result = cuDeviceGet(&cu_device_, cuda_device_);
  if (result != CUDA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Unable to access GPU device {}.", cuda_device_);
    return false;
  }

  result = cuDevicePrimaryCtxRetain(&cu_context_, cu_device_);
  if (result != CUDA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to retain primary context");
    return false;
  }

  return true;
}

bool NVVideoDecoder::configure_decoder() {
  std::string upper_codec = codec_;
  std::transform(upper_codec.begin(), upper_codec.end(), upper_codec.begin(), ::toupper);

  auto codec_it = CODEC_IDS.find(upper_codec);
  if (codec_it == CODEC_IDS.end()) {
    HOLOSCAN_LOG_ERROR("Unsupported codec: {}", codec_);
    return false;
  }

  const ScopedContextPush context(cu_context_);
  if (!context.ok()) {
    HOLOSCAN_LOG_ERROR("Failed to make the decoder's context current");
    return false;
  }
  try {
    // Keep decoded frames in GPU memory and match the encoder's low-latency tuning.
    constexpr bool kUseDeviceFrame = true;
    constexpr bool kLowLatency = true;
    decoder_ =
        std::make_unique<NvDecoder>(cu_context_, kUseDeviceFrame, codec_it->second, kLowLatency);
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to create decoder: {}", e.what());
    return false;
  }

  return true;
}

int NVVideoDecoder::decode_frame(const uint8_t* input_data, size_t input_size,
                                 std::vector<uint8_t*>& frame_ptrs) {
  if (!initialized_ || !decoder_) {
    return -1;
  }

  try {
    // Every packet the logger records is exactly one picture. Saying so makes the parser release
    // it now instead of waiting for the next packet to show where it ends.
    int n_frames =
        decoder_->Decode(input_data, static_cast<int>(input_size), CUVID_PKT_ENDOFPICTURE);
    frame_ptrs.clear();
    frame_ptrs.reserve(n_frames);
    for (int i = 0; i < n_frames; ++i) {
      frame_ptrs.push_back(decoder_->GetFrame());
    }
    return n_frames;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Decoding failed: {}", e.what());
    return -1;
  }
}

int NVVideoDecoder::get_width() const {
  if (!decoder_) {
    return 0;
  }
  return decoder_->GetWidth();
}

int NVVideoDecoder::get_height() const {
  if (!decoder_) {
    return 0;
  }
  return decoder_->GetHeight();
}

int NVVideoDecoder::get_pitch() const {
  if (!decoder_) {
    return 0;
  }
  return decoder_->GetDeviceFramePitch();
}

void NVVideoDecoder::cleanup() {
  // NvDecoder's destructor pushes its own context for the teardown and pops it again, so no
  // push is needed here.
  decoder_.reset();

  if (cu_context_) {
    cuDevicePrimaryCtxRelease(cu_device_);
    cu_context_ = nullptr;
  }

  initialized_ = false;
}

}  // namespace nv_video_decoder
