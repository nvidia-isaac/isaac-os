/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nv_video_encoder.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "NvEncoder/NvEncoderCuda.h"
#include "cuda.h"
#include "holoscan/logger/logger.hpp"
#include "nvEncodeAPI.h"

namespace nv_video_encoder {

// Codec GUIDs. Mirrors NVVideoDecoder's CODEC_IDS so a config that resolves
// on one side resolves on the other; "H265" and "HEVC" both map to HEVC.
static const std::map<std::string, GUID> CODEC_GUIDS = {{"H264", NV_ENC_CODEC_H264_GUID},
                                                        {"H265", NV_ENC_CODEC_HEVC_GUID},
                                                        {"HEVC", NV_ENC_CODEC_HEVC_GUID}};

namespace {

// The GUID for a codec spelling, in any case, or nullopt for one this encoder does not know.
[[nodiscard]] std::optional<GUID> ResolveCodec(const std::string& codec) {
  std::string upper = codec;
  std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
  const auto found = CODEC_GUIDS.find(upper);
  if (found == CODEC_GUIDS.end()) {
    return std::nullopt;
  }
  return found->second;
}

// A minimal session that can also say which codecs the device's encoder offers. Subclassed only
// to reach the session's function table and handle, which NvEncoder keeps protected.
class ProbeSession : public NvEncoderCuda {
 public:
  using NvEncoderCuda::NvEncoderCuda;

  // Whether this device's encoder lists `codec` among the encode GUIDs it supports. A session
  // opening proves only that the silicon has an encoder, not that it has one for every codec: an
  // H.264-only part opens a session just fine and then refuses an H.265 configuration on the first
  // frame. Any failure to ask reads as unsupported.
  [[nodiscard]] bool supports(const GUID& codec) const {
    uint32_t count = 0;
    if (m_nvenc.nvEncGetEncodeGUIDCount(m_hEncoder, &count) != NV_ENC_SUCCESS || count == 0) {
      return false;
    }
    std::vector<GUID> guids(count);
    uint32_t returned = 0;
    if (m_nvenc.nvEncGetEncodeGUIDs(m_hEncoder, guids.data(), count, &returned) != NV_ENC_SUCCESS) {
      return false;
    }
    return std::any_of(guids.begin(), guids.begin() + returned, [&codec](const GUID& offered) {
      return std::memcmp(&offered, &codec, sizeof(GUID)) == 0;
    });
  }
};

// Whether `cuda_device` can open an NVENC session, by opening one, and whether that session's
// encoder supports `codec`. Kept apart from the cache in is_available() so that is only about
// remembering the answer.
[[nodiscard]] bool ProbeEncoderSession(std::uint32_t cuda_device, const GUID& codec) {
  if (cuInit(0) != CUDA_SUCCESS) {
    return false;
  }
  int device_count = 0;
  if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count == 0) {
    return false;
  }
  // An ordinal this host does not have fails here, which is what turns a mistyped device index
  // into an ordinary "no encoder available" answer instead of a session that only fails later,
  // on the first frame.
  CUdevice dev = 0;
  if (cuDeviceGet(&dev, static_cast<int>(cuda_device)) != CUDA_SUCCESS) {
    return false;
  }
  // Use the device's primary context (same pattern as setup_cuda()): the
  // primary context API has a stable signature across CUDA toolkit major
  // versions, whereas cuCtxCreate is macro-redirected to cuCtxCreate_v4
  // (4-arg form) in CUDA 13+ headers. Retain bumps a refcount; release
  // drops it without tearing the context down for any other holder.
  CUcontext probe_ctx = nullptr;
  if (cuDevicePrimaryCtxRetain(&probe_ctx, dev) != CUDA_SUCCESS) {
    return false;
  }
  if (cuCtxPushCurrent(probe_ctx) != CUDA_SUCCESS) {
    cuDevicePrimaryCtxRelease(dev);
    return false;
  }
  bool ok = false;
  try {
    // Minimal session: NvEncoderCuda's constructor calls
    // nvEncOpenEncodeSessionEx, which is what returns
    // NV_ENC_ERR_UNSUPPORTED_DEVICE on NVENC-less silicon. NVML's
    // nvmlDeviceGetEncoderCapacity was tried first but turned out to
    // report a non-zero capacity on A100 despite the silicon shipping
    // 0 NVENC engines - it's a utilization-style API, not a presence
    // indicator. Session-open is the canonical capability test.
    ProbeSession probe(probe_ctx, 64, 64, NV_ENC_BUFFER_FORMAT_ARGB, 0);
    ok = probe.supports(codec);
  } catch (const NVENCException&) {
    ok = false;
  }
  cuCtxPopCurrent(nullptr);
  cuDevicePrimaryCtxRelease(dev);
  return ok;
}

}  // namespace

bool NVVideoEncoder::is_available(std::uint32_t cuda_device, const std::string& codec) {
  // A codec this encoder does not know is unavailable on every device; there is nothing to probe.
  const std::optional<GUID> guid = ResolveCodec(codec);
  if (!guid) {
    return false;
  }
  // Probing opens a real NVENC session, so each answer is kept. The cache is keyed by device
  // because NVENC capability is a property of one GPU, not of the host: a machine can pair an
  // encoder-less A100 with a display GPU that has one. It is keyed by codec too, since one GPU can
  // offer H.264 and not H.265, and by the GUID rather than the spelling so that "H265" and "HEVC"
  // share an entry. The mutex is held across the probe, so concurrent callers - one per encoded
  // topic, all starting at once - probe a given device and codec at most once between them.
  static std::mutex cache_mutex;
  using CodecKey = std::array<std::byte, sizeof(GUID)>;
  static std::map<std::pair<std::uint32_t, CodecKey>, bool> cache;
  const auto codec_key = std::bit_cast<CodecKey>(*guid);
  const std::scoped_lock lock(cache_mutex);
  const auto cached = cache.find({cuda_device, codec_key});
  if (cached != cache.end()) {
    return cached->second;
  }

  const bool result = ProbeEncoderSession(cuda_device, *guid);

  cache.try_emplace({cuda_device, codec_key}, result);
  return result;
}

NVVideoEncoder::NVVideoEncoder(uint32_t width, uint32_t height, const std::string& codec,
                               uint32_t bitrate, uint32_t frame_rate, uint32_t cuda_device)
    : width_(width),
      height_(height),
      codec_(codec),
      bitrate_(bitrate),
      frame_rate_(frame_rate),
      cuda_device_(cuda_device) {
  // Validate parameters
  if (width == 0 || height == 0) {
    throw std::invalid_argument("Width and height must be greater than 0");
  }
  if (bitrate == 0) {
    throw std::invalid_argument("Bitrate must be greater than 0");
  }
  if (frame_rate == 0) {
    throw std::invalid_argument("Frame rate must be greater than 0");
  }
}

NVVideoEncoder::~NVVideoEncoder() { cleanup(); }

bool NVVideoEncoder::initialize() {
  if (initialized_) {
    return true;
  }

  try {
    if (!setup_cuda()) {
      HOLOSCAN_LOG_ERROR("Failed to setup CUDA");
      throw std::runtime_error("Failed to setup CUDA");
    }

    if (!configure_encoder()) {
      HOLOSCAN_LOG_ERROR("Failed to configure encoder");
      throw std::runtime_error("Failed to configure encoder");
    }

    initialized_ = true;
    return true;
  } catch (const std::runtime_error& e) {
    HOLOSCAN_LOG_ERROR("Initialization failed: {}", e.what());
    cleanup();
    throw;  // Re-throw the exception instead of returning false
  }
}

bool NVVideoEncoder::setup_cuda() {
  // Initialize CUDA.
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to initialize CUDA");
    return false;
  }

  // Get the CUDA device.
  result = cuDeviceGet(&cu_device_, cuda_device_);
  if (result != CUDA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Unable to access GPU device {}.", cuda_device_);
    return false;
  }

  // Retain the primary context for the device.
  result = cuDevicePrimaryCtxRetain(&cu_context_, cu_device_);
  if (result != CUDA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to retain primary context");
    return false;
  }

  // Confirm the retained context can actually be made current, then put the stack back as it
  // was found. Nothing below needs it to stay current: every NvEncoderCuda entry point that
  // touches CUDA (AllocateInputBuffers, CopyToDeviceFrame, ReleaseCudaResources) pushes
  // cu_context_ around its own work, and the NVENC session API addresses the device through
  // the session handle rather than the current context.
  //
  // Leaving it pushed here would be a push and a pop on two different threads. CUDA context
  // stacks are per-thread, so the pop would not undo this push: it would pop whatever the
  // destroying thread happened to have current. That is exactly what happens when an encoder
  // is created lazily on a worker thread and destroyed from a shutdown path on another.
  result = cuCtxPushCurrent(cu_context_);
  if (result != CUDA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to push current context");
    return false;
  }
  // The pop is what makes this thread's stack whole again, so a failure to pop is a failure to
  // initialize: the context would still be on this thread's stack, the very state this function
  // exists to avoid leaving behind.
  result = cuCtxPopCurrent(nullptr);
  if (result != CUDA_SUCCESS) {
    HOLOSCAN_LOG_ERROR(
        "Failed to restore the thread's CUDA context stack after confirming the "
        "encoder context");
    return false;
  }

  return true;
}

bool NVVideoEncoder::configure_encoder() {
  // Create encoder with ARGB input format (4 channels).
  // TODO: Consider making input format configurable.
  // Try different format - maybe the issue is ARGB vs RGBA in the encoder.
  NV_ENC_BUFFER_FORMAT input_format = NV_ENC_BUFFER_FORMAT_ARGB;

  try {
    encoder_ = std::make_unique<NvEncoderCuda>(cu_context_, width_, height_, input_format, 0);
  } catch (const NVENCException& e) {
    HOLOSCAN_LOG_ERROR("Failed to create encoder: {}", e.what());
    return false;
  }

  // Get codec and preset GUIDs.
  std::string upper_codec = codec_;
  std::transform(upper_codec.begin(), upper_codec.end(), upper_codec.begin(), ::toupper);

  auto codec_it = CODEC_GUIDS.find(upper_codec);
  if (codec_it == CODEC_GUIDS.end()) {
    HOLOSCAN_LOG_ERROR("Unsupported codec: {}", codec_);
    return false;
  }
  GUID guidCodec = codec_it->second;

  // Use P3 preset for good balance.
  // TODO: Consider making preset configurable.
  // Currently hardcoded to P3 preset for good balance.
  // In the future, may want to support a parameter to adjust quality.
  GUID guidPreset = NV_ENC_PRESET_P3_GUID;

  // Initialize encoder parameters.
  NV_ENC_INITIALIZE_PARAMS initializeParams = {};
  NV_ENC_CONFIG encodeConfig = {};
  initializeParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
  encodeConfig.version = NV_ENC_CONFIG_VER;
  initializeParams.encodeConfig = &encodeConfig;

  // Create default encoder parameters.
  encoder_->CreateDefaultEncoderParams(&initializeParams, guidCodec, guidPreset,
                                       NV_ENC_TUNING_INFO_LOW_LATENCY);

  // Set frame rate.
  initializeParams.frameRateNum = frame_rate_;
  initializeParams.frameRateDen = 1;

  // Configure encoding parameters.
  encodeConfig.rcParams.averageBitRate = bitrate_;
  encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;  // Constant bitrate.
  encodeConfig.frameIntervalP = 1;

  // Configure buffer settings.
  encodeConfig.rcParams.vbvBufferSize =
      (encodeConfig.rcParams.averageBitRate * initializeParams.frameRateDen /
       initializeParams.frameRateNum) *
      2;  // 2 seconds buffer.
  encodeConfig.rcParams.maxBitRate = encodeConfig.rcParams.averageBitRate;
  encodeConfig.rcParams.vbvInitialDelay = encodeConfig.rcParams.vbvBufferSize;

  // Create the encoder.
  try {
    encoder_->CreateEncoder(&initializeParams);
  } catch (const NVENCException& e) {
    HOLOSCAN_LOG_ERROR("Failed to create encoder: {}", e.what());
    return false;
  }

  return true;
}

size_t NVVideoEncoder::encode_frame(CUdeviceptr input_frame, std::vector<uint8_t>& output_data) {
  return encode_frame_into(input_frame, output_data);
}

size_t NVVideoEncoder::encode_frame(CUdeviceptr input_frame, std::vector<std::byte>& output_data) {
  return encode_frame_into(input_frame, output_data);
}

template <typename ByteT>
size_t NVVideoEncoder::encode_frame_into(CUdeviceptr input_frame, std::vector<ByteT>& output_data) {
  if (!initialized_ || !encoder_) {
    return 0;
  }

  try {
    // Get input frame from encoder.
    const NvEncInputFrame* encoderInputFrame = encoder_->GetNextInputFrame();
    if (!encoderInputFrame) {
      HOLOSCAN_LOG_ERROR("Failed to get input frame");
      return 0;
    }

    // Copy frame data to GPU memory.
    //
    // The SDK spells every buffer address as a void*, for host and device memory alike: the
    // encoder's own input buffer comes back through inputPtr, and the source frame goes in as
    // pSrcFrame, which for CU_MEMORYTYPE_DEVICE it casts straight back to a CUdeviceptr
    // (NvEncoderCuda.cpp). Both are device addresses and are named as such here, once, so the
    // conversion from the SDK's type is the only one and the call itself reads in the right
    // types. Neither is dereferenced on the host.
    //
    // bit_cast rather than reinterpret_cast: it converts between the integer and the pointer of the
    // same width, and refuses to compile if they ever differ, which a reinterpret_cast does not.
    const auto encoder_input = std::bit_cast<CUdeviceptr>(encoderInputFrame->inputPtr);
    auto source_frame = std::bit_cast<void*>(input_frame);
    NvEncoderCuda::CopyToDeviceFrame(
        cu_context_, source_frame, 0, encoder_input, encoderInputFrame->pitch,
        encoder_->GetEncodeWidth(), encoder_->GetEncodeHeight(), CU_MEMORYTYPE_DEVICE,
        encoderInputFrame->bufferFormat, encoderInputFrame->chromaOffsets,
        encoderInputFrame->numChromaPlanes);

    // Encode the frame.
    std::vector<NvEncOutputFrame> vPacket;
    NV_ENC_PIC_PARAMS picParams = {};
    picParams.version = NV_ENC_PIC_PARAMS_VER;
    picParams.encodePicFlags =
        NV_ENC_PIC_FLAG_FORCEINTRA | NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;

    encoder_->EncodeFrame(vPacket, &picParams);

    // Process output.
    if (vPacket.empty()) {
      return 0;
    }

    // Copy encoded data to output buffer. The SDK owns vPacket and destroys it with this scope,
    // so the packet has to be copied out here whatever the caller's element type is.
    //
    // assign() over a reinterpreted pointer range rather than resize() then memcpy(): resize()
    // value-initializes, which would zero the whole buffer before the copy overwrote it. The
    // cast is reading one byte type's storage through another, which is exactly what std::byte
    // and unsigned char are both permitted to do, and it is a no-op for the uint8_t overload.
    const auto& packet = vPacket[0].frame;
    const auto* const first = reinterpret_cast<const ByteT*>(packet.data());
    output_data.assign(first, first + packet.size());

    return output_data.size();

  } catch (const NVENCException& e) {
    HOLOSCAN_LOG_ERROR("Encoding failed: {}", e.what());
    return 0;
  }
}

void NVVideoEncoder::cleanup() {
  if (encoder_) {
    // Pushed around the teardown rather than left current: this runs on whichever thread drops
    // the last reference, which is not in general the thread that called initialize().
    const bool pushed = cu_context_ != nullptr && cuCtxPushCurrent(cu_context_) == CUDA_SUCCESS;
    if (!pushed) {
      // A context that cannot be made current is one the encoder cannot release its buffers into,
      // and NvEncoderCuda's own teardown pushes it without checking and then pops unconditionally,
      // so destroying the encoder here would pop whatever context this thread has current -- the
      // stack corruption this change exists to prevent. The encoder is abandoned instead: its
      // session and device buffers stay allocated until the process exits, which is the lesser
      // harm, and the failure is reported loudly.
      HOLOSCAN_LOG_ERROR(
          "Unable to make the encoder's CUDA context current to tear the encoder down. The "
          "encoder is being abandoned rather than destroyed, so its session and device memory "
          "stay allocated until the process exits.");
      static_cast<void>(encoder_.release());
    } else {
      try {
        std::vector<NvEncOutputFrame> vPacket;
        encoder_->EndEncode(vPacket);
      } catch (const NVENCException& e) {
        HOLOSCAN_LOG_ERROR("Error during encoder cleanup: {}", e.what());
      }
      encoder_.reset();
      if (cuCtxPopCurrent(nullptr) != CUDA_SUCCESS) {
        HOLOSCAN_LOG_ERROR(
            "Failed to restore the thread's CUDA context stack after tearing the "
            "encoder down.");
      }
    }
  }

  if (cu_context_) {
    cuDevicePrimaryCtxRelease(cu_device_);
    cu_context_ = nullptr;
  }

  initialized_ = false;
}

}  // namespace nv_video_encoder
