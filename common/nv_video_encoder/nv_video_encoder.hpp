/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "NvEncoder/NvEncoderCuda.h"
#include "cuda.h"

namespace nv_video_encoder {

/**
 * @brief NV video encoder using NVIDIA Video Codec SDK
 *
 * This class provides basic video encoding functionality for h264/HEVC formats.
 * It's designed to be lightweight and focused on core encoding operations.
 */
class NVVideoEncoder {
 public:
  /**
   * @brief Probe whether the host has at least one CUDA device with a usable
   *        NVENC engine.
   *
   * The dGPU driver package installs libnvidia-encode.so on every host
   * regardless of the GPU SKU plugged in, so a successful library load is
   * not sufficient to know NVENC will work at runtime. GA100 (A100) and
   * GH100 (H100) silicon ship 0 NVENC engines and return
   * NV_ENC_ERR_UNSUPPORTED_DEVICE when nvEncOpenEncodeSessionEx is called.
   *
   * This helper opens and immediately closes a minimal NvEncoderCuda
   * session on the given CUDA device inside a try-catch. The constructor calls
   * nvEncOpenEncodeSessionEx, which is the actual API that returns
   * NV_ENC_ERR_UNSUPPORTED_DEVICE on NVENC-less silicon - the same call
   * the production encoder makes, so the probe and the production path
   * disagree only on parameters, not on capability detection.
   *
   * NVML alternatives were considered (nvmlDeviceGetEncoderCapacity)
   * but found unreliable: the API is designed as a utilization
   * percentage, not a presence indicator, and reports a non-zero
   * capacity on cards that have no NVENC silicon at all.
   *
   * The result is cached for the life of the process, per device and codec, on
   * first call: a failed probe stays failed until the process restarts. The
   * cache is safe to consult from several threads at once. The probe uses
   * cuDevicePrimaryCtxRetain/Release rather than cuCtxCreate so it
   * works across CUDA toolkit major versions (CUDA 13+ headers
   * redirect cuCtxCreate to a 4-arg cuCtxCreate_v4).
   *
   * A session opening is not enough on its own: NVENC silicon differs in
   * which codecs it offers, and an H.264-only part opens a session and
   * then refuses an H.265 configuration on the first frame. So the probe
   * also asks the open session which encode GUIDs it supports, and the
   * answer is per device and codec.
   *
   * @param cuda_device CUDA device ordinal to probe. An ordinal this
   *        host does not have is reported as unavailable rather than
   *        as an error, so a caller can treat a mistyped device index
   *        the same way it treats NVENC-less silicon. Deliberately has
   *        no default: a caller that probes one device and then encodes
   *        on another gets a green probe and records nothing, so every
   *        call must name the device it is asking about.
   * @param codec Codec to ask about, in any case ("h264", "H265" or
   *        "HEVC"; the last two are the same codec). Defaults to H.264,
   *        which every NVENC part offers, so the default asks "is there
   *        an encoder at all". A caller about to encode another codec
   *        must name it. A codec this encoder does not know is reported
   *        as unavailable.
   * @return true if NVENC silicon is present on that device and a
   *         session can be opened that supports the codec.
   */
  static bool is_available(std::uint32_t cuda_device, const std::string& codec = "H264");

  /**
   * @brief Constructor
   * @param width Video frame width
   * @param height Video frame height
   * @param codec Codec type ("h264" or "HEVC")
   * @param bitrate Target bitrate in bits per second
   * @param frame_rate Frame rate (fps)
   * @param cuda_device CUDA device ordinal
   */
  NVVideoEncoder(uint32_t width, uint32_t height, const std::string& codec = "h264",
                 uint32_t bitrate = 10'000'000, uint32_t frame_rate = 30, uint32_t cuda_device = 0);

  /**
   * @brief Destructor
   */
  ~NVVideoEncoder();

  /**
   * @brief Initialize the encoder
   * @return true if successful
   * @throws std::runtime_error if initialization fails
   */
  bool initialize();

  /**
   * @brief Encode a video frame
   * @param input_frame Device address of the input frame. Always device memory: the frame is
   *        copied into the encoder's input buffer as CU_MEMORYTYPE_DEVICE, so a host pointer
   *        cannot be passed here
   * @param output_data Output buffer for encoded data
   * @return Size of encoded data in bytes, 0 if failed
   */
  size_t encode_frame(CUdeviceptr input_frame, std::vector<uint8_t>& output_data);

  /**
   * @brief Encode a video frame into a byte buffer
   *
   * Identical to the overload above except for the output element type. Exists so a caller whose
   * destination is already std::byte -- foxglove::schemas::CompressedVideo::data, for one -- can
   * be filled directly and then moved from, instead of receiving a vector<uint8_t> it has to copy
   * again to change type. The packet the SDK produces has to be copied out once either way; this
   * is about not copying it twice.
   *
   * @param input_frame Device address of the input frame. Always device memory: the frame is
   *        copied into the encoder's input buffer as CU_MEMORYTYPE_DEVICE, so a host pointer
   *        cannot be passed here
   * @param output_data Output buffer for encoded data
   * @return Size of encoded data in bytes, 0 if failed
   */
  size_t encode_frame(CUdeviceptr input_frame, std::vector<std::byte>& output_data);

  /**
   * @brief Get CUDA context
   * @return CUDA context
   */
  CUcontext get_cuda_context() const { return cu_context_; }

  /**
   * @brief Get encoder width
   * @return Encoder width in pixels
   */
  uint32_t get_encode_width() const { return width_; }

  /**
   * @brief Get encoder height
   * @return Encoder height in pixels
   */
  uint32_t get_encode_height() const { return height_; }

 private:
  // Configuration parameters
  const uint32_t width_;
  const uint32_t height_;
  const std::string codec_;
  const uint32_t bitrate_;
  const uint32_t frame_rate_;
  const uint32_t cuda_device_;

  // CUDA resources
  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_ = 0;
  // Encoder
  std::unique_ptr<NvEncoderCuda> encoder_;
  bool initialized_ = false;

  // Helper methods
  // Shared body of the two encode_frame() overloads, which differ only in the element type they
  // copy the SDK's packet into. Defined in the .cpp and instantiated only there.
  template <typename ByteT>
  size_t encode_frame_into(CUdeviceptr input_frame, std::vector<ByteT>& output_data);

  bool setup_cuda();
  bool configure_encoder();
  void cleanup();
};

}  // namespace nv_video_encoder
