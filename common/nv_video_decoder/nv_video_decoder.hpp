/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cuda.h"

// Forward declaration — full definition is in nv_video_decoder.cpp via NvDecoder/NvDecoder.h.
class NvDecoder;

namespace nv_video_decoder {

/**
 * @brief NV video decoder using NVIDIA Video Codec SDK
 *
 * This class provides basic video decoding functionality for H264/HEVC formats.
 * It is the symmetric counterpart to nv_video_encoder::NVVideoEncoder.
 *
 * Input:  H264/HEVC Annex-B bitstream (CPU memory)
 * Output: Decoded frames in NV12 format (GPU device pointers)
 *
 * A single call to decode_frame may return zero or more frames depending on
 * decoder buffering.  Callers must process all returned frames before the next
 * call to decode_frame, as the underlying GPU buffers are reused.
 */
class NVVideoDecoder {
 public:
  /**
   * @brief Probe whether `cuda_device` has a usable NVDEC engine for `codec`.
   *
   * Uses cuvidGetDecoderCaps (the canonical "is NVDEC silicon present and does it support codec
   * X" call) on the device's primary context, with 4:2:0 8-bit as the baseline query. Both
   * bIsSupported and nNumNVDECs must be non-zero. cuDevicePrimaryCtxRetain is used instead of
   * cuCtxCreate because CUDA 13 headers redirect cuCtxCreate to a four-argument form.
   *
   * Cached per (device, codec) for the life of the process.
   *
   * @param cuda_device CUDA device ordinal
   * @param codec "h264", "h265" or "hevc", case-insensitive
   * @return false for an unknown codec or device, or when the engine does not support it
   */
  static bool is_available(uint32_t cuda_device = 0, const std::string& codec = "h264");

  /**
   * @brief Constructor
   * @param codec Codec type ("h264" or "HEVC")
   * @param cuda_device CUDA device ordinal
   */
  explicit NVVideoDecoder(const std::string& codec = "h264", uint32_t cuda_device = 0);

  /**
   * @brief Destructor
   */
  ~NVVideoDecoder();

  /**
   * @brief Initialize the decoder
   * @return true if successful
   * @throws std::runtime_error if initialization fails
   */
  bool initialize();

  /**
   * @brief Decode a compressed video packet
   *
   * Every call passes one whole picture (an access unit). Each packet is flagged to the decoder
   * as the end of a picture, so a picture split across several calls is not supported. Passing
   * (nullptr, 0) flushes the decoder and returns the frames it still buffers.
   *
   * @param input_data  Annex-B bitstream data (CPU memory)
   * @param input_size  Size of input data in bytes
   * @param frame_ptrs  Output: GPU device pointers to decoded NV12 frames.
   *                    Each pointer is valid until the next decode_frame call.
   * @return Number of decoded frames available in frame_ptrs, 0 if no frame
   *         is ready yet (decoder may buffer internally), -1 on error
   */
  int decode_frame(const uint8_t* input_data, size_t input_size, std::vector<uint8_t*>& frame_ptrs);

  /**
   * @brief Get CUDA context
   * @return CUDA context
   */
  CUcontext get_cuda_context() const { return cu_context_; }

  /**
   * @brief Get decoded frame width (valid after first successful decode)
   * @return Frame width in pixels, 0 if not yet decoded
   */
  int get_width() const;

  /**
   * @brief Get decoded frame height (valid after first successful decode)
   * @return Frame height in pixels, 0 if not yet decoded
   */
  int get_height() const;

  /**
   * @brief Get the device frame pitch in bytes (valid after first successful decode)
   * @return Pitch in bytes, 0 if not yet decoded
   */
  int get_pitch() const;

 private:
  const std::string codec_;
  const uint32_t cuda_device_;

  // CUDA resources
  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_ = 0;

  // Decoder
  std::unique_ptr<NvDecoder> decoder_;
  bool initialized_ = false;

  bool setup_cuda();
  bool configure_decoder();
  void cleanup();
};

}  // namespace nv_video_decoder
