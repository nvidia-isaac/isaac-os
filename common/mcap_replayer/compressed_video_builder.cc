/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/mcap_replayer/compressed_video_builder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/sensor_io/schemas/image_generated.h"
#include <flatbuffers/flatbuffers.h>
#include <fmt/format.h>

#ifdef ISAAC_ENABLE_NV_VIDEO_SDK
#include <nppi_color_conversion.h>
#include <nppi_data_exchange_and_initialization.h>

#include <array>
#include <exception>

#include "holoscan/logger/logger.hpp"
#include <cuda.h>

#include "common/mcap_logger/compressed_video_proto.h"
#include "common/mcap_logger/gpu_scope.h"
#include "common/nv_video_decoder/nv_video_decoder.hpp"
#endif

namespace isaac::os::mcap_replayer {

namespace {

constexpr std::uint32_t kRgbChannels = 3;
constexpr std::uint32_t kRgbaChannels = 4;

using Pixels = std::unique_ptr<std::byte[]>;

#ifdef ISAAC_ENABLE_NV_VIDEO_SDK

// Alpha for RGBA output: video carries none, so every pixel is opaque.
constexpr Npp8u kOpaqueAlpha = 255;

// A device allocation that grows to the largest frame seen and is reused for every frame after, as
// the logger's staging buffers are, so steady state does no cuMemAlloc or cuMemFree. The decoder's
// context must be current for every call that touches one.
struct DeviceBuffer {
  CUdeviceptr address = 0;
  std::size_t capacity = 0;
};

void free_buffer(DeviceBuffer& buffer) {
  if (buffer.address != 0) {
    cuMemFree(buffer.address);
  }
  buffer = {};
}

[[nodiscard]] bool ensure_capacity(DeviceBuffer& buffer, std::size_t bytes) {
  if (bytes <= buffer.capacity) {
    return true;
  }
  free_buffer(buffer);
  if (cuMemAlloc(&buffer.address, bytes) != CUDA_SUCCESS) {
    buffer = {};
    return false;
  }
  buffer.capacity = bytes;
  return true;
}

// Decodes the foxglove.CompressedVideo messages of one topic, one at a time, in recorded order.
// Every frame the logger records is an independently decodable IDR with in-band parameter sets, so
// one message in gives one frame out; the decoder is created on the first message, from its
// `format`, and reused after. Not thread safe, and may be destroyed on any thread.
class DecodeSession {
 public:
  explicit DecodeSession(std::uint32_t cuda_device) : cuda_device_(cuda_device) {}

  // Frees the colour-conversion buffers while the decoder, which owns their context, still exists.
  // This may run on any thread, so the context is made current for the frees rather than assumed.
  ~DecodeSession() {
    if (planar_.address == 0 && rgb_.address == 0 && rgba_.address == 0) {
      return;
    }
    const auto context =
        mcap_logger::PushContext(decoder_ ? decoder_->get_cuda_context() : nullptr);
    if (!context) {
      HOLOSCAN_LOG_ERROR(
          "Unable to make the video decoder's GPU context current to free its colour-conversion "
          "buffers. They stay allocated until the process exits.");
      return;
    }
    free_buffer(rgb_);
    free_buffer(rgba_);
    free_buffer(planar_);
  }
  DecodeSession(const DecodeSession&) = delete;
  DecodeSession& operator=(const DecodeSession&) = delete;

  // Decodes one recorded message into width x height host pixels with `channels` channels (3 for
  // RGB, 4 for RGBA with alpha 255). The error is a reason in words fit for a terminal.
  holoscan::expected<Pixels, std::string> decode(std::span<const std::byte> payload,
                                                 std::uint32_t width, std::uint32_t height,
                                                 std::uint32_t channels) {
    const auto fields = mcap_logger::ReadCompressedVideo(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size()));
    if (!fields.has_value()) {
      return holoscan::make_unexpected(std::string("its video message is damaged"));
    }
    if (fields->data.empty()) {
      return holoscan::make_unexpected(std::string("its video message holds no video data"));
    }

    if (auto ready = ensure_decoder(fields->format); !ready) {
      return holoscan::make_unexpected(std::move(ready.error()));
    }

    // The first frame is the one taken if a stream ever returns more than one.
    std::vector<std::uint8_t*> frames;
    const int frame_count =
        decoder_->decode_frame(fields->data.data(), fields->data.size(), frames);
    if (frame_count < 0) {
      return holoscan::make_unexpected(std::string("the video decoder reported an error"));
    }
    if (frame_count == 0) {
      return holoscan::make_unexpected(std::string("the video decoder produced no frame for it"));
    }
    if (decoder_->get_width() != static_cast<int>(width) ||
        decoder_->get_height() != static_cast<int>(height)) {
      return holoscan::make_unexpected(
          fmt::format("the video decodes to {}x{} but its image metadata says {}x{}",
                      decoder_->get_width(), decoder_->get_height(), width, height));
    }
    return convert(frames.front(), width, height, channels);
  }

 private:
  // Creates the decoder for `format` on first use. Creating it failing is latched: the same reason
  // every time, without retrying (and re-logging inside the decoder) at message rate.
  // The codec is probed first (the answer is cached) so a codec this GPU cannot decode, or one this
  // library does not know, is one clear latched reason rather than a decoder that starts and then
  // fails inside its parser on every frame.
  holoscan::expected<void, std::string> ensure_decoder(std::string_view format) {
    if (decoder_) {
      return {};
    }
    if (!creation_error_.empty()) {
      return holoscan::make_unexpected(creation_error_);
    }
    const std::string format_name(format);
    if (!nv_video_decoder::NVVideoDecoder::is_available(cuda_device_, format_name)) {
      creation_error_ =
          fmt::format("GPU {} has no hardware video decoder for '{}'", cuda_device_, format_name);
      return holoscan::make_unexpected(creation_error_);
    }
    try {
      auto decoder = std::make_unique<nv_video_decoder::NVVideoDecoder>(format_name, cuda_device_);
      decoder->initialize();
      decoder_ = std::move(decoder);
    } catch (const std::exception& error) {
      creation_error_ = fmt::format("the video decoder for '{}' could not be created on GPU {}: {}",
                                    format_name, cuda_device_, error.what());
      return holoscan::make_unexpected(creation_error_);
    }
    return {};
  }

  // NV12 on the device to RGB or RGBA on the host. NV12 is two planes: full-resolution luma,
  // then interleaved half-resolution chroma, the second starting pitch * height bytes in. The
  // decoder reuses its frame buffer on the next decode_frame(), so this finishes first.
  holoscan::expected<Pixels, std::string> convert(const std::uint8_t* nv12, std::uint32_t width,
                                                  std::uint32_t height, std::uint32_t channels) {
    const int pitch = decoder_->get_pitch();
    if (pitch <= 0) {
      return holoscan::make_unexpected(std::string("the video decoder reported no frame pitch"));
    }
    // The NV12 frame belongs to the decoder's context, and so do the buffers below.
    const auto context = mcap_logger::PushContext(decoder_->get_cuda_context());
    if (!context) {
      return holoscan::make_unexpected(
          std::string("the video decoder's GPU context could not be made current"));
    }
    const auto npp_context = mcap_logger::MakeNppStreamContext(cuda_device_);
    if (!npp_context.has_value()) {
      return holoscan::make_unexpected(
          fmt::format("GPU {} could not be described to NPP for colour conversion", cuda_device_));
    }

    const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
    const std::size_t chroma_count = static_cast<std::size_t>(width / 2) * (height / 2);
    const bool wants_alpha = channels == kRgbaChannels;
    if (!ensure_capacity(planar_, pixel_count + 2 * chroma_count) ||
        !ensure_capacity(rgb_, pixel_count * kRgbChannels) ||
        (wants_alpha && !ensure_capacity(rgba_, pixel_count * kRgbaChannels))) {
      return holoscan::make_unexpected(
          std::string("GPU memory for colour conversion could not be allocated"));
    }

    const NppiSize roi = {static_cast<int>(width), static_cast<int>(height)};
    const auto rgb_step = static_cast<int>(width * kRgbChannels);
    // NV12 is converted to planar 4:2:0 first and that to RGB, not with the one-call
    // nppiNV12ToRGB variants: on this NPP those treat luma as full range and apply a chroma matrix
    // that is neither BT.601 nor BT.709, so a recorded (200, 100, 50) came back as (176, 110, 47).
    // The planar route reproduces the BT.601 limited-range stream NVENC writes to within one count.
    const std::array<const Npp8u*, 2> nv12_planes = {
        nv12, nv12 + static_cast<std::size_t>(pitch) * height};
    Npp8u* const luma = mcap_logger::DevicePointer<Npp8u>(planar_.address);
    std::array<Npp8u*, 3> planar = {luma, luma + pixel_count, luma + pixel_count + chroma_count};
    const std::array<const Npp8u*, 3> planar_read = {planar[0], planar[1], planar[2]};
    // NPP takes the step array as a mutable int*, so this one is not const.
    std::array<int, 3> planar_steps = {static_cast<int>(width), static_cast<int>(width / 2),
                                       static_cast<int>(width / 2)};
    if (nppiNV12ToYUV420_8u_P2P3R_Ctx(nv12_planes.data(), pitch, planar.data(), planar_steps.data(),
                                      roi, *npp_context) != NPP_SUCCESS ||
        nppiYCbCr420ToRGB_8u_P3C3R_Ctx(planar_read.data(), planar_steps.data(),
                                       mcap_logger::DevicePointer<Npp8u>(rgb_.address), rgb_step,
                                       roi, *npp_context) != NPP_SUCCESS) {
      return holoscan::make_unexpected(std::string("colour conversion on the GPU failed"));
    }
    CUdeviceptr source = rgb_.address;
    if (wants_alpha) {
      // Index 3 of the destination order selects the constant, so this appends an opaque alpha.
      constexpr std::array<int, 4> kKeepOrderAddAlpha = {0, 1, 2, 3};
      if (nppiSwapChannels_8u_C3C4R_Ctx(mcap_logger::DevicePointer<const Npp8u>(rgb_.address),
                                        rgb_step, mcap_logger::DevicePointer<Npp8u>(rgba_.address),
                                        static_cast<int>(width * kRgbaChannels), roi,
                                        kKeepOrderAddAlpha.data(), kOpaqueAlpha,
                                        *npp_context) != NPP_SUCCESS) {
        return holoscan::make_unexpected(std::string("adding the alpha channel on the GPU failed"));
      }
      source = rgba_.address;
    }

    const std::size_t byte_count = pixel_count * channels;
    // Left uninitialized: the copy below fills all of it.
    Pixels pixels = std::make_unique_for_overwrite<std::byte[]>(byte_count);
    if (cuMemcpyDtoH(pixels.get(), source, byte_count) != CUDA_SUCCESS) {
      return holoscan::make_unexpected(std::string("copying the decoded frame to the host failed"));
    }
    return pixels;
  }

  const std::uint32_t cuda_device_;
  std::unique_ptr<nv_video_decoder::NVVideoDecoder> decoder_;
  // Why creating the decoder failed, once it has. Construction already refuses a topic whose codec
  // the GPU cannot decode, so this is the fallback for a later message that names a different
  // codec, or a decoder that cannot be created: later frames drop with this reason instead of
  // retrying at message rate.
  std::string creation_error_;
  // Device buffers for the colour conversion (planar 4:2:0 scratch, RGB, RGBA), held for the
  // decoder's life (see DeviceBuffer).
  DeviceBuffer planar_;
  DeviceBuffer rgb_;
  DeviceBuffer rgba_;
};

#else  // !ISAAC_ENABLE_NV_VIDEO_SDK

class DecodeSession {
 public:
  explicit DecodeSession(std::uint32_t) {}
  holoscan::expected<Pixels, std::string> decode(std::span<const std::byte>, std::uint32_t,
                                                 std::uint32_t, std::uint32_t) {
    return holoscan::make_unexpected(std::string(
        "this build has no NVIDIA Video Codec SDK, so it cannot decode recorded video"));
  }
};

#endif  // ISAAC_ENABLE_NV_VIDEO_SDK

}  // namespace

std::optional<std::string> video_decode_unavailable_reason(std::uint32_t cuda_device,
                                                           std::string_view codec) {
#ifdef ISAAC_ENABLE_NV_VIDEO_SDK
  const std::string codec_name(codec);
  if (nv_video_decoder::NVVideoDecoder::is_available(cuda_device, codec_name)) {
    return std::nullopt;
  }
  return fmt::format("GPU {} has no hardware video decoder for '{}'", cuda_device, codec_name);
#else
  static_cast<void>(cuda_device);
  static_cast<void>(codec);
  return "this build has no NVIDIA Video Codec SDK";
#endif
}

MessageBuilder make_compressed_video_builder(std::uint32_t cuda_device) {
  // Copy-constructible, as std::function requires, and shared by every copy of this builder.
  auto session = std::make_shared<DecodeSession>(cuda_device);
  return
      [session](
          std::span<const std::byte> video, std::span<const std::byte> sidecar,
          std::uint64_t log_time_ns) -> holoscan::expected<std::unique_ptr<Message>, std::string> {
        // Bytes off disk: verified before reading, as the tensor builder does.
        flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t*>(sidecar.data()),
                                       sidecar.size());
        if (!holoscan::schema::VerifyImageBuffer(verifier)) {
          return holoscan::make_unexpected(
              std::string("its image metadata is damaged, or is not image metadata at all"));
        }
        const auto* image = holoscan::schema::GetImage(sidecar.data());
        std::uint32_t channels = 0;
        switch (image->encoding()) {
          case holoscan::schema::ImageEncoding_RGB8:
            channels = kRgbChannels;
            break;
          case holoscan::schema::ImageEncoding_RGBA8:
            channels = kRgbaChannels;
            break;
          default:
            return holoscan::make_unexpected(fmt::format(
                "its image metadata says '{}', but video is only ever recorded from RGB8 or RGBA8",
                holoscan::schema::EnumNameImageEncoding(image->encoding())));
        }

        // 4:2:0 video halves the chroma in each direction, so an odd size has no whole chroma
        // sample at its edge, and the decoder reports an odd width rounded up.
        if (image->width() % 2 != 0 || image->height() % 2 != 0) {
          return holoscan::make_unexpected(
              fmt::format("video replay supports only even frame sizes, but its image metadata "
                          "says {}x{}",
                          image->width(), image->height()));
        }

        const auto width = static_cast<std::uint32_t>(image->width());
        const auto height = static_cast<std::uint32_t>(image->height());
        auto pixels = session->decode(video, width, height, channels);
        if (!pixels) {
          return holoscan::make_unexpected(std::move(pixels.error()));
        }

        const std::vector<std::int64_t> shape{
            static_cast<std::int64_t>(static_cast<std::size_t>(width) * height * channels)};
        auto tensor =
            detail::adopt_host_buffer(shape, DLDataType{kDLUInt, 8U, 1U}, std::move(*pixels));
        if (!tensor) {
          return holoscan::make_unexpected(std::string(tensor.error()));
        }

        auto message = std::make_unique<TensorMessage>();
        message->metadata.assign(sidecar.begin(), sidecar.end());
        message->tensors.push_back(std::move(*tensor));
        message->log_time_ns = log_time_ns;
        return message;
      };
}

}  // namespace isaac::os::mcap_replayer
