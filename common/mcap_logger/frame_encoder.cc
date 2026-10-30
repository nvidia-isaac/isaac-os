/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/mcap_logger/frame_encoder.h"

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "holoscan/logger/logger.hpp"
#include <fmt/format.h>

#include "common/mcap_logger/failure_log_schedule.h"

#ifdef ISAAC_ENABLE_NV_VIDEO_SDK
#include <nppi_data_exchange_and_initialization.h>

#include <cuda.h>

#include "common/image_schema/image_schema.h"
#include "common/mcap_logger/gpu_scope.h"
#include "common/nv_video_encoder/nv_video_encoder.hpp"
#endif

namespace isaac::os::mcap_logger {

FrameEncoder::FrameEncoder(VideoEncodeOptions options, std::string_view topic)
    : options_(std::move(options)),
      topic_label_(topic.empty() ? std::string("an unnamed topic")
                                 : fmt::format("topic '{}'", topic)) {
  // Validated before the probe so a zero bitrate or frame rate throws on every host, including
  // one with no encoder at all, instead of being reported as an absent encoder.
  ValidateVideoEncodeOptions(options_);
  if (!hardware_available(options_.cuda_device, options_.codec)) {
    throw std::runtime_error(fmt::format(
        "The MCAP logger's {} asks for {} hardware video encoding on GPU {}, but that GPU is "
        "missing or has no usable hardware video encoder for it. Set a cuda_device that has one, "
        "or attach the topic without video encoding to record raw tensors instead.",
        topic_label_, FoxgloveCodecFormat(options_.codec), options_.cuda_device));
  }
}

std::atomic<std::uint64_t> FrameEncoder::frames_read_in_place_{0};

std::uint64_t FrameEncoder::frames_read_in_place() { return frames_read_in_place_.load(); }

void FrameEncoder::log_failure(std::string_view message) {
  ++failure_count_;
  if (IsFailureReportDue(failure_count_)) {
    HOLOSCAN_LOG_ERROR("{} ({} frames dropped so far on {}.)", message, failure_count_,
                       topic_label_);
  }
}

#ifdef ISAAC_ENABLE_NV_VIDEO_SDK

namespace {

// The encoder's input buffer is word-ordered ARGB: within each 32-bit pixel blue occupies the
// lowest byte, then green, then red, then alpha. In byte order that is B, G, R, A, while both
// source encodings this accepts are red-first, so red and blue have to be exchanged. The
// identity order {0, 1, 2, 3} would compile and encode just as happily while recording every
// frame with its red and blue channels swapped.
constexpr std::array<int, 4> kRedBlueSwapOrder = {2, 1, 0, 3};
constexpr Npp8u kOpaqueAlpha = 255;
constexpr std::uint32_t kArgbBytesPerPixel = 4;

// How many device buffers encoders have ever allocated for staging, across all of them. Only for
// FrameEncoder::staging_allocations(): no assertion on a recording can otherwise tell a session
// that reuses its buffers from one that reallocates them on every frame. A function-local static,
// not a namespace-scope variable: it has to be mutable, and a mutable global is what the rule
// against non-const globals exists to prevent.
std::atomic<std::uint64_t>& StagingAllocationCount() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}

// The header holds these buffers as unsigned long long so it need not include the CUDA headers;
// this is what makes handing them to the driver as a CUdeviceptr correct.
static_assert(std::is_same_v<CUdeviceptr, unsigned long long>);

// Allocates `buffer` if the session has not yet. A session's frame size is fixed by its first frame
// -- a frame of another size is refused, not re-encoded -- so the size asked for is the same on
// every call, and allocating and freeing per frame would only add a driver call, and an implicit
// synchronization, to each one. Needs the encoder's context current. Left untouched on failure: a
// failed allocation leaves the output pointer unspecified, and handing that to cuMemFree() would be
// worse than the failure it reports.
[[nodiscard]] bool EnsureDeviceBuffer(CUdeviceptr& buffer, std::size_t bytes) {
  if (buffer != 0) {
    return true;
  }
  CUdeviceptr allocated = 0;
  if (cuMemAlloc(&allocated, bytes) != CUDA_SUCCESS) {
    return false;
  }
  StagingAllocationCount().fetch_add(1);
  buffer = allocated;
  return true;
}

// Where a staged copy of the frame lands: the ARGB output itself for four-channel frames, which
// are then swapped in place, and a buffer of its own for three-channel frames, since the
// conversion widens them and cannot read and write the same pixels. That second buffer is only
// made for a session that has such a frame, because a frame NPP can read where it lies, or a
// four-channel one, never needs it.
[[nodiscard]] bool ReserveLanding(std::uint32_t channels, std::size_t bytes, CUdeviceptr argb,
                                  CUdeviceptr& landing_buffer, CUdeviceptr& landing) {
  if (channels == kArgbBytesPerPixel) {
    landing = argb;
    return true;
  }
  if (!EnsureDeviceBuffer(landing_buffer, bytes)) {
    return false;
  }
  landing = landing_buffer;
  return true;
}

// The channel count of a source encoding this encoder accepts, or nullopt for any other.
//
// Sizing is image_schema's to say and is asked of it rather than restated here. Which encodings
// are accepted is this encoder's own policy, and deliberately narrower than what image_schema can
// size: the encoder consumes ARGB and the matching decoder returns NV12, so only RGB8 and RGBA8
// survive a record-then-replay round trip. BGR8, BGRA8 and MONO8 all have a defined size and are
// refused anyway -- a BGR frame sent down the RGB path would come back with red and blue swapped.
// Both accepted encodings hold one byte per channel, so bytes per pixel is the channel count.
[[nodiscard]] std::optional<std::uint32_t> EncodableChannelCount(
    holoscan::schema::ImageEncoding encoding) {
  if (encoding != holoscan::schema::ImageEncoding_RGB8 &&
      encoding != holoscan::schema::ImageEncoding_RGBA8) {
    return std::nullopt;
  }
  const auto bytes_per_pixel = image_schema::bytes_per_pixel(encoding);
  return bytes_per_pixel ? std::optional<std::uint32_t>(*bytes_per_pixel) : std::nullopt;
}

// True when the encoder can reach `frame` where it lies, which is the only case it may be
// converted in place instead of being staged through a copy.
//
// NPP launches its kernel in the encoder's context and does not validate the addresses it is
// handed, so a pointer that context cannot resolve is not a launch-time error: the conversion
// reports NPP_SUCCESS and the fault surfaces afterwards as a sticky CUDA error that poisons the
// encoder's context for the rest of the run. A staging copy has no such limit, and is always
// correct and only ever slower, so every uncertain answer -- a failed query included -- is a
// false.
//
// The driver is asked the question directly rather than having it inferred from how the memory
// was allocated. The encoder's context is already current by the time this runs, so the access
// flags are reported for the device backing that context -- which is the device NPP will launch
// on. Device granularity is the right granularity here: under unified addressing reachability is
// a property of the device, not of the context that happened to allocate.
//
// One caveat that matters if this is ever reused: the flags are only checked for *some* access,
// not for write access. Measured on this host, the frames a live pipeline produces report READ
// alone, which is sufficient because the conversions only ever read the source -- but this test
// would wrongly admit such a frame to anything that wanted to write through the pointer.
//
// A frame on a different GPU with no peer access should report NONE and be staged. That is the
// documented behaviour rather than something measured here; it needs a second GPU to confirm.
bool IsAddressableByEncoder(CUdeviceptr frame) {
  unsigned int flags = 0;
  if (cuPointerGetAttribute(&flags, CU_POINTER_ATTRIBUTE_ACCESS_FLAGS, frame) != CUDA_SUCCESS) {
    return false;
  }
  return flags != CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE;
}

// Swaps `source`'s channels into `argb`, as ARGB. `source` is the caller's own tensor, when NPP can
// read it where it lies, or a staged copy of it. Both the callers and this function rely on that
// distinction, so it is worth being exact about what each conversion may do.
//
// The out-of-place NPP entry points read `source` and write `argb`, whose ranges must not
// overlap: they do not, being the caller's tensor or a private copy, and a buffer allocated for
// this frame. Reading is also all they may do to a tensor. It is refcounted and shared with every
// other consumer of the same sample, so swapping its channels in place would corrupt what those
// consumers see. The one in-place swap, the four-channel staged path, is recognised by
// `source == argb`: the copy already sits in our own output buffer, where swapping in place is
// the cheaper finish because it adds no second allocation to read from.
[[nodiscard]] bool SwapToArgb(std::uint32_t channels, std::uint32_t width, std::uint32_t height,
                              CUdeviceptr source, CUdeviceptr argb,
                              const NppStreamContext& npp_context) {
  const NppiSize roi = {static_cast<int>(width), static_cast<int>(height)};
  const auto argb_step = static_cast<int>(width * kArgbBytesPerPixel);
  NppStatus status = NPP_SUCCESS;
  if (channels != kArgbBytesPerPixel) {
    // Adds the opaque alpha channel the encoder's four-byte pixels need in the same pass.
    status = nppiSwapChannels_8u_C3C4R_Ctx(DevicePointer<const Npp8u>(source),
                                           static_cast<int>(width * channels),
                                           DevicePointer<Npp8u>(argb), argb_step, roi,
                                           kRedBlueSwapOrder.data(), kOpaqueAlpha, npp_context);
  } else if (source == argb) {
    status = nppiSwapChannels_8u_C4IR_Ctx(DevicePointer<Npp8u>(argb), argb_step, roi,
                                          kRedBlueSwapOrder.data(), npp_context);
  } else {
    status = nppiSwapChannels_8u_C4R_Ctx(DevicePointer<const Npp8u>(source), argb_step,
                                         DevicePointer<Npp8u>(argb), argb_step, roi,
                                         kRedBlueSwapOrder.data(), npp_context);
  }
  return status == NPP_SUCCESS;
}

// Copies a device-resident frame the encoder could not read where it lies into staging, and
// converts that copy. The unified cuMemcpy rather than cuMemcpyDtoD: this path is reached
// precisely when the encoder could not address the frame, and the most likely reason is a frame
// on a different GPU -- the configuration someone lands on by encoding on a secondary card, which
// is a reasonable thing to want when the GPU running the pipeline has no NVENC engine at all
// (A100 and H100 ship NVDEC without it). cuMemcpyDtoD resolves both addresses in the current
// context, so it is the wrong call for the one case this path exists to serve. cuMemcpy infers
// each side's device from the pointer under unified addressing and copies between them, staging
// through the host itself when the two devices have no peer mapping. For a source on the
// encoder's own device -- what every test here exercises -- the two are equivalent.
[[nodiscard]] bool ConvertDeviceFrame(CUdeviceptr frame, std::uint32_t channels,
                                      std::uint32_t width, std::uint32_t height, CUdeviceptr argb,
                                      CUdeviceptr& landing_buffer,
                                      const NppStreamContext& npp_context) {
  const std::size_t bytes = std::size_t{width} * height * channels;
  CUdeviceptr landing = 0;
  return ReserveLanding(channels, bytes, argb, landing_buffer, landing) &&
         cuMemcpy(landing, frame, bytes) == CUDA_SUCCESS &&
         SwapToArgb(channels, width, height, landing, argb, npp_context);
}

// Copies a host-resident frame into staging and converts that copy. NPP cannot read host memory,
// so a host frame is always staged. The copy is named for its source rather than inferred,
// because the driver rejects a host-to-device copy handed a device address and the reverse
// instead of sorting it out itself.
//
// The frame is a span, not a pointer: it carries the number of bytes to copy, which the caller has
// already checked against what the tensor actually holds, so this does not work out a length of
// its own and cannot disagree with it.
[[nodiscard]] bool ConvertHostFrame(std::span<const std::byte> frame, std::uint32_t channels,
                                    std::uint32_t width, std::uint32_t height, CUdeviceptr argb,
                                    CUdeviceptr& landing_buffer,
                                    const NppStreamContext& npp_context) {
  CUdeviceptr landing = 0;
  return ReserveLanding(channels, frame.size(), argb, landing_buffer, landing) &&
         cuMemcpyHtoD(landing, frame.data(), frame.size()) == CUDA_SUCCESS &&
         SwapToArgb(channels, width, height, landing, argb, npp_context);
}
}  // namespace

FrameEncoder::~FrameEncoder() {
  if (argb_buffer_ == 0 && landing_buffer_ == 0) {
    return;
  }
  // Freed under the encoder's context, which the encoder session still holds: this body runs
  // before the members are destroyed. Pushed here because the destructor can run on any thread.
  if (const auto scope = PushContext(encoder_->get_cuda_context())) {
    if (landing_buffer_ != 0) {
      cuMemFree(landing_buffer_);
    }
    if (argb_buffer_ != 0) {
      cuMemFree(argb_buffer_);
    }
  }
}

std::uint64_t FrameEncoder::staging_allocations() { return StagingAllocationCount().load(); }

bool FrameEncoder::hardware_available(std::uint32_t cuda_device, VideoCodec codec) {
  return nv_video_encoder::NVVideoEncoder::is_available(cuda_device, NvencCodecName(codec));
}

std::nullopt_t FrameEncoder::note_unusable_encoder(std::uint32_t width, std::uint32_t height,
                                                   std::string_view reason) {
  encoder_unusable_ = true;
  HOLOSCAN_LOG_ERROR(
      "Unable to start recording video for {} at {}x{} with codec {} on GPU {}: {}. It will "
      "record nothing for the rest of the run; rerun with video encoding turned off to record "
      "its raw camera frames instead.",
      topic_label_, width, height, foxglove_codec_format(), options_.cuda_device, reason);
  return std::nullopt;
}

bool FrameEncoder::check_encoder(std::uint32_t width, std::uint32_t height) {
  if (encoder_ != nullptr) {
    if (width == width_ && height == height_) {
      return true;
    }
    // A recorded video stream carries one resolution for its whole life, so a resized frame
    // has to be dropped rather than encoded against the session's original dimensions.
    log_failure(fmt::format(
        "Unable to record a camera frame as video: this video was started at {}x{} and the "
        "frame is {}x{}. Dropping the frame.",
        width_, height_, width, height));
    return false;
  }
  try {
    auto candidate = std::make_unique<nv_video_encoder::NVVideoEncoder>(
        width, height, NvencCodecName(options_.codec), options_.bitrate, options_.frame_rate,
        options_.cuda_device);
    candidate->initialize();
    encoder_ = std::move(candidate);
  } catch (const std::runtime_error& error) {
    // initialize()'s documented failure: the SDK's own exceptions are absorbed inside it and come
    // back as this, so it is the one type that means "this device will not open a session".
    note_unusable_encoder(width, height, error.what());
    return false;
  }
  width_ = width;
  height_ = height;
  return true;
}

std::optional<std::vector<std::byte>> FrameEncoder::encode(
    const holoscan::Tensor& tensor, std::uint32_t width, std::uint32_t height,
    holoscan::schema::ImageEncoding encoding) {
  // Held across the whole call rather than just the session setup below. The encoder's
  // input-buffer ring, the width/height this session was lazily sized to, encoder_unusable_ and
  // the failure-log counter are all read or mutated past this point, so a second caller
  // carrying another frame waits here instead of overlapping. In the logger a topic has one poll
  // thread, so this is never contended; it is what keeps the class safe to use from anywhere else.
  const std::scoped_lock lock(mutex_);
  if (encoder_unusable_) {
    // Already reported, once, when the first frame failed to open an encoding session.
    return std::nullopt;
  }
  try {
    const std::optional<std::uint32_t> encodable_channels = EncodableChannelCount(encoding);
    if (!encodable_channels) {
      log_failure(fmt::format(
          "Unable to record a camera frame as video: its image encoding ({}) is not one the "
          "video encoder accepts; only RGB8 and RGBA8 are. Dropping the frame.",
          static_cast<int>(encoding)));
      return std::nullopt;
    }
    const std::uint32_t channels = *encodable_channels;

    if (width == 0 || height == 0) {
      log_failure(fmt::format(
          "Unable to record a camera frame as video: the frame reports {}x{} pixels. Dropping "
          "the frame.",
          width, height));
      return std::nullopt;
    }

    const DLDataType dtype = tensor.dtype();
    if (tensor.ndim() != 1 || dtype.code != kDLUInt || dtype.bits != 8 || dtype.lanes != 1 ||
        tensor.data() == nullptr) {
      log_failure(
          "Unable to record a camera frame as video: the frame is not a flat buffer of 8-bit "
          "pixel bytes. Dropping the frame.");
      return std::nullopt;
    }

    const std::size_t source_bytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * channels;
    const auto available_bytes = static_cast<std::uint64_t>(tensor.nbytes());
    if (available_bytes < source_bytes) {
      log_failure(fmt::format(
          "Unable to record a camera frame as video: {}x{} pixels need {} bytes but the frame "
          "holds {}. Dropping the frame.",
          width, height, source_bytes, available_bytes));
      return std::nullopt;
    }

    const auto device_type = tensor.device().device_type;
    const bool on_host = device_type == kDLCPU || device_type == kDLCUDAHost;
    const bool on_device = device_type == kDLCUDA || device_type == kDLCUDAManaged;
    if (!on_host && !on_device) {
      log_failure(fmt::format(
          "Unable to record a camera frame as video: the frame is in memory the GPU cannot read "
          "(memory type {}). Dropping the frame.",
          static_cast<int>(device_type)));
      return std::nullopt;
    }

    if (!check_encoder(width, height)) {
      return std::nullopt;
    }

    // Makes the encoder's context current for the rest of this call, popped however this scope is
    // left. The two staging buffers outlive the call: they are made for the first frame and reused
    // by every later one, since the session's frame size is fixed.
    const auto context = PushContext(encoder_->get_cuda_context());
    if (!context ||
        !EnsureDeviceBuffer(argb_buffer_, std::size_t{width} * height * kArgbBytesPerPixel)) {
      log_failure("Unable to prepare the GPU for video encoding. Dropping the frame.");
      return std::nullopt;
    }

    // Built from the encoder's device, not zero-initialized: a zeroed context names device 0,
    // which is the wrong GPU whenever options_.cuda_device is not 0. See MakeNppStreamContext().
    const std::optional<NppStreamContext> npp_context = MakeNppStreamContext(options_.cuda_device);
    if (!npp_context) {
      log_failure(
          fmt::format("Unable to describe GPU {} to NPP for video encoding. Dropping the frame.",
                      options_.cuda_device));
      return std::nullopt;
    }

    // A frame arrives either in host memory (kDLCPU, kDLCUDAHost) or already on the GPU
    // (kDLCUDA, kDLCUDAManaged). One the encoder's context can already address is converted
    // where it lies, which spares a real camera pipeline a full-frame device-to-device copy on
    // every frame; anything else is staged through a copy first, because NPP cannot read host
    // memory and must not be pointed at an address this context cannot resolve.
    bool converted = false;
    if (on_device) {
      // tensor.data() is a void* whether the frame is in host or device memory; the driver API
      // wants a CUdeviceptr for the second, so the address is converted here, once.
      const auto device_address = std::bit_cast<CUdeviceptr>(tensor.data());
      if (IsAddressableByEncoder(device_address)) {
        frames_read_in_place_.fetch_add(1);
        converted = SwapToArgb(channels, width, height, device_address, argb_buffer_, *npp_context);
      } else {
        converted = ConvertDeviceFrame(device_address, channels, width, height, argb_buffer_,
                                       landing_buffer_, *npp_context);
      }
    } else if (on_host) {
      // source_bytes was checked against tensor.nbytes() above, so this span covers bytes the
      // tensor really holds.
      const std::span<const std::byte> host_frame(static_cast<const std::byte*>(tensor.data()),
                                                  source_bytes);
      converted = ConvertHostFrame(host_frame, channels, width, height, argb_buffer_,
                                   landing_buffer_, *npp_context);
    }
    if (!converted) {
      log_failure(
          "Unable to convert a camera frame to the format the video encoder requires. Dropping "
          "the frame.");
      return std::nullopt;
    }

    std::vector<std::byte> bitstream;
    if (encoder_->encode_frame(argb_buffer_, bitstream) == 0) {
      log_failure("Unable to encode a camera frame as video. Dropping the frame.");
      return std::nullopt;
    }
    // std::byte, so the sink can move this straight into the Foxglove message rather than copy
    // it. The encoder's own copy out of the SDK's packet is unavoidable -- that buffer dies with
    // encode_frame()'s scope -- but it is now the only copy this frame's bitstream takes.
    return bitstream;
  } catch (const std::bad_alloc&) {
    // The one failure here that is not a decision this function made: a frame's worth of host
    // memory for the bitstream. Anything else thrown is a bug, not a dropped frame, and is left to
    // reach the poller's own handler, which reports it and keeps the topic recording.
    log_failure("Unable to encode a camera frame as video: out of memory. Dropping the frame.");
    return std::nullopt;
  }
}

#else

FrameEncoder::~FrameEncoder() = default;

std::uint64_t FrameEncoder::staging_allocations() { return 0; }

bool FrameEncoder::hardware_available(std::uint32_t, VideoCodec) { return false; }

std::optional<std::vector<std::byte>> FrameEncoder::encode(const holoscan::Tensor&, std::uint32_t,
                                                           std::uint32_t,
                                                           holoscan::schema::ImageEncoding) {
  return std::nullopt;
}

#endif

}  // namespace isaac::os::mcap_logger
