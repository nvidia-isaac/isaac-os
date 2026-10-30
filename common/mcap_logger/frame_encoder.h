/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/sensor_io/image_metadata.hpp"

#include "common/mcap_logger/video_encode_options.h"

#ifdef ISAAC_ENABLE_NV_VIDEO_SDK
// Forward declared rather than included: the NVIDIA Video Codec SDK headers this type is built
// from would otherwise become a compile-time dependency of everything that records video.
namespace nv_video_encoder {
class NVVideoEncoder;
}  // namespace nv_video_encoder
#endif

namespace isaac::os::mcap_logger {

// Owns one topic's hardware video-encoding session. The underlying session keeps a ring of input
// buffers that every encode call mutates, so it is not safe to use from two threads at once:
// every call here is serialized on mutex_. The session is sized lazily from the first frame, so
// callers never restate width/height alongside the tensor contract.
class FrameEncoder {
 public:
  // Builds the encoder for one topic, or throws when this host cannot encode on the device the
  // options name. "Cannot encode" means the GPU is missing or has no usable hardware video
  // encoder -- an A100 or H100 ships a decoder but no encoder, for instance -- and it throws
  // std::runtime_error naming the device and what to change. A topic that asked for video must not
  // quietly record something else: a recording of raw tensors where video was requested looks like
  // a working run until someone opens the file, and a single warning line is easy to miss. A
  // caller that wants a fallback asks hardware_available() first and decides, visibly, what to
  // record instead.
  //
  // Also throws std::invalid_argument for a zero bitrate or frame rate, checked before the probe:
  // that is wrong on every machine, and must stay distinguishable from a host that cannot encode.
  //
  // Every encoder that exists has therefore passed its probe. That is not a promise that encoding
  // will succeed: the probe cannot foresee a frame size the encoder rejects, a device whose
  // session limit is exhausted after the probe passed, or one lost to a driver reset, so encode()
  // keeps its own guard for those. Nothing here opens the session -- that happens lazily on the
  // first frame.
  //
  // `topic` names the topic this encoder records for, so that a failure message can say which of
  // several encoded topics it is about. Optional: an unnamed encoder reports itself as such.
  explicit FrameEncoder(VideoEncodeOptions options, std::string_view topic = {});

  // True when `cuda_device` is a GPU this host has, and that GPU has a usable hardware video
  // encoder that offers `codec`. A device ordinal this host does not have answers false rather than
  // throwing. The codec matters because encoders differ in what they offer: an H.264-only part
  // opens a session and then refuses an H.265 configuration on the first frame, which a probe that
  // only opened a session would call available. The default, H.264, is what every NVENC part
  // offers, so it asks whether there is an encoder at all.
  // The constructor asks this for the options it was handed. Call it directly to decide, before
  // constructing, whether to ask for video at all, and always name the device.
  [[nodiscard]] static bool hardware_available(std::uint32_t cuda_device,
                                               VideoCodec codec = VideoCodec::kH264);

  ~FrameEncoder();
  FrameEncoder(const FrameEncoder&) = delete;
  FrameEncoder& operator=(const FrameEncoder&) = delete;

  // `tensor` is rank-1 contiguous uint8 pixel bytes; width/height/encoding come from the
  // sample's own holoscan.schema.Image metadata, since the tensor carries no shape of its own.
  //
  // nullopt on any failure (dimensions changed, unsupported encoding, short buffer, running out of
  // memory), and it reports those rather than throwing them. An exception it did not anticipate is
  // a bug, not a dropped frame, and propagates.
  //
  // Callers must drop the frame rather than fall back to a raw write: by the time this runs the
  // topic's MCAP channel already exists as foxglove.CompressedVideo, and one channel carries one
  // schema for its whole life.
  //
  // Failing to start the encoding session on the first frame is permanent rather than per-frame:
  // it is reported once, at error level, and every later frame then returns nullopt without
  // touching the GPU again.
  //
  // Safe to call from several threads: the whole call is serialized on mutex_, which it has to
  // be, since the underlying session's input-buffer ring is mutated on every encode. A topic's
  // frames are submitted by its one poll thread, so in the logger this lock is never contended;
  // it is what makes the class safe to use from anywhere else. Exclusion is not ordering: callers
  // that need frames encoded in a given order have to submit them in that order.
  std::optional<std::vector<std::byte>> encode(const holoscan::Tensor& tensor, std::uint32_t width,
                                               std::uint32_t height,
                                               holoscan::schema::ImageEncoding encoding);

  // How many frames this process has converted straight out of the caller's tensor, rather than
  // copying into a staging buffer first. Exists so that tests can observe which of the two
  // staging paths a frame actually took, and nothing else: no committed assertion can otherwise
  // tell a live zero-copy path from one that has quietly begun staging every frame, because
  // both produce identical recordings. That failure has happened once already, and was caught
  // only by deliberately breaking the conversion and noticing the tests still passed.
  //
  // Process-wide rather than per-encoder because the encoder a recording graph uses is owned
  // deep inside the logger and never handed out; a test reads this before and after a run and
  // compares. Monotonic and never reset, so read it as a difference.
  [[nodiscard]] static std::uint64_t frames_read_in_place();

  // How many device buffers encoders have allocated for staging a frame, process-wide. Exists so a
  // test can tell a session that reuses its buffers from one that allocates them afresh for every
  // frame, which no recording can show. Monotonic and never reset, so read it as a difference.
  [[nodiscard]] static std::uint64_t staging_allocations();

  // The lowercase codec spelling foxglove.CompressedVideo's `format` field expects.
  [[nodiscard]] std::string_view foxglove_codec_format() const {
    return FoxgloveCodecFormat(options_.codec);
  }

 private:
  // Counts for frames_read_in_place() above. Atomic because several topics encode concurrently
  // on the logger's worker threads, each through its own FrameEncoder.
  static std::atomic<std::uint64_t> frames_read_in_place_;

  // Reports a dropped frame at error level, on the schedule in failure_log_schedule.h: the first
  // drop and then every tenth, each with the running total. A stream that fails on every frame
  // therefore says so at once and keeps saying so, and is never silent. Silent was the failure
  // this replaces: a topic recording nothing for the rest of a run while the log promised to stay
  // quiet. Called with mutex_ held.
  void log_failure(std::string_view message);

#ifdef ISAAC_ENABLE_NV_VIDEO_SDK
  // Records that this topic can never encode anything, and reports it once. Deliberately not
  // routed through log_failure(): this is the single message that explains a topic which records
  // nothing at all for the whole run, so it must not wait its turn on log_failure()'s
  // schedule. Called with mutex_ held; returns nullopt so encode() can return it directly.
  std::nullopt_t note_unusable_encoder(std::uint32_t width, std::uint32_t height,
                                       std::string_view reason);

  // Checks that the encoding session is ready for a width x height frame, opening it first if this
  // is the first frame, so encode() has one question to ask about the session rather than three.
  // True when encoder_ is ready; false when this frame cannot be encoded, after saying why.
  //
  // Sized from the first frame because a frame's own metadata is the only place its resolution is
  // stated. Opening is attempted exactly once: it reports every failure by throwing, and every
  // such failure -- a GPU with no video encoder, a device index this machine does not have -- is
  // permanent for the run, so a retry would only reopen a session the driver has already refused.
  // That failure latches encoder_unusable_ through note_unusable_encoder(). A frame whose size
  // differs from the session's is dropped, not encoded against the original dimensions.
  // Called with mutex_ held.
  [[nodiscard]] bool check_encoder(std::uint32_t width, std::uint32_t height);
#endif

  VideoEncodeOptions options_;
  std::mutex mutex_;
#ifdef ISAAC_ENABLE_NV_VIDEO_SDK
  std::unique_ptr<nv_video_encoder::NVVideoEncoder> encoder_;  // guarded by mutex_
  // Device buffers for staging a frame, allocated for the first frame and reused by every later one
  // (see encode()) and freed in the destructor under the encoder's context. Held as unsigned long
  // long, which is what a CUdeviceptr is, so this header need not include the CUDA headers.
  unsigned long long argb_buffer_ = 0;     // guarded by mutex_
  unsigned long long landing_buffer_ = 0;  // guarded by mutex_
#endif
  std::uint32_t width_ = 0;   // guarded by mutex_
  std::uint32_t height_ = 0;  // guarded by mutex_
  // Set when the very first frame fails to open a hardware encoding session. That failure is a
  // property of the machine and its configuration -- a GPU that has no video encoder, for
  // instance -- so no later frame can succeed where the first one failed. Latched so the attempt
  // happens exactly once: without it, every frame for the rest of the run would open a brand-new
  // encoding session on a device that has already refused one.
  bool encoder_unusable_ = false;  // guarded by mutex_
  // How this encoder names itself in messages: "topic 'camera'", or "an unnamed topic".
  std::string topic_label_;
  std::uint64_t failure_count_ = 0;  // guarded by mutex_
};

}  // namespace isaac::os::mcap_logger
