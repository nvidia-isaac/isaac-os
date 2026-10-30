/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include "holoscan/channel/runtime.hpp"
#include "holoscan/channel/tensor_schema_endpoints.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/logger/logger.hpp"
#include "holoscan/schema/binary_schema.hpp"
#include "holoscan/schema/flatbuffer_schema_package.hpp"
#include <cuda_runtime.h>
#include <flatbuffers/flatbuffers.h>
#include <fmt/format.h>
// Required: DescriptorFlatBufferTensorViewTraits<holoscan::schema::Image> is specialized only
// here. Without it, the primary template resolves every tensor view to kNotImplemented at run
// time with no compile error.
#include "holoscan/sensor_io/image_metadata.hpp"
#include "holoscan/sensor_io/sensor_schema_package.hpp"

#include "common/mcap_logger/failure_log_schedule.h"
#include "common/mcap_logger/frame_encoder.h"
#include "common/mcap_logger/mcap_sink.h"
#include "common/mcap_logger/tensor_frame_builder.h"
#include "common/mcap_logger/topic_poller.h"
#include "common/mcap_logger/video_encode_options.h"
#include "tensor_frame_bfbs_generated.h"
#include "tensor_frame_generated.h"

namespace isaac::os::mcap_logger {

// Every tensor-schema topic this component writes shares this one root schema, regardless of the
// attached type T -- see tensor_frame.fbs's file comment for why.
inline constexpr std::string_view kTensorFrameCanonicalName = "isaac.mcap_logger.TensorFrame";

// Tensor-schema counterpart to TopicPoller<T> (topic_poller.h) -- owns one topic's
// ChannelTensorSchemaSubscriber<Payload> and blocks on it on a dedicated thread, same as
// TopicPoller<T>. The same thread materializes every tensor of each received bundle to bytes and
// writes it as a single TensorFrame FlatBuffers message via McapSink::write_flatbuffer_schema
// before receiving the next, so a topic's samples are written in the order they arrived.
//
// Given a FrameEncoder, an image topic takes an alternate path instead: the frame is
// hardware-encoded and written as a foxglove.CompressedVideo message on `<topic>`, alongside its
// holoscan.schema.Image metadata on `<topic>/metadata` at the same log time, since
// foxglove.CompressedVideo has no room for that metadata.
//
// ChannelResolvedTensorBundle<Payload> holds read claims on the tensors' backing allocations. The
// bundle, and so its claims, lives only for the write that handles it. stop() joins the poll
// thread, which first writes what the queue still holds, so every claim is released before this
// object's subscriber is destroyed. Nothing here relies on a claim outliving its subscriber or the
// ChannelRuntime: HsdkMcapLogger::shutdown() destroys every poller before it resets the runtime.
template <typename Payload>
class TensorTopicPoller : public PollerBase {
 public:
  TensorTopicPoller(std::string unique_id,
                    holoscan::ChannelTensorSchemaSubscriber<Payload> subscriber,
                    std::shared_ptr<McapSink> sink, std::uint32_t queue_depth,
                    std::shared_ptr<FrameEncoder> encoder = nullptr)
      : PollerBase(std::move(unique_id), queue_depth),
        subscriber_(std::move(subscriber)),
        sink_(std::move(sink)),
        encoder_(std::move(encoder)),
        thread_(&TensorTopicPoller::poll_loop, this) {}

  ~TensorTopicPoller() { stop(); }

  TensorTopicPoller(const TensorTopicPoller&) = delete;
  TensorTopicPoller& operator=(const TensorTopicPoller&) = delete;

  void stop() {
    stop_source_.request_stop();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  using RootType = typename holoscan::ChannelResolvedTensorBundle<Payload>::root_type;

  // Blocks until a bundle is received, a stop is requested, or the subscriber fails for good. The
  // SDK has no wait-and-receive for a multi-tensor bundle, so this waits for readiness and then
  // receives, going back to the wait if the receive loses its race (kNotReady).
  [[nodiscard]] holoscan::expected<holoscan::ChannelResolvedTensorBundle<Payload>, holoscan::Error>
  wait_and_receive_tensors(const std::stop_token& token) {
    while (true) {
      auto ready = subscriber_.wait_for_sample(NextWaitDeadline(subscriber_), token);
      if (!ready && ready.error().code == holoscan::ErrorCode::kTimeout) {
        continue;  // No publisher connected yet.
      }
      if (!ready) {
        return holoscan::make_unexpected(ready.error());
      }
      auto bundle = subscriber_.receive_tensors();
      if (bundle || bundle.error().code != holoscan::ErrorCode::kNotReady) {
        return bundle;
      }
    }
  }

  void poll_loop() {
    // RootType is the metadata table (e.g. holoscan::schema::Image), not Payload itself --
    // metadata_bytes() is already a serialized RootType flatbuffer, so only identity()/
    // binary_schema() are needed here, never encode().
    const auto& descriptor = holoscan::schema_package<RootType>();
    const std::stop_token token = stop_source_.get_token();

    while (true) {
      auto bundle = wait_and_receive_tensors(token);
      if (!bundle) {
        report_receive_ended(bundle.error());
        break;  // Stop requested, or the subscriber is permanently unusable.
      }
      write(descriptor, *bundle);
    }

    // Nothing more will be waited for: write what the queue still holds, without blocking. The
    // queue holds at most queue_depth samples, so that many receives drain it, and a producer
    // still publishing cannot keep this thread from finishing.
    for (std::uint32_t drained = 0; drained < queue_depth(); ++drained) {
      auto bundle = subscriber_.receive_tensors();
      if (!bundle) {
        break;
      }
      write(descriptor, *bundle);
    }
  }

  // Records one received bundle: as video when this topic has an encoder, otherwise as a
  // TensorFrame. log_time is the runtime's graph-emit time, read here before the write so the
  // recorded time is the capture-side one however long the write takes.
  template <typename Bundle>
  void write(const holoscan::SchemaPackageDescriptor<RootType>& descriptor, Bundle& bundle) {
    note_sequence(bundle.sample_metadata.sequence);
    const std::uint64_t log_time_ns =
        static_cast<std::uint64_t>(bundle.sample_metadata.publish_timestamp_ns);
    run_guarded([&] {
      const std::string_view schema_name = descriptor.identity().canonical_name;
      // Encoding needs width/height/encoding, which only holoscan.schema.Image
      // carries, so the encoded path exists only for that root type.
      if constexpr (std::is_same_v<RootType, holoscan::schema::Image>) {
        if (encoder_ != nullptr) {
          write_encoded_image(*encoder_, *sink_, unique_id(), schema_name, log_time_ns, bundle,
                              materialize_failure_log_count_);
          return;
        }
      }
      write_tensor_frame(*sink_, unique_id(), schema_name, log_time_ns, bundle,
                         materialize_failure_log_count_);
    });
  }

  // Records one sample as a TensorFrame: every tensor it carries, materialized to host bytes,
  // alongside its metadata. Static, with everything it touches passed in, so it can be exercised
  // without a poller. A tensor that cannot be read drops the whole sample, reported on the shared
  // failure schedule.
  template <typename Bundle>
  static void write_tensor_frame(McapSink& sink, std::string_view unique_id,
                                 std::string_view schema_name, std::uint64_t log_time_ns,
                                 Bundle& bundle, std::uint64_t& failure_count) {
    flatbuffers::FlatBufferBuilder builder;
    const auto built =
        BuildTensorFrame(builder, bundle, schema_name, holoscan::binary_schema<RootType>());
    if (!built) {
      const std::uint64_t count = ++failure_count;
      if (IsFailureReportDue(count)) {
        HOLOSCAN_LOG_WARN(
            "Unable to read a tensor from the MCAP logger's topic '{}': {} ({} such failures so "
            "far.)",
            unique_id, holoscan::runtime_error_message(built.error()), count);
      }
      return;
    }

    const auto binary_schema = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(::isaac::mcap_logger::TensorFrameBinarySchema::data()),
        ::isaac::mcap_logger::TensorFrameBinarySchema::size());
    const auto payload = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), builder.GetSize());
    sink.write_flatbuffer_schema(unique_id, kTensorFrameCanonicalName, binary_schema, log_time_ns,
                                 payload);
  }

  // Records one frame of an image topic as video, plus a sidecar channel carrying the metadata
  // the video cannot hold. Static, with everything it touches passed in, so it can be exercised
  // without a poller. Every failure either logs itself in the callee or is a dropped frame; nothing
  // is written raw, because this topic's channel already exists as foxglove.CompressedVideo.
  //
  // A topic's frames are encoded one at a time, in the order they were received, by the topic's
  // own poll thread, so they reach the encoder in capture order. That ordering is what a stream
  // with inter-frame dependencies would need, and today nothing depends on it: NVVideoEncoder
  // forces an IDR carrying its own SPS/PPS on every frame, so each frame is independently
  // decodable anyway. log_time_ns is read before the write, so the recorded times are the
  // capture-side times; frames of different topics still reach the file in whatever order their
  // threads finish, so a reader restores cross-topic order by reading in log-time order.
  template <typename Bundle>
  static void write_encoded_image(FrameEncoder& encoder, McapSink& sink, std::string_view unique_id,
                                  std::string_view schema_name, std::uint64_t log_time_ns,
                                  Bundle& bundle, std::uint64_t& size_failure_count) {
    // root() is nullptr when metadata is empty; GetRoot() on those bytes would read past
    // the buffer. bundle.tensors is a fixed std::array<Tensor, Payload::max_tensors>
    // (1 for ImageT), so there is no runtime count to check here.
    const holoscan::schema::Image* image = bundle.root();
    if (image == nullptr || image->width() <= 0 || image->height() <= 0) {
      // Without a size there is nothing to hand the encoder. Say so, on the shared failure
      // schedule: a topic that always publishes empty metadata would otherwise record nothing all
      // run with no explanation.
      const std::uint64_t count = ++size_failure_count;
      if (IsFailureReportDue(count)) {
        HOLOSCAN_LOG_WARN(
            "Unable to encode a frame from the MCAP logger's topic '{}': {} ({} such failures so "
            "far.)",
            unique_id,
            image == nullptr ? "it carries no image metadata"
                             : "its image metadata reports a non-positive size",
            count);
      }
      return;
    }
    // Not const: the bitstream is moved into the Foxglove message below rather than
    // copied into it.
    auto encoded = encoder.encode(bundle.tensors[0], static_cast<std::uint32_t>(image->width()),
                                  static_cast<std::uint32_t>(image->height()), image->encoding());
    if (!encoded) {
      // Dropped rather than written raw: this topic's channel already exists as
      // foxglove.CompressedVideo, and an MCAP channel carries one schema for its life.
      // FrameEncoder::encode() has already logged why, rate-limited.
      return;
    }
    // The frame this camera declared, not the topic it is recorded on. The sidecar
    // below republishes the same value, so taking it from anywhere else would leave the
    // two channels describing different coordinate frames for one picture. Absent
    // metadata leaves it empty, which is how Foxglove reads "no frame stated".
    const auto* header = image->header();
    const auto* frame_id = header != nullptr ? header->frame_id() : nullptr;
    if (!sink.write_compressed_video(
            unique_id, frame_id != nullptr ? frame_id->string_view() : std::string_view{},
            log_time_ns, std::move(*encoded), encoder.foxglove_codec_format())) {
      return;
    }

    // Written only if the video write succeeded, so the two channels stay frame-for-frame
    // joinable. A permanently unusable video channel would otherwise leave a growing trail of
    // metadata entries no video frame ever matches.
    //
    // Sidecar channel: foxglove.CompressedVideo has no room for capture_timestamp_ns,
    // exposure_time_ns, gain, color_space, encoding or the ROI offsets, so the metadata
    // this frame already carries would otherwise be discarded. Written with the same
    // log_time_ns as the video above, so a reader joins the two channels on log_time.
    // The schema recorded on the sidecar channel is the attached type's own name --
    // "holoscan.schema.Image" -- not the tensor wrapper the raw-tensor path writes.
    //
    // That name is load-bearing, and deliberately not one any image converter claims.
    // The Foxglove extension under tools/foxglove/isaac-to-foxglove-converter registers
    // its image converter on "isaac.mcap_logger.TensorFrame" and registers nothing
    // for "holoscan.schema.Image", so a sidecar channel is never offered to an Image
    // panel. Registering "holoscan.schema.Image" as a source of foxglove.RawImage would
    // break that: every sidecar channel would start being offered as a renderable image
    // with no pixels behind it, because the frame's pixels travel out of band on the
    // video channel and these bytes are metadata only. Route the metadata somewhere
    // other than an image panel instead.
    sink.write_flatbuffer_schema(VideoMetadataChannelName(unique_id), schema_name,
                                 holoscan::binary_schema<holoscan::schema::Image>(), log_time_ns,
                                 bundle.metadata_bytes());
  }

  holoscan::ChannelTensorSchemaSubscriber<Payload> subscriber_;
  std::shared_ptr<McapSink> sink_;
  // Null on a topic recorded as raw tensors; non-null routes this topic through NVENC instead.
  // Const: this topic's encoder is made once, for this topic, and never replaced. Only this topic's
  // poll thread calls it, and no other topic is handed it.
  const std::shared_ptr<FrameEncoder> encoder_;
  std::stop_source stop_source_;
  std::uint64_t materialize_failure_log_count_ = 0;  // Only this topic's poll thread touches it.
  std::thread thread_;
};

}  // namespace isaac::os::mcap_logger
