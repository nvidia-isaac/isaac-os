/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "foxglove/channel.hpp"
#include "foxglove/mcap.hpp"
#include "foxglove/schemas.hpp"
#include "foxglove/server.hpp"

#include "common/mcap_logger/transparent_string_hash.h"

namespace isaac::os::mcap_logger {

// The chunk compression a recording uses when none is asked for. The values are the Foxglove SDK's
// own foxglove::McapCompression, so there is no second enum to keep in step.
inline constexpr foxglove::McapCompression kDefaultMcapCompression = foxglove::McapCompression::Lz4;

// Configures McapSink's two independent sinks. At least one of output_path/enable_stream must
// select a usable sink -- McapSink's constructor throws otherwise. Both can be set together (the
// common case: record to file and watch live), or either alone (file-only, the default; or
// stream-only, e.g. a debug session with nothing persisted to disk).
struct McapSinkOptions {
  // nullopt disables file writing entirely. When set, McapSink opens (and later finalizes) an
  // MCAP file at this exact path -- McapSink does not resolve directories or timestamps itself;
  // that's HsdkMcapLogger's job.
  std::optional<std::filesystem::path> output_path;

  // Enables a foxglove::WebSocketServer bound to this sink's own Context, so every channel this
  // sink creates streams live to any connected Foxglove client in addition to (or instead of)
  // being written to file.
  bool enable_stream = false;

  // Matches the Foxglove SDK's own default. A loopback bind is unreachable from anywhere but the
  // application's own network namespace -- forward the port (e.g. over SSH) or override this to
  // "0.0.0.0" or a specific interface to reach it from elsewhere.
  std::string websocket_host = "127.0.0.1";

  // Matches Foxglove's own client default (8765). Pass 0 to let the OS assign a free ephemeral
  // port instead -- read the actual bound port back via McapSink::websocket_port() afterward.
  std::uint16_t websocket_port = 8765;

  // How each chunk of the MCAP file is compressed. Applies to the whole file, so every topic shares
  // it. Ignored when output_path is unset: a stream-only sink writes no file.
  foxglove::McapCompression compression = kDefaultMcapCompression;
};

// Wraps the MCAP file writer and/or a live Foxglove WebSocket server. Each write call takes a
// mutex only around channel lookup/create and the append -- callers are expected to have already
// encoded the sample to bytes.
//
// Only writes MCAP's log_time; a message's own capture/acquisition time, if it has one, belongs
// in its payload, not here.
class McapSink {
 public:
  // Throws std::runtime_error if:
  //  - neither options.output_path nor options.enable_stream selects a usable sink, or
  //  - options.output_path is set but its parent directory can't be created or the underlying
  //    MCAP writer fails to open it, or
  //  - options.enable_stream is set but the websocket server fails to bind (e.g. the port is
  //    already in use).
  // A sink with no usable output is not a degraded-but-functional object (unlike, say, one
  // broken topic's channel, which only drops that topic's own samples) -- letting construction
  // succeed anyway would mean a whole graph run could complete with every single sample silently
  // dropped and no hard failure anywhere.
  explicit McapSink(McapSinkOptions options);
  ~McapSink();

  McapSink(const McapSink&) = delete;
  McapSink& operator=(const McapSink&) = delete;

  // Message bytes are a single FlatBuffers message matching the channel's schema; decodable by
  // generic MCAP/Foxglove tooling directly. Every write*() method below produces this.
  static constexpr std::string_view kFlatBufferMessageEncoding = "flatbuffer";

  // Writes one flatbuffer-encoded sample. schema_name/binary_schema describe unique_id's schema,
  // used on first write and ignored thereafter. log_time_ns drives MCAP's log_time (replay
  // pacing). Returns false (and logs, rate-limited) on any Foxglove SDK error. When streaming is
  // enabled, this also fans the same sample out live -- no separate call is needed.
  bool write_flatbuffer_schema(std::string_view unique_id, std::string_view schema_name,
                               std::span<const std::byte> binary_schema, std::uint64_t log_time_ns,
                               std::span<const std::byte> payload);

  // Writes one NVENC-encoded frame as a foxglove.CompressedVideo message. Unlike
  // write_flatbuffer_schema(), the schema and message encoding are protobuf -- the Foxglove SDK
  // owns this schema, so the typed channel registers it. A topic uses this or
  // write_flatbuffer_schema() for its whole life, never both: an MCAP channel carries exactly one
  // schema, so mixing the two on one unique_id is not representable.
  //
  // frame_id is the coordinate frame the source image declared, not unique_id: a reader
  // resolving this video against a transform needs the sensor's frame, and the sidecar channel
  // publishes that same value, so deriving it from the topic name here would make the two
  // channels disagree about the frame they describe.
  bool write_compressed_video(std::string_view unique_id, std::string_view frame_id,
                              std::uint64_t log_time_ns, std::vector<std::byte> encoded,
                              std::string_view codec);

  // nullopt if streaming isn't enabled. Otherwise the port actually bound -- notably different
  // from McapSinkOptions::websocket_port when that was 0 ("let the OS pick").
  [[nodiscard]] std::optional<std::uint16_t> websocket_port() const noexcept;

 private:
  struct Channel {
    std::unique_ptr<foxglove::RawChannel> data_channel;
    std::unique_ptr<foxglove::schemas::CompressedVideoChannel> video_channel;
  };

  // Shared implementation of write_flatbuffer_schema() above.
  bool write_encoded(std::string_view unique_id, std::string_view schema_name,
                     std::span<const std::byte> binary_schema, std::uint64_t log_time_ns,
                     std::span<const std::byte> payload);

  // Returns the cached channel for unique_id, creating it on first call. A creation failure is
  // remembered in failed_channels_ so later calls fail fast instead of re-registering a fresh,
  // duplicate channel on every subsequent sample.
  Channel* get_or_create_channel(std::string_view unique_id, std::string_view schema_name,
                                 std::span<const std::byte> binary_schema,
                                 std::string_view message_encoding);

  // Logs (rate-limited via channel_creation_failure_log_count_) that unique_id's channel isn't
  // usable -- either the underlying Foxglove SDK create() call failed, or (from the mismatch
  // guards in write_encoded()/write_compressed_video()) unique_id already has a channel of the
  // other kind. Shared so the message wording and suppression rule live in one place.
  void log_channel_creation_failure(std::string_view unique_id, std::string_view reason);

  // Logs (rate-limited via schema_mismatch_log_count_) that unique_id was asked to write under
  // one schema while already writing another. Deliberately not log_channel_creation_failure():
  // that message ends "No further samples for this topic will be recorded", which is false here
  // -- the mismatch is not memoized in failed_channels_, precisely so the topic's existing
  // channel keeps working. Its own counter too, so a stream of mismatches cannot exhaust the
  // budget that genuine RawChannel::create() failures on other topics report through.
  void log_schema_mismatch(std::string_view unique_id);

  // Logs (rate-limited via log_call_failure_log_count_) that a log() call for unique_id's channel
  // failed. Shared by write_encoded() and write_compressed_video().
  void log_write_failure(std::string_view unique_id, foxglove::FoxgloveError error);

  std::optional<std::filesystem::path> output_path_;
  // Explicitly created, not default-constructed: the default constructor returns the SDK's shared
  // global context, and every sink on a context receives every channel logged on it. Two McapSink
  // instances sharing one context would therefore write each other's samples into their own
  // outputs -- the SDK's own guidance is a separate Context per group of sinks that should see the
  // same data. test_mcap_sink.cc covers both halves of this: the shared-context fan-out and this
  // sink's isolation from another live instance.
  foxglove::Context context_ = foxglove::Context::create();
  // Either or both of these may be null, but never both at once -- the constructor throws rather
  // than leaving a McapSink with no usable sink at all.
  std::unique_ptr<foxglove::McapWriter> writer_;
  std::unique_ptr<foxglove::WebSocketServer> server_;
  std::mutex channels_mutex_;
  // Searched on every sample written, by the topic name as a string_view, so both are transparent:
  // a lookup builds no temporary string. See transparent_string_hash.h.
  std::unordered_map<std::string, Channel, TransparentStringHash, std::equal_to<>> channels_;
  std::unordered_set<std::string, TransparentStringHash, std::equal_to<>> failed_channels_;

  // Rate-limits the failure logs below so a persistently broken channel can't flood the log at
  // the poller's poll rate for the life of the process.
  std::atomic<std::uint64_t> channel_creation_failure_log_count_{0};
  std::atomic<std::uint64_t> schema_mismatch_log_count_{0};
  std::atomic<std::uint64_t> log_call_failure_log_count_{0};
};

}  // namespace isaac::os::mcap_logger
