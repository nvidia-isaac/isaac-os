/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/mcap_logger/mcap_sink.h"

#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include "holoscan/logger/logger.hpp"
#include <fmt/format.h>

#include "common/mcap_logger/failure_log_schedule.h"

namespace isaac::os::mcap_logger {

McapSink::McapSink(McapSinkOptions options)
    : output_path_(options.output_path
                       ? std::make_optional(
                             std::filesystem::absolute(*options.output_path).lexically_normal())
                       : std::nullopt) {
  if (!output_path_ && !options.enable_stream) {
    throw std::runtime_error(
        "The MCAP logger's sink was constructed with neither file writing nor streaming "
        "enabled -- at least one output must be usable.");
  }

  if (output_path_) {
    std::error_code ec;
    std::filesystem::create_directories(output_path_->parent_path(), ec);
    if (ec) {
      throw std::runtime_error(
          fmt::format("Unable to create the MCAP logger's output directory '{}': {}",
                      output_path_->parent_path().string(), ec.message()));
    }

    // McapWriterOptions::path is a non-owning std::string_view-like member -- it must be backed
    // by a variable that outlives the McapWriter::create() call below, not a temporary.
    const std::string path_str = output_path_->string();
    foxglove::McapWriterOptions writer_options;
    writer_options.context = context_;
    writer_options.path = path_str;
    writer_options.profile = "isaac_mcap_logger";
    writer_options.compression = options.compression;

    auto result = foxglove::McapWriter::create(writer_options);
    if (!result.has_value()) {
      throw std::runtime_error(
          fmt::format("Unable to create the MCAP logger's output file '{}': {}",
                      output_path_->string(), foxglove::strerror(result.error())));
    }
    writer_ = std::make_unique<foxglove::McapWriter>(std::move(result.value()));
    HOLOSCAN_LOG_INFO("The MCAP logger's output file is ready: {}", output_path_->string());
  }

  if (options.enable_stream) {
    foxglove::WebSocketServerOptions server_options;
    server_options.context = context_;
    server_options.name = "isaac_mcap_logger";
    server_options.host = options.websocket_host;
    server_options.port = options.websocket_port;

    auto result = foxglove::WebSocketServer::create(std::move(server_options));
    if (!result.has_value()) {
      throw std::runtime_error(fmt::format(
          "Unable to start the MCAP logger's websocket server on {}:{}: {}", options.websocket_host,
          options.websocket_port, foxglove::strerror(result.error())));
    }
    server_ = std::make_unique<foxglove::WebSocketServer>(std::move(result.value()));
    HOLOSCAN_LOG_INFO("The MCAP logger's websocket server is listening on {}:{}",
                      options.websocket_host, server_->port());
  }
}

McapSink::~McapSink() {
  // Every Channel in channels_ holds RawChannels that reference this sink's context_/writer_/
  // server_; drop them before either sink is torn down below, rather than relying on channels_'
  // declaration order relative to writer_/server_ to destroy them first.
  channels_.clear();

  if (writer_) {
    auto error = writer_->close();
    if (error != foxglove::FoxgloveError::Ok) {
      HOLOSCAN_LOG_ERROR("Unable to close the MCAP logger's output file '{}': {}",
                         output_path_->string(), foxglove::strerror(error));
    }
    writer_.reset();
  }

  if (server_) {
    auto error = server_->stop();
    if (error != foxglove::FoxgloveError::Ok) {
      HOLOSCAN_LOG_ERROR("Unable to stop the MCAP logger's websocket server: {}",
                         foxglove::strerror(error));
    }
    server_.reset();
  }
}

McapSink::Channel* McapSink::get_or_create_channel(std::string_view unique_id,
                                                   std::string_view schema_name,
                                                   std::span<const std::byte> binary_schema,
                                                   std::string_view message_encoding) {
  auto it = channels_.find(unique_id);
  if (it != channels_.end()) {
    return &it->second;
  }
  if (failed_channels_.contains(unique_id)) {
    // Already logged once, when the failure was first recorded below -- fail fast without
    // re-registering a channel with the SDK.
    return nullptr;
  }

  foxglove::Schema schema;
  schema.name = schema_name;
  // The schema blob itself is a FlatBuffers binary schema on both paths; only the message bytes
  // differ, which is what message_encoding records below.
  schema.encoding = kFlatBufferMessageEncoding;
  schema.data = binary_schema.data();
  schema.data_len = binary_schema.size();

  auto data_result = foxglove::RawChannel::create(unique_id, message_encoding, schema, context_);
  if (!data_result.has_value()) {
    log_channel_creation_failure(unique_id, foxglove::strerror(data_result.error()));
    failed_channels_.emplace(unique_id);
    return nullptr;
  }

  Channel channel{
      .data_channel = std::make_unique<foxglove::RawChannel>(std::move(data_result.value())),
  };
  auto [inserted_it, inserted] =
      channels_.insert_or_assign(std::string(unique_id), std::move(channel));
  return &inserted_it->second;
}

void McapSink::log_channel_creation_failure(std::string_view unique_id, std::string_view reason) {
  const std::uint64_t count = channel_creation_failure_log_count_.fetch_add(1) + 1;
  if (IsFailureReportDue(count)) {
    HOLOSCAN_LOG_ERROR(
        "Unable to create the MCAP logger's channel for topic '{}': {}. No further samples "
        "for this topic will be recorded. ({} such failures so far.)",
        unique_id, reason, count);
  }
}

void McapSink::log_schema_mismatch(std::string_view unique_id) {
  const std::uint64_t count = schema_mismatch_log_count_.fetch_add(1) + 1;
  if (IsFailureReportDue(count)) {
    HOLOSCAN_LOG_ERROR(
        "Unable to record a sample on the MCAP logger's topic '{}': it is already writing a "
        "different schema on this sink, and an MCAP channel carries one schema for its whole "
        "life. This sample is dropped; the topic's existing channel is unaffected. ({} such "
        "failures so far.)",
        unique_id, count);
  }
}

void McapSink::log_write_failure(std::string_view unique_id, foxglove::FoxgloveError error) {
  const std::uint64_t count = log_call_failure_log_count_.fetch_add(1) + 1;
  if (IsFailureReportDue(count)) {
    HOLOSCAN_LOG_ERROR("Unable to record a sample for topic '{}': {}. ({} such failures so far.)",
                       unique_id, foxglove::strerror(error), count);
  }
}

bool McapSink::write_flatbuffer_schema(std::string_view unique_id, std::string_view schema_name,
                                       std::span<const std::byte> binary_schema,
                                       std::uint64_t log_time_ns,
                                       std::span<const std::byte> payload) {
  return write_encoded(unique_id, schema_name, binary_schema, log_time_ns, payload);
}

bool McapSink::write_encoded(std::string_view unique_id, std::string_view schema_name,
                             std::span<const std::byte> binary_schema, std::uint64_t log_time_ns,
                             std::span<const std::byte> payload) {
  const std::scoped_lock lock(channels_mutex_);
  Channel* channel =
      get_or_create_channel(unique_id, schema_name, binary_schema, kFlatBufferMessageEncoding);
  if (channel == nullptr) {
    return false;
  }
  if (channel->data_channel == nullptr) {
    // unique_id already has a write_compressed_video() channel -- an MCAP channel carries one
    // schema for its whole life, so this topic can't also become a flatbuffer channel. Not
    // memoized in failed_channels_: that would also poison the topic's working video channel.
    log_schema_mismatch(unique_id);
    return false;
  }

  auto error = channel->data_channel->log(payload.data(), payload.size(), log_time_ns);
  if (error != foxglove::FoxgloveError::Ok) {
    log_write_failure(unique_id, error);
    return false;
  }
  return true;
}

bool McapSink::write_compressed_video(std::string_view unique_id, std::string_view frame_id,
                                      std::uint64_t log_time_ns, std::vector<std::byte> encoded,
                                      std::string_view codec) {
  // Assembled before the lock, not inside it. Copying the compressed frame into the message is
  // the most expensive thing this function does, and channels_mutex_ is the single lock every
  // topic's write contends on -- doing the copy under it would stall an unrelated joint-state or
  // IMU write for the length of a video frame's memcpy. Held only for the channel lookup and the
  // log() call below.
  foxglove::schemas::CompressedVideo message;
  message.timestamp =
      foxglove::schemas::Timestamp{static_cast<std::uint32_t>(log_time_ns / 1'000'000'000),
                                   static_cast<std::uint32_t>(log_time_ns % 1'000'000'000)};
  message.frame_id = std::string(frame_id);
  message.format = std::string(codec);
  // Moved, not copied. The encoder already produced std::byte for exactly this reason, so the
  // frame's bitstream is copied once -- out of the SDK's packet, which does not outlive the call
  // that produced it -- and never again on its way to the wire.
  message.data = std::move(encoded);

  const std::scoped_lock lock(channels_mutex_);
  Channel* channel = nullptr;
  auto it = channels_.find(unique_id);
  if (it != channels_.end()) {
    channel = &it->second;
  } else if (!failed_channels_.contains(unique_id)) {
    auto result = foxglove::schemas::CompressedVideoChannel::create(unique_id, context_);
    if (!result.has_value()) {
      log_channel_creation_failure(unique_id, foxglove::strerror(result.error()));
      failed_channels_.emplace(unique_id);
    } else {
      Channel new_channel{
          .video_channel = std::make_unique<foxglove::schemas::CompressedVideoChannel>(
              std::move(result.value())),
      };
      auto [inserted_it, inserted] =
          channels_.insert_or_assign(std::string(unique_id), std::move(new_channel));
      channel = &inserted_it->second;
    }
  }
  if (channel == nullptr) {
    return false;
  }
  if (channel->video_channel == nullptr) {
    // unique_id already has a write_flatbuffer_schema() channel -- an MCAP channel carries one
    // schema for its whole life, so this topic can't also become a CompressedVideo channel. Not
    // memoized in failed_channels_: that would also poison the topic's working flatbuffer channel.
    log_schema_mismatch(unique_id);
    return false;
  }

  const auto error = channel->video_channel->log(message, log_time_ns);
  if (error != foxglove::FoxgloveError::Ok) {
    log_write_failure(unique_id, error);
    return false;
  }
  return true;
}

std::optional<std::uint16_t> McapSink::websocket_port() const noexcept {
  return server_ ? std::make_optional(server_->port()) : std::nullopt;
}

}  // namespace isaac::os::mcap_logger
