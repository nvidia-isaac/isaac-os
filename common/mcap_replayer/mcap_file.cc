/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/mcap_replayer/mcap_file.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <fmt/format.h>

#include "common/mcap_logger/compressed_video_proto.h"
#include "common/mcap_logger/video_encode_options.h"
#include "tensor_frame_generated.h"

namespace isaac::os::mcap_replayer {

McapFile::McapFile(std::unique_ptr<mcap::McapReader> reader, bool should_loop)
    : reader_(std::move(reader)), should_loop_(should_loop) {
  if (const auto& statistics = reader_->statistics(); statistics.has_value()) {
    message_counts_.emplace(statistics->channelMessageCounts.begin(),
                            statistics->channelMessageCounts.end());
  }
  for (const auto& [channel_id, channel_ptr] : reader_->channels()) {
    // The reader lists channels in no particular order, so the lowest id is kept: the channel the
    // recorder created first for the topic, the same one whichever order they come in.
    const auto [entry, added] = topic_2_channels_.try_emplace(channel_ptr->topic, channel_id);
    if (!added) {
      entry->second = std::min(entry->second, channel_id);
    }
  }
  // Warned once the choice is final, so the message names the channel that is replayed.
  std::unordered_map<std::string, std::size_t> channels_per_topic;
  for (const auto& [channel_id, channel_ptr] : reader_->channels()) {
    ++channels_per_topic[channel_ptr->topic];
  }
  for (const auto& [topic, count] : channels_per_topic) {
    if (count > 1) {
      HOLOSCAN_LOG_WARN(
          "Topic '{}' is recorded on {} channels. Only channel {}, the lowest, is replayed; "
          "messages on the others are skipped.",
          topic, count, topic_2_channels_.at(topic));
    }
    // A topic the logger recorded as encoded video is a protobuf foxglove.CompressedVideo channel
    // plus a companion channel carrying its image metadata. Pair the two from here on, so next()
    // only ever returns them joined. A video channel without its companion stays unpaired, and
    // verify_topic() rejects it when someone asks for it as a type.
    const mcap::ChannelId video_id = topic_2_channels_.at(topic);
    const mcap::ChannelPtr channel = reader_->channel(video_id);
    const mcap::SchemaPtr schema = channel ? reader_->schema(channel->schemaId) : nullptr;
    if (schema && schema->name == kCompressedVideoSchemaName &&
        channel->messageEncoding == "protobuf") {
      const auto sidecar =
          topic_2_channels_.find(::isaac::os::mcap_logger::VideoMetadataChannelName(topic));
      if (sidecar != topic_2_channels_.end()) {
        pairs_.try_emplace(video_id, VideoPair{.sidecar_channel = sidecar->second, .topic = topic});
        sidecar_to_video_.emplace(sidecar->second, video_id);
      }
    }
  }
}

McapFile McapFile::open(const std::filesystem::path& path, bool should_loop) {
  auto reader = std::make_unique<mcap::McapReader>();

  const mcap::Status open_status = reader->open(path.string());
  if (!open_status.ok()) {
    throw std::runtime_error(fmt::format("Could not open the recording at '{}': {}.", path.string(),
                                         open_status.message));
  }

  const mcap::Status summary_status =
      reader->readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);
  if (!summary_status.ok()) {
    throw std::runtime_error(
        fmt::format("Could not read channel information from the recording at '{}': {}.",
                    path.string(), summary_status.message));
  }

  McapFile file(std::move(reader), should_loop);
  file.reset_view();
  if (!file.last_problem_->empty()) {
    throw std::runtime_error(
        fmt::format("Could not read the messages in the recording at '{}': "
                    "{}.",
                    path.string(), *file.last_problem_));
  }
  return file;
}

std::vector<mcap::ChannelId> McapFile::channel_ids() const {
  std::vector<mcap::ChannelId> ids;
  const auto channels = reader_->channels();
  ids.reserve(channels.size());
  for (const auto& [id, channel_ptr] : channels) {
    ids.push_back(id);
  }
  // mcap's channel map is unordered; sort for a deterministic listing.
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::optional<std::string> McapFile::topic_of(mcap::ChannelId channel_id) const {
  const mcap::ChannelPtr channel = reader_->channel(channel_id);
  if (!channel) {
    return std::nullopt;
  }
  return channel->topic;
}

std::optional<std::uint64_t> McapFile::message_count(mcap::ChannelId channel_id) const {
  if (!message_counts_.has_value()) {
    return std::nullopt;
  }
  const auto entry = message_counts_->find(channel_id);
  // A channel absent from the statistics holds no messages.
  return entry == message_counts_->end() ? 0U : entry->second;
}

std::optional<mcap::ChannelId> McapFile::channel_of(std::string_view topic) const {
  const auto entry = topic_2_channels_.find(std::string(topic));
  if (entry == topic_2_channels_.end()) {
    return std::nullopt;
  }
  return entry->second;
}

std::string McapFile::describe_recorded_topics() const {
  std::vector<std::string> topics;
  topics.reserve(topic_2_channels_.size());
  for (const auto& [topic, channel_id] : topic_2_channels_) {
    topics.push_back(topic);
  }
  std::sort(topics.begin(), topics.end());

  std::string joined;
  for (std::size_t i = 0; i < topics.size(); ++i) {
    if (i > 0) {
      joined += ", ";
    }
    joined += "'" + topics[i] + "'";
  }
  return joined;
}

namespace {

// Whether one recorded tensor-envelope message says it carries the type named type_name, with
// exactly binary_schema as its schema. Logs why not.
bool envelope_matches(const mcap::MessageView& message_view, std::string_view topic,
                      std::string_view type_name, std::span<const std::byte> binary_schema) {
  // Unchecked bytes off disk: verify before touching the root, or a truncated record would be
  // read past the end of the buffer.
  flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t*>(message_view.message.data),
                                 message_view.message.dataSize);
  if (!::isaac::mcap_logger::VerifyTensorFrameBuffer(verifier)) {
    HOLOSCAN_LOG_WARN("The first recorded message on topic '{}' is damaged.", topic);
    return false;
  }
  const auto* frame = ::isaac::mcap_logger::GetTensorFrame(message_view.message.data);

  // Both are optional in the format, so a well-formed frame can still lack them.
  const auto* recorded_binary_schema = frame->metadata_binary_schema();
  const auto* recorded_name = frame->metadata_schema_name();
  if (recorded_binary_schema == nullptr || recorded_name == nullptr) {
    HOLOSCAN_LOG_WARN("Topic '{}' does not say what type its messages carry.", topic);
    return false;
  }
  const std::span<const std::byte> recorded_binary_schema_bytes(
      reinterpret_cast<const std::byte*>(recorded_binary_schema->data()),
      recorded_binary_schema->size());
  const bool matches = recorded_name->string_view() == type_name &&
                       std::ranges::equal(recorded_binary_schema_bytes, binary_schema);
  if (!matches) {
    HOLOSCAN_LOG_WARN("Topic '{}' is recorded as '{}', not the type '{}' it was asked for.", topic,
                      recorded_name->str(), type_name);
  }
  return matches;
}

// The codec one recorded video message says it was encoded with, or nothing after logging why not.
std::optional<std::string> codec_of_video_message(const mcap::MessageView& message_view,
                                                  std::string_view topic) {
  const auto fields = ::isaac::os::mcap_logger::ReadCompressedVideo(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(message_view.message.data),
      message_view.message.dataSize));
  if (!fields.has_value()) {
    HOLOSCAN_LOG_WARN("The first recorded video message on topic '{}' is damaged.", topic);
    return std::nullopt;
  }
  if (fields->format.empty()) {
    HOLOSCAN_LOG_WARN(
        "The first recorded video message on topic '{}' does not say which codec it was encoded "
        "with.",
        topic);
    return std::nullopt;
  }
  return std::string(fields->format);
}

}  // namespace

bool McapFile::verify_topic_as(std::string_view topic, std::string_view type_name,
                               std::span<const std::byte> binary_schema) {
  const auto channel_entry = topic_2_channels_.find(std::string(topic));
  if (channel_entry == topic_2_channels_.end()) {
    HOLOSCAN_LOG_WARN("No topic named '{}' was found in the recording. Recorded topics: {}.", topic,
                      describe_recorded_topics());
    return false;
  }
  const mcap::ChannelPtr channel = reader_->channel(channel_entry->second);
  const mcap::SchemaPtr schema = channel ? reader_->schema(channel->schemaId) : nullptr;
  if (!schema) {
    HOLOSCAN_LOG_WARN("The topic '{}' was not recorded with a schema.", topic);
    return false;
  }

  if (const auto as_sidecar = sidecar_to_video_.find(channel_entry->second);
      as_sidecar != sidecar_to_video_.end()) {
    // Replay joins this channel to its video, so the video's topic is the one to name.
    const mcap::ChannelPtr video = reader_->channel(as_sidecar->second);
    const std::string video_topic = video ? video->topic : std::string("its video");
    HOLOSCAN_LOG_WARN(
        "Topic '{}' carries the metadata of encoded video topic '{}'; name '{}' instead.", topic,
        video_topic, video_topic);
    return false;
  }

  if (const auto pair = pairs_.find(channel_entry->second); pair != pairs_.end()) {
    return verify_paired_video(topic, channel_entry->second, pair->second, type_name,
                               binary_schema);
  }

  if (schema->name != kTensorFrameSchemaName) {
    const bool matches =
        schema->name == type_name && std::ranges::equal(schema->data, binary_schema);
    if (!matches) {
      HOLOSCAN_LOG_WARN("Topic '{}' is recorded as '{}', not the type '{}' it was asked for.",
                        topic, schema->name, type_name);
    }
    return matches;
  }

  return verify_tensor_envelope(topic, channel_entry->second, type_name, binary_schema);
}

bool McapFile::verify_paired_video(std::string_view topic, mcap::ChannelId video_channel,
                                   VideoPair& pair, std::string_view type_name,
                                   std::span<const std::byte> binary_schema) {
  // Recorded as encoded video, so the topic's type is the type of its metadata channel: the video
  // decodes to what that metadata describes.
  const mcap::ChannelPtr sidecar = reader_->channel(pair.sidecar_channel);
  const mcap::SchemaPtr sidecar_schema = sidecar ? reader_->schema(sidecar->schemaId) : nullptr;
  const bool matches = sidecar_schema && sidecar_schema->name == type_name &&
                       sidecar->messageEncoding == "flatbuffer" &&
                       std::ranges::equal(sidecar_schema->data, binary_schema);
  if (!matches) {
    HOLOSCAN_LOG_WARN(
        "Topic '{}' is recorded as encoded video of '{}', not the type '{}' it was asked for.",
        topic, sidecar_schema ? sidecar_schema->name : std::string("nothing readable"), type_name);
    return false;
  }
  if (!pair.codec.empty()) {
    return true;
  }

  // The codec decides whether the video can be decoded at all, so a topic whose codec cannot be
  // read is refused here rather than failing frame by frame during replay.
  std::optional<std::string> codec;
  std::string read_error;
  switch (visit_first_message(topic, video_channel, read_error,
                              [&](const mcap::MessageView& message_view) {
                                codec = codec_of_video_message(message_view, topic);
                              })) {
    case FirstMessage::kVisited:
      break;
    case FirstMessage::kUnreadable:
      HOLOSCAN_LOG_WARN(
          "Unable to read the first video message of topic '{}' to check its codec: {}.", topic,
          read_error);
      return false;
    case FirstMessage::kNone:
      HOLOSCAN_LOG_WARN(
          "The encoded video topic '{}' has no recorded video messages to read its codec from.",
          topic);
      return false;
  }
  if (!codec.has_value()) {
    return false;
  }
  pair.codec = std::move(*codec);
  return true;
}

bool McapFile::verify_tensor_envelope(std::string_view topic, mcap::ChannelId channel,
                                      std::string_view type_name,
                                      std::span<const std::byte> binary_schema) {
  // The channel's schema is the TensorFrame envelope, so the payload's real name and binary schema
  // come from the replayed channel's first message.
  bool matches = false;
  std::string read_error;
  switch (
      visit_first_message(topic, channel, read_error, [&](const mcap::MessageView& message_view) {
        matches = envelope_matches(message_view, topic, type_name, binary_schema);
      })) {
    case FirstMessage::kVisited:
      return matches;
    case FirstMessage::kUnreadable:
      // Validation fails as it would for any other recording whose type cannot be checked.
      HOLOSCAN_LOG_WARN("Unable to read the first message of topic '{}' to check its type: {}.",
                        topic, read_error);
      return false;
    case FirstMessage::kNone:
      break;
  }
  HOLOSCAN_LOG_WARN("The topic '{}' has no recorded messages to check its nested schema against.",
                    topic);
  return false;
}

McapFile::FirstMessage McapFile::visit_first_message(
    std::string_view topic, mcap::ChannelId channel, std::string& read_error,
    const std::function<void(const mcap::MessageView&)>& visit) {
  // A separate view on the same reader leaves next()'s position undisturbed, and reading in
  // log-time order matches what replay reads. The view is filtered by topic, so another channel
  // recorded under the same topic is stepped past: only the channel replay uses may vouch for it.
  mcap::ReadMessageOptions peek_options;
  peek_options.readOrder = mcap::ReadMessageOptions::ReadOrder::LogTimeOrder;
  peek_options.topicFilter = [topic](std::string_view candidate) { return candidate == topic; };
  try {
    mcap::LinearMessageView peek_view = reader_->readMessages(
        [](const mcap::Status&) {
          // Ignored here: replay reads these messages again and reports a problem then.
        },
        peek_options);
    for (auto message = peek_view.begin(); message != peek_view.end(); ++message) {
      if ((*message).channel->id == channel) {
        visit(*message);
        return FirstMessage::kVisited;
      }
    }
  } catch (const std::exception& error) {
    // The reader throws on some damaged data instead of reporting it.
    read_error = error.what();
    return FirstMessage::kUnreadable;
  }
  return FirstMessage::kNone;
}

std::optional<std::string> McapFile::video_codec_of(std::string_view topic) const {
  const auto channel_entry = topic_2_channels_.find(std::string(topic));
  if (channel_entry == topic_2_channels_.end()) {
    return std::nullopt;
  }
  const auto pair = pairs_.find(channel_entry->second);
  if (pair == pairs_.end() || pair->second.codec.empty()) {
    return std::nullopt;
  }
  return pair->second.codec;
}

bool McapFile::is_tensor_channel(std::string_view topic) const {
  const auto channel_entry = topic_2_channels_.find(std::string(topic));
  if (channel_entry == topic_2_channels_.end()) {
    return false;
  }
  if (pairs_.contains(channel_entry->second)) {
    return true;
  }
  const mcap::ChannelPtr channel = reader_->channel(channel_entry->second);
  if (!channel) {
    return false;
  }
  const mcap::SchemaPtr schema = reader_->schema(channel->schemaId);
  return schema && schema->name == kTensorFrameSchemaName;
}

void McapFile::reset_view() {
  // iterator_ refers into *view_, so drop it before view_ is replaced.
  iterator_.reset();
  advance_pending_ = false;
  pass_read_ = 0;
  pass_returned_ = 0;
  last_problem_->clear();
  mcap::ReadMessageOptions options;
  options.readOrder = mcap::ReadMessageOptions::ReadOrder::LogTimeOrder;
  try {
    // The reader reports a failed read by calling this and then ending iteration; capture the
    // reason so an unreadable recording is distinguishable from one that simply ended.
    view_ = std::make_unique<mcap::LinearMessageView>(reader_->readMessages(
        [problem = last_problem_.get()](const mcap::Status& status) { *problem = status.message; },
        options));
    iterator_.emplace(view_->begin());
  } catch (const std::exception& error) {
    fail(error.what());
  }
}

void McapFile::fail(std::string_view reason) {
  // The reader throws on some damaged data (a chunk claiming an impossible size, say) instead of
  // reporting it. That is the same failure, so it is kept the same way.
  *last_problem_ = reason;
  failed_ = true;
  iterator_.reset();
  view_.reset();
}

bool McapFile::is_paired_video(std::string_view topic) const {
  const std::optional<mcap::ChannelId> channel = channel_of(topic);
  return channel.has_value() && pairs_.contains(*channel);
}

void McapFile::drop_held_halves() {
  for (auto& [video_channel, pair] : pairs_) {
    if (pair.holds_video || pair.holds_sidecar) {
      ++pair.dropped;
    }
    pair.holds_video = false;
    pair.holds_sidecar = false;
    if (pair.dropped > 0 && !pair.warned) {
      pair.warned = true;
      HOLOSCAN_LOG_WARN(
          "{} messages of the encoded video topic '{}' (and its metadata) were left out of the "
          "replay because their partner never arrived at the same log time.",
          pair.dropped, pair.topic);
    }
  }
}

std::uint64_t McapFile::dropped_halves() const {
  std::uint64_t total = 0;
  for (const auto& [video_channel, pair] : pairs_) {
    total += pair.dropped;
  }
  return total;
}

ReadResult McapFile::next(RecordedMessage& out) {
  if (failed_) {
    return ReadResult::kEnd;
  }
  try {
    if (!iterator_) {
      reset_view();
    } else {
      if (advance_pending_) {
        ++(*iterator_);
        advance_pending_ = false;
      }
      if (*iterator_ == view_->end()) {
        // Anything still held will never be completed, whether the pass wraps or the read ends.
        drop_held_halves();
        if (!should_loop_ || !last_problem_->empty()) {
          return ReadResult::kEnd;
        }
        if (pass_read_ > 0 && pass_returned_ == 0) {
          // Wrapping would only read the same unusable messages again, forever.
          fail(
              "No video message in the recording has a matching metadata message, so nothing can "
              "be replayed");
          return ReadResult::kEnd;
        }
        reset_view();
      }
    }
    if (failed_ || *iterator_ == view_->end()) {
      // Failed, or no messages at all, or none readable: a freshly reset iterator is already at
      // end().
      return ReadResult::kEnd;
    }
    const ReadResult result = read_current(out);
    if (result == ReadResult::kMessage) {
      ++pass_returned_;
    }
    return result;
  } catch (const std::exception& error) {
    fail(error.what());
    return ReadResult::kEnd;
  }
}

bool McapFile::offer_half(mcap::ChannelId channel_id, std::uint64_t log_time_ns,
                          std::span<const std::byte> payload, RecordedMessage& out) {
  const auto sidecar_entry = sidecar_to_video_.find(channel_id);
  const bool is_sidecar = sidecar_entry != sidecar_to_video_.end();
  const mcap::ChannelId video_channel = is_sidecar ? sidecar_entry->second : channel_id;
  VideoPair& pair = pairs_.at(video_channel);

  // Whole. The held half's own bytes back its span and the reader's buffer backs the half just
  // read, so nothing is copied here; both stay valid until the next call, which is the earliest
  // either is touched again.
  const bool partner_held = is_sidecar ? pair.holds_video : pair.holds_sidecar;
  if (partner_held && pair.held_log_time_ns == log_time_ns) {
    pair.holds_video = false;
    pair.holds_sidecar = false;
    const std::span<const std::byte> held(pair.held_bytes);
    out.channel_id = video_channel;
    out.log_time_ns = log_time_ns;
    out.payload = is_sidecar ? held : payload;
    out.sidecar = is_sidecar ? payload : held;
    return true;
  }
  // No partner at this instant. A pair holds at most one half, so anything still held is either
  // the same kind as this message or its partner at another time: it can never be completed and
  // is overwritten, which drops it like any message that is not returned.
  if (pair.holds_video || pair.holds_sidecar) {
    ++pair.dropped;
  }
  pair.holds_video = !is_sidecar;
  pair.holds_sidecar = is_sidecar;
  pair.held_log_time_ns = log_time_ns;
  pair.held_bytes.assign(payload.begin(), payload.end());
  return false;
}

ReadResult McapFile::read_current(RecordedMessage& out) {
  while (true) {
    const mcap::MessageView& message_view = **iterator_;
    const mcap::ChannelId channel_id = message_view.channel->id;
    const std::uint64_t log_time_ns = message_view.message.logTime;
    const std::span<const std::byte> payload(
        reinterpret_cast<const std::byte*>(message_view.message.data),
        message_view.message.dataSize);
    // Set before anything below can return: the next call steps past this message, unless the
    // code at the bottom of this loop has already done so.
    advance_pending_ = true;
    ++pass_read_;

    if (!pairs_.contains(channel_id) && !sidecar_to_video_.contains(channel_id)) {
      // Not part of a pair: handed back as recorded.
      out.channel_id = channel_id;
      out.log_time_ns = log_time_ns;
      out.payload = payload;
      out.sidecar = {};
      return ReadResult::kMessage;
    }
    if (offer_half(channel_id, log_time_ns, payload, out)) {
      return ReadResult::kMessage;
    }

    // Nothing to return yet: step on to the next message of this pass. The end of the pass is left
    // for the next call, which takes the ordinary end-of-file path.
    ++(*iterator_);
    advance_pending_ = false;
    if (*iterator_ == view_->end()) {
      return ReadResult::kNothingReady;
    }
  }
}

}  // namespace isaac::os::mcap_replayer
