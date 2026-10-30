/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "holoscan/logger/logger.hpp"
#include "holoscan/schema/binary_schema.hpp"
#include "holoscan/schema/schema_package.hpp"

#include "mcap/reader.hpp"

namespace isaac::os::mcap_replayer {

// The channel schema every tensor-schema stream is recorded under (common/mcap_logger/
// tensor_frame.fbs). The channel's own schema is this fixed envelope; the payload type's real name
// and binary schema are nested inside each message.
inline constexpr std::string_view kTensorFrameSchemaName = "isaac.mcap_logger.TensorFrame";

// The schema an encoded topic's video channel is recorded under (see
// common/mcap_logger/video_encode_options.h). A companion channel at
// VideoMetadataChannelName(topic) carries the image metadata the video has no room for. The
// constructor pairs the two, so a recording's encoded topics come out of next() as one joined
// message each.
inline constexpr std::string_view kCompressedVideoSchemaName = "foxglove.CompressedVideo";

// One recorded message: its channel, recorded log time (nanoseconds) and raw payload bytes. The
// payload is a view of the reader's own buffer, valid until the next McapFile::next() call.
//
// A paired video topic comes out as one message made of both its recorded halves: channel_id is
// the video channel's, payload the video bytes, sidecar the metadata bytes recorded with them.
// sidecar is empty for every other message.
struct RecordedMessage {
  mcap::ChannelId channel_id = 0;
  std::uint64_t log_time_ns = 0;
  std::span<const std::byte> payload;
  std::span<const std::byte> sidecar;
};

// What one McapFile::next() call found.
enum class ReadResult {
  // out holds the next message.
  kMessage,
  // The call read messages but has nothing to return yet and is not finished: it stepped past the
  // held half of a pair and reached the end of the pass. Call next() again; that call takes the
  // ordinary end-of-file path.
  kNothingReady,
  // The recording is exhausted, unreadable, or (when not looping) finished for good.
  kEnd,
};

// A thin wrapper over mcap::McapReader: the rest of this library reads a recording only through
// it. Channel lookups go straight to the reader's own channel map, parsed by open() via
// readSummary(), rather than a duplicate table.
//
// reader_ and view_ are held through unique_ptr because mcap::LinearMessageView references the
// McapReader and its Iterator references the view; both need addresses that survive a McapFile
// move (e.g. out of open()'s return value).
class McapFile {
 public:
  McapFile(McapFile&&) noexcept = default;
  McapFile& operator=(McapFile&&) noexcept = default;
  McapFile(const McapFile&) = delete;
  McapFile& operator=(const McapFile&) = delete;
  ~McapFile() = default;

  // Opens path, parses its summary section and prepares to read its messages. should_loop selects
  // next()'s behavior at end of file: wrap to the start, or stop for good. Throws
  // std::runtime_error naming path and the reader's description of the problem on failure,
  // including a recording whose messages cannot be read in log-time order (distinct from one that
  // simply holds no messages).
  [[nodiscard]] static McapFile open(const std::filesystem::path& path, bool should_loop = false);

  // The recorded channel ids, sorted ascending (mcap's own map is unordered) so listings of what
  // a file contains are deterministic.
  [[nodiscard]] std::vector<mcap::ChannelId> channel_ids() const;

  [[nodiscard]] std::optional<std::string> topic_of(mcap::ChannelId channel_id) const;
  // How many messages the recording holds on a channel, from its statistics section; nullopt if
  // the recording has none. Read once when the file is opened.
  [[nodiscard]] std::optional<std::uint64_t> message_count(mcap::ChannelId channel_id) const;
  // The channel recorded under topic, if any. A topic recorded on several channels resolves to the
  // lowest channel id; the constructor warns about the others, whose messages are not replayed.
  [[nodiscard]] std::optional<mcap::ChannelId> channel_of(std::string_view topic) const;

  // Checks that topic is recorded as exactly the schema T expects. Returns false on any failure
  // (no such topic, no schema, no message to read a tensor envelope's nested schema from, a
  // damaged first message, or a mismatch with T), logging the specific reason as a warning.
  template <typename T>
  [[nodiscard]] bool verify_topic(std::string_view topic) {
    return verify_topic_as(topic, holoscan::schema_package<T>().identity().canonical_name,
                           std::span<const std::byte>(holoscan::binary_schema<T>()));
  }

  // True when topic's messages are wrapped in the tensor envelope rather than being the payload
  // type's own bytes. False for an unknown topic or one with no schema.
  [[nodiscard]] bool is_tensor_channel(std::string_view topic) const;

  // True when topic's replayed channel is a video channel paired with its metadata channel.
  [[nodiscard]] bool is_paired_video(std::string_view topic) const;

  // The codec a paired video topic was recorded in ("h264" or "h265"), so a caller can check that
  // it can be decoded before replay starts. Read from the topic's first video message by
  // verify_topic(), which fails for a video topic whose codec cannot be read; empty until that has
  // succeeded for the topic, and for a topic that is not paired video.
  [[nodiscard]] std::optional<std::string> video_codec_of(std::string_view topic) const;

  // Fills out and returns kMessage with the next message in log-time order. At end of file it
  // wraps to the start if opened with should_loop (the caller sees a log time that goes
  // backwards); otherwise it returns kEnd for good, as it does for a file with no readable
  // messages.
  //
  // out.payload and out.sidecar view memory that stays valid only until the next call (the
  // reader's buffer, or the held half of a pair): a caller that keeps the bytes longer copies
  // them. Messages of a paired topic are returned only once both halves have arrived; a half whose
  // partner never comes is dropped.
  //
  // A looping recording whose whole pass returns nothing, because every message is a half whose
  // partner never arrives, ends the read the same way, with the cause in read_problem(): replaying
  // it again would never produce a message.
  //
  // A read failing partway through a pass ends iteration the way end of file does, but leaves the
  // reader's reason in read_problem(), including where the reader throws on damaged data instead
  // of reporting it. Such an end is final even when looping, since wrapping would replay only the
  // readable start of a damaged recording again and again.
  [[nodiscard]] ReadResult next(RecordedMessage& out);

  // Why the last pass over the recording ended early, or empty if it did not.
  [[nodiscard]] const std::string& read_problem() const { return *last_problem_; }

  // How many halves of paired video topics were dropped so far because their partner never arrived
  // at the same log time, summed over all pairs and passes. For diagnostics and tests.
  [[nodiscard]] std::uint64_t dropped_halves() const;

 private:
  McapFile(std::unique_ptr<mcap::McapReader> reader, bool should_loop);

  // The body of verify_topic<T>(), out of line so the reader's API is not used from a public
  // header: type_name and binary_schema describe the type the topic is wanted as.
  [[nodiscard]] bool verify_topic_as(std::string_view topic, std::string_view type_name,
                                     std::span<const std::byte> binary_schema);

  // (Re)creates the message-iteration view from the beginning of the file, in log-time order.
  void reset_view();

  // Records that reading failed for reason, ending iteration for good.
  void fail(std::string_view reason);

  // The recorded topics, quoted and comma-separated in sorted order, for messages naming what the
  // recording actually holds.
  [[nodiscard]] std::string describe_recorded_topics() const;

  // A video channel and its metadata channel, with the one half waiting for the other, if any:
  // at most one of `holds_video` and `holds_sidecar` is true. The held half is copied because the
  // reader reuses its buffer on the next read, and `held_bytes` keeps its capacity when the slot
  // empties, so holding a half allocates nothing once it has grown to the largest message.
  struct VideoPair {
    mcap::ChannelId sidecar_channel = 0;
    // The video topic's name, for messages about the pair.
    std::string topic;
    bool holds_video = false;
    bool holds_sidecar = false;
    std::uint64_t held_log_time_ns = 0;
    std::vector<std::byte> held_bytes;
    // The codec the video was recorded in, read by verify_topic(); empty until then.
    std::string codec;
    // Halves discarded without a partner, and whether that has been reported yet.
    std::uint64_t dropped = 0;
    bool warned = false;
  };

  // The pieces of verify_topic_as(), one for each way a topic can be recorded: encoded video (its
  // metadata channel's type, and its codec read and kept in `pair`) and the tensor envelope (the
  // payload's own type, read from the first message). Both log why they return false.
  [[nodiscard]] bool verify_paired_video(std::string_view topic, mcap::ChannelId video_channel,
                                         VideoPair& pair, std::string_view type_name,
                                         std::span<const std::byte> binary_schema);
  [[nodiscard]] bool verify_tensor_envelope(std::string_view topic, mcap::ChannelId channel,
                                            std::string_view type_name,
                                            std::span<const std::byte> binary_schema);

  // What visit_first_message() found.
  enum class FirstMessage { kVisited, kNone, kUnreadable };

  // Calls visit with the first message recorded on channel, in log-time order as replay reads
  // them, without disturbing next()'s position. kNone if the channel has no message; kUnreadable,
  // with the reader's explanation in read_error, if the reader throws on damaged data.
  [[nodiscard]] FirstMessage visit_first_message(
      std::string_view topic, mcap::ChannelId channel, std::string& read_error,
      const std::function<void(const mcap::MessageView&)>& visit);

  // Discards every half still waiting and reports the pairs that have dropped halves (once per
  // pair); called when the pass ends.
  void drop_held_halves();

  // Handles the message the iterator is on: a message outside any pair is returned as recorded,
  // a half of a pair is completed with its held partner or held itself, and while halves are being
  // held it steps the iterator on by itself until something can be returned or the pass ends.
  ReadResult read_current(RecordedMessage& out);

  // Offers a message of a paired channel (the video channel or its metadata channel) to its pair.
  // Returns true, with `out` filled, if that completes the pair; otherwise holds this half
  // (overwriting anything stale) and returns false.
  bool offer_half(mcap::ChannelId channel_id, std::uint64_t log_time_ns,
                  std::span<const std::byte> payload, RecordedMessage& out);

  std::unique_ptr<mcap::McapReader> reader_;
  // Whether next() wraps at end of file or stops for good. Fixed at construction.
  bool should_loop_;
  // The reader's report from the last reset_view(), empty if none. open() consults it for a more
  // useful error than "no messages". Behind a pointer because the reader's callback, copied into
  // the view, writes here and needs an address that survives a McapFile move.
  std::unique_ptr<std::string> last_problem_ = std::make_unique<std::string>();
  std::unique_ptr<mcap::LinearMessageView> view_;
  // Optional because Iterator has no public default constructor.
  std::optional<mcap::LinearMessageView::Iterator> iterator_;
  // Set when the reader threw: the view is unusable, and next() reports the end from then on.
  bool failed_ = false;
  // Set once next() has handed out the message iterator_ is at. Stepping past it waits for the next
  // call, because moving the iterator may release the buffer that message's payload views.
  bool advance_pending_ = false;
  // Messages read from the file, and messages returned to the caller, during the current pass.
  std::uint64_t pass_read_ = 0;
  std::uint64_t pass_returned_ = 0;

  std::unordered_map<std::string, mcap::ChannelId> topic_2_channels_;
  // Registered pairs by video channel id, and the reverse lookup for metadata channels.
  std::unordered_map<mcap::ChannelId, VideoPair> pairs_;
  std::unordered_map<mcap::ChannelId, mcap::ChannelId> sidecar_to_video_;
  // Per-channel message counts, unset when the recording has no statistics section.
  std::optional<std::unordered_map<mcap::ChannelId, std::uint64_t>> message_counts_;
};

}  // namespace isaac::os::mcap_replayer
