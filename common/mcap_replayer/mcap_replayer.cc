/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/mcap_replayer/mcap_replayer.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "holoscan/channel/runtime.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/logger/logger.hpp"

#include "common/mcap_replayer/pacing.h"

namespace isaac::os::mcap_replayer {

namespace {

// How long to wait before looking again for a message the reader has not decoded yet.
constexpr auto kBufferPollInterval = std::chrono::milliseconds(1);

// How much to buffer before the timeline starts, so the first messages are not charged for the
// time the reader needs to get ahead of them. Small next to the buffer itself: filling the whole
// buffer first would read tens of megabytes of tensor data before the first message goes out.
constexpr std::size_t kPrimeMessages = 32;
constexpr auto kPrimePollInterval = std::chrono::milliseconds(2);

// How long every attach()ed topic's consumer has to subscribe before the run gives up and stops
// without replaying anything. Running out of it is fatal, so it is sized for the slowest
// plausible graph start rather than the typical one:
// run_async() is ordinarily tens of milliseconds, but a graph creating a CUDA context or
// allocating device memory can take about a second, more on a busy machine (a 100 ms budget was
// measured losing a dozen opening messages). The wait ends as soon as the last consumer
// subscribes, so a healthy run does not pay for the margin, and a failing one reports within
// seconds.
constexpr auto kSubscriberWaitDeadline = std::chrono::seconds(5);

// The names as one comma-separated list in alphabetical order, so a message listing topics reads
// the same from run to run; empty_text if there are none.
std::string joined_sorted(std::vector<std::string_view> names, const char* empty_text) {
  std::sort(names.begin(), names.end());
  std::string joined;
  for (const std::string_view name : names) {
    if (!joined.empty()) {
      joined += ", ";
    }
    joined += name;
  }
  return joined.empty() ? std::string(empty_text) : joined;
}

ReplayOptions validate(ReplayOptions options) {
  if (!(options.playback_speed > 0.0)) {
    throw std::invalid_argument("The replay speed must be greater than zero.");
  }
  if (options.buffer_messages == 0) {
    throw std::invalid_argument("Replay must be allowed to buffer at least one message.");
  }
  if (options.start_stall_timeout <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument("The start-up stall timeout must be greater than zero.");
  }
  return options;
}

}  // namespace

McapReplayer::McapReplayer(const std::filesystem::path& path, ReplayOptions options)
    : options_(validate(options)) {
  file_ = McapFile::open(path, options_.loop);
  context_ = std::make_shared<ReplayContext>(options_.buffer_messages, options_.playback_speed);
}

McapReplayer::~McapReplayer() { stop(); }

std::string McapReplayer::replayed_topics() const {
  std::vector<std::string_view> names;
  for (const auto& [channel_id, topic] : context_->topics) {
    names.push_back(topic.name);
  }
  return joined_sorted(std::move(names), "none");
}

void McapReplayer::add_replay_topic(std::string_view topic, std::unique_ptr<PortOwner> port,
                                    MessageBuilder build) {
  // verify_topic<T>() has already found the topic, so it has a channel.
  const mcap::ChannelId channel = file_->channel_of(topic).value();
  const bool added = context_->topics
                         .emplace(channel, ReplayTopic{.name = std::string(topic),
                                                       .port = std::move(port),
                                                       .build = std::move(build)})
                         .second;
  if (!added) {
    throw std::invalid_argument(
        "Topic '" + std::string(topic) +
        "' is named more than once; each topic is replayed on a single port.");
  }
}

std::string McapReplayer::endpoint_name(std::string_view topic) const {
  return options_.operator_name + "/" + std::string(topic);
}

void McapReplayer::start(const holoscan::ChannelEndpointCatalog& catalog) {
  if (started_.exchange(true)) {
    throw std::logic_error(
        "This replay has already been started; replaying the recording again means a new "
        "replay.");
  }

  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    throw std::runtime_error("Unable to create this replay's ChannelRuntime: " +
                             std::string(holoscan::runtime_error_message(runtime.error())) + ".");
  }
  runtime_ = std::move(*runtime);
  // Only attach()ed topics were imported, so only they have an endpoint to publish on.
  for (auto& [channel_id, topic] : context_->topics) {
    if (topic.attached) {
      topic.port->attach(*runtime_, catalog, endpoint_name(topic.name));
    }
  }
  prefetch_thread_ = std::thread([this] { prefetch_loop(); });
  pacing_thread_ = std::thread([this] {
    pacing_loop();
    finished_.store(true);
  });
}

void McapReplayer::stop() {
  if (!started_.load()) {
    return;
  }
  // Makes the work below run once even when the destructor follows an explicit call.
  if (stopped_.exchange(true)) {
    return;
  }

  request_stop();
  if (prefetch_thread_.joinable()) {
    prefetch_thread_.join();
  }
  if (pacing_thread_.joinable()) {
    pacing_thread_.join();
  }
  // Both threads have finished; release the decoded messages nobody will receive, since the
  // graph may keep the context alive long after the run.
  context_->buffer.clear();

  context_->stats.log_summary();
}

void McapReplayer::request_stop() {
  {
    const std::lock_guard<std::mutex> lock(stop_mutex_);
    stop_requested_.store(true);
  }
  stop_signal_.notify_all();
  // The prefetch thread may be blocked in a full buffer's push(), which the signal above cannot
  // reach, and nothing will drain the buffer once pacing has stopped.
  context_->buffer.close();
}

void McapReplayer::wait_for_stop_until(std::chrono::steady_clock::time_point until) {
  std::unique_lock<std::mutex> lock(stop_mutex_);
  stop_signal_.wait_until(lock, until, [this] { return stop_requested_.load(); });
}

void McapReplayer::prefetch_loop() {
  RecordedMessage message;
  while (!stopping()) {
    const ReadResult result = file_->next(message);
    if (result == ReadResult::kNothingReady) {
      continue;  // Read past the held half of a pair; the stop check above runs before the next
                 // call.
    }
    if (result == ReadResult::kEnd) {
      // Nothing left to read (finished its pass, empty, or unreadable). pacing_loop() ends the run
      // once what is already buffered has drained.
      if (!file_->read_problem().empty()) {
        context_->stats.record_read_failed();
        HOLOSCAN_LOG_ERROR(
            "Reading the recording failed partway through, so the replay ends early: {}.",
            file_->read_problem());
      }
      context_->buffer.mark_reader_finished();
      return;
    }
    context_->messages_read.fetch_add(1);

    const auto found = context_->topics.find(message.channel_id);
    // Not a topic this run was asked to replay, or one declared but never attach()ed: there is no
    // consumer or endpoint to send it to.
    if (found == context_->topics.end() || !found->second.attached) {
      continue;
    }
    auto built = found->second.build(message.payload, message.sidecar, message.log_time_ns);
    if (!built) {
      // Counted and logged here, not in the builder, because the topic is known here.
      context_->stats.record_decode_failure(found->second.stats_id, built.error());
      continue;
    }

    // Blocks while the buffer is full, which is the only backpressure this thread gets.
    if (!context_->buffer.push(
            BufferedEntry{.message = std::move(*built), .channel_id = found->first})) {
      return;
    }
  }
}

std::size_t McapReplayer::buffering_target() const {
  const std::size_t usual = std::min(kPrimeMessages, options_.buffer_messages);
  if (options_.loop) {
    return usual;
  }
  std::uint64_t attached_messages = 0;
  for (const auto& [channel_id, topic] : context_->topics) {
    if (!topic.attached) {
      continue;
    }
    const std::optional<std::uint64_t> count = file_->message_count(channel_id);
    if (!count.has_value()) {
      return usual;
    }
    attached_messages += *count;
  }
  return static_cast<std::size_t>(std::min<std::uint64_t>(usual, attached_messages));
}

bool McapReplayer::prime() {
  const auto began = std::chrono::steady_clock::now();
  const auto subscriber_deadline = began + kSubscriberWaitDeadline;
  const std::size_t target = buffering_target();
  std::uint64_t previous_read = 0;
  auto last_progress = began;
  bool buffered = false;
  // Each attach()ed topic's subscribed flag latches once its consumer is seen: it is not expected
  // to leave during start-up, and only the missing ones are named if the wait runs out. A topic
  // nobody attach()ed has nothing to wait for.

  // Polled together, so start-up costs the longer wait, not the sum.
  while (!stopping()) {
    const auto now = std::chrono::steady_clock::now();
    if (!buffered) {
      const std::uint64_t read = context_->messages_read.load();
      if (read > previous_read) {
        previous_read = read;
        last_progress = now;
      }
      if (context_->buffer.size() >= target || context_->buffer.reader_finished()) {
        // Buffered enough, or the whole recording is (a reader that finished has nothing more).
        buffered = true;
      } else if (now - last_progress >= options_.start_stall_timeout) {
        // Nothing read for the whole timeout: a reader that is stuck, not merely slow or skipping
        // past messages nobody asked for.
        report_start_stalled(read);
        return false;
      }
    }

    bool every_consumer_subscribed = true;
    for (auto& [channel_id, topic] : context_->topics) {
      if (!topic.attached || topic.subscribed) {
        continue;
      }
      // A failed refresh means not ready yet (the consuming graph may still be coming up); the
      // deadline bounds how long that lasts.
      topic.subscribed =
          topic.port->update_connections().has_value() && topic.port->has_subscribers();
      every_consumer_subscribed = every_consumer_subscribed && topic.subscribed;
    }

    if (buffered && every_consumer_subscribed) {
      return true;
    }
    // Rechecked here: a stop requested since the loop condition is the caller's choice, and must
    // not be reported as a missing consumer.
    if (!every_consumer_subscribed && now >= subscriber_deadline && !stopping()) {
      report_missing_consumers();
      return false;
    }
    wait_for_stop_until(now + kPrimePollInterval);
  }
  return false;
}

void McapReplayer::report_start_stalled(std::uint64_t messages_read) {
  if (stopping()) {
    return;
  }
  context_->stats.record_start_stalled();
  // An error, logged once: the run never began, and replay cannot keep the recording's timing
  // behind a reader that is not delivering.
  HOLOSCAN_LOG_ERROR(
      "Replay stopped without sending anything: reading the recording made no progress for {:.1f} "
      "seconds while the run was starting, after {} messages had been read. Check that the file is "
      "readable and the storage holding it is responding.",
      std::chrono::duration<double>(options_.start_stall_timeout).count(), messages_read);
}

void McapReplayer::report_missing_consumers() {
  std::vector<std::string_view> missing_names;
  for (const auto& [channel_id, topic] : context_->topics) {
    if (topic.attached && !topic.subscribed) {
      context_->stats.record_never_subscribed(topic.stats_id);
      missing_names.push_back(topic.name);
    }
  }
  std::string missing;
  std::sort(missing_names.begin(), missing_names.end());
  for (const std::string_view name : missing_names) {
    if (!missing.empty()) {
      missing += ", ";
    }
    missing += "'" + std::string(name) + "'";
  }
  // An error, logged once: the run never began, and without it a consumer that never connected
  // looks the same as a recording holding nothing on those topics.
  HOLOSCAN_LOG_ERROR(
      "Replay stopped without sending anything: no consumer subscribed to {} within {} seconds of "
      "the run starting. Every topic wired to a consumer needs that consumer's graph to be running "
      "before replay can begin; check that the graph was started and has not stopped.",
      missing, std::chrono::duration_cast<std::chrono::seconds>(kSubscriberWaitDeadline).count());
}

void McapReplayer::pacing_loop() {
  // Once per run, not again when a looping recording wraps.
  if (!prime()) {
    // Either a stop was already requested or a consumer never subscribed. In the latter case
    // nothing else would stop the prefetch thread from decoding into a buffer nobody drains. A
    // later stop() or the destructor still joins both threads and logs the summary once.
    request_stop();
    return;
  }

  // Constructed here, not when buffering ended, so its origin is taken when there is a recorded
  // time to map it to; an origin taken while the buffer was empty would be in the past and the
  // opening stretch would go out at full rate. Owned by this thread alone (no synchronization);
  // release_if_due() rebases it when a looping reader's wrap reaches the front of the buffer,
  // which can happen while messages from the previous pass are still waiting.
  PacingClock clock(options_.playback_speed);
  // Recorded time of the last released message, which release_if_due() compares against to detect
  // a wrap. Unset before the first release and right after a rebase.
  std::optional<std::uint64_t> previous_log_time;

  while (!stopping()) {
    ReplayBuffer::ReleaseOutcome outcome =
        context_->buffer.release_if_due(clock, previous_log_time);

    if (outcome.released.has_value()) {
      previous_log_time = outcome.released->message->log_time_ns;
      warn_if_recording_has_no_timeline(*previous_log_time);
      publish(*outcome.released);
      // More may already be due (many, at high playback speed), so ask again without sleeping.
      continue;
    }

    // Nothing is due yet: sleep until the next message is, or until a stop.
    if (outcome.sleep_until.has_value()) {
      wait_for_stop_until(*outcome.sleep_until);
      continue;
    }

    // The buffer is empty. reader_finished comes from the same locked look, so a last message
    // pushed just before the reader finished is not missed.
    if (outcome.reader_finished) {
      // Nothing more can become due: the recording finished its one pass (the ordinary case
      // without ReplayOptions::loop), held nothing replayable, or could not be read further.
      return;
    }

    // Nothing decoded yet. The prefetch thread cannot signal this one, so poll.
    wait_for_stop_until(std::chrono::steady_clock::now() + kBufferPollInterval);
  }
}

void McapReplayer::warn_if_recording_has_no_timeline(std::uint64_t log_time) {
  ++released_total_;
  if (!first_log_time_.has_value()) {
    first_log_time_ = log_time;
  } else if (log_time != *first_log_time_) {
    log_time_has_advanced_ = true;
  }
  // With one recorded time throughout, every message is always due, so they go out as fast as the
  // consumer accepts them (and repeat without pause when looping). Detected after more than a
  // buffer's worth of messages without the recorded time moving, which no recording of a running
  // system produces.
  if (warned_about_a_recording_without_a_timeline_ || log_time_has_advanced_ ||
      released_total_ <= options_.buffer_messages) {
    return;
  }
  warned_about_a_recording_without_a_timeline_ = true;
  HOLOSCAN_LOG_WARN(
      "Every message in this recording carries the same recorded time, so there is no recorded "
      "cadence to replay: the messages are being sent as fast as the consumer accepts them, and "
      "the recording repeats without pause.");
}

void McapReplayer::publish(BufferedEntry& released) {
  // Published on the pacing thread itself. With attach()'s kDropOldest policy the publish
  // overwrites an unread sample rather than waiting for the consumer, so a slow consumer cannot
  // drag the schedule.
  const auto found = context_->topics.find(released.channel_id);
  if (found == context_->topics.end()) {
    // Entries are only built for topics in this table, which does not change once the run starts,
    // so this means the replay's own bookkeeping is wrong. Warned every time: it should never
    // happen, so there is no flood to rate limit.
    HOLOSCAN_LOG_WARN(
        "A replayed message was dropped because its topic is no longer known to the replay "
        "(channel {} of the recording). This is a fault in the replay itself, not in the "
        "recording or the consumer.",
        released.channel_id);
    return;
  }
  const ReplayTopic& topic = found->second;
  if (topic.port->emit(*released.message)) {
    record_delivery(topic, released);
  } else {
    record_refusal(topic);
  }
}

void McapReplayer::record_delivery(const ReplayTopic& topic, const BufferedEntry& released) {
  // Lag is measured against this entry's own due time, clamped at zero (a message sent
  // fractionally early is on time, and a negative would drag the average below zero). One clock
  // reading serves both the lag and the stats' comparison against the recording.
  const auto emitted_at = std::chrono::steady_clock::now();
  const auto lag =
      std::max(emitted_at - released.message->target, std::chrono::steady_clock::duration::zero());
  context_->stats.record_emitted(topic.stats_id,
                                 std::chrono::duration_cast<std::chrono::nanoseconds>(lag),
                                 released.message->log_time_ns, emitted_at);
}

void McapReplayer::record_refusal(const ReplayTopic& topic) {
  // Counted and, at most once per interval per topic, warned about by the stats: a topic that
  // cannot publish once usually cannot publish at message rate.
  context_->stats.record_refused_emit(topic.stats_id);
}

}  // namespace isaac::os::mcap_replayer
