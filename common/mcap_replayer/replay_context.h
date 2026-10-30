/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "holoscan/channel/runtime.hpp"

#include "common/mcap_replayer/decoder.h"
#include "common/mcap_replayer/mcap_file.h"
#include "common/mcap_replayer/message.h"
#include "common/mcap_replayer/pacing.h"
#include "common/mcap_replayer/replay_stats.h"
#include "common/mcap_replayer/typed_port.h"

namespace isaac::os::mcap_replayer {

// One decoded message waiting its turn. It names its channel rather than pointing at its topic, so
// the pacing thread finds the port and everything else through ReplayContext::topics.
struct BufferedEntry {
  std::unique_ptr<Message> message;
  mcap::ChannelId channel_id = 0;
};

// One replayed topic, keyed by its channel id in ReplayContext::topics.
struct ReplayTopic {
  // What it is reported and wired under.
  std::string name;
  // Holds the ImportedTopic and the publisher this topic emits through.
  std::unique_ptr<PortOwner> port;
  // Turns a recorded payload into a Message for port. Runs on the prefetch thread and never
  // touches port.
  MessageBuilder build;
  // This topic's index in the run's stats, set when it is attach()ed.
  ReplayStats::TopicId stats_id = 0;
  // Set by McapReplayer::attach<T>() during compose(). A topic nobody attach()ed is legitimate --
  // nothing will ever subscribe to it -- so it is neither read nor published, and the start-up
  // gate (McapReplayer::prime()) waits on, and fails the run over, only the topics this flags.
  bool attached = false;
  // Latched by prime() once a consumer has been seen subscribed. Pacing thread only.
  bool subscribed = false;
};

// The queue between the prefetch thread (push()) and the pacing thread (release_if_due()).
// Inspecting the front entry, deciding whether it is due and popping it happen under one lock, so
// the pacing thread never acts on a front it saw under an earlier acquisition. A released entry
// is handed straight back to the pacing thread to publish.
class ReplayBuffer {
 public:
  explicit ReplayBuffer(std::size_t max_messages) : max_messages_(max_messages) {}

  // Appends one decoded message, blocking while the buffer is full (the only backpressure the
  // prefetch thread gets; nothing is discarded). Returns false once close() has been called, so a
  // producer that nobody will drain again stops instead of hanging.
  bool push(BufferedEntry entry) {
    std::unique_lock<std::mutex> lock(mutex_);
    drained_.wait(lock, [this] { return closed_ || size_ < max_messages_; });
    if (closed_) {
      return false;
    }
    entries_.push_back(std::move(entry));
    ++size_;
    return true;
  }

  // What one call to release_if_due() found. released is engaged when the front entry was due and
  // has been popped. sleep_until is engaged when the buffer holds something that is not due yet:
  // the wall-clock instant to sleep until before asking again. Both empty means the buffer is
  // empty right now.
  struct ReleaseOutcome {
    std::optional<BufferedEntry> released;
    std::optional<std::chrono::steady_clock::time_point> sleep_until;
    // True only when released and sleep_until are both unset and the reader will never push again.
    // Read in the same locked step as the emptiness check: checking the two separately would let
    // the reader push its last message and finish in between, and that message would be lost.
    bool reader_finished = false;
  };

  // Releases the front entry if it is due now according to clock, stamping its due time. Only the
  // pacing thread calls this; clock and previous_log_time belong to it and are touched under the
  // buffer lock only because that is where the front entry is.
  //
  // Log-times are non-decreasing within a pass, so a front entry earlier than previous_log_time
  // means the reader wrapped back to the start of a looping recording. The clock is rebased here
  // against that entry, and previous_log_time reset so the check does not fire again for it.
  [[nodiscard]] ReleaseOutcome release_if_due(PacingClock& clock,
                                              std::optional<std::uint64_t>& previous_log_time) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (entries_.empty()) {
      return {.reader_finished = reader_finished_};
    }
    const std::uint64_t log_time = entries_.front().message->log_time_ns;
    if (previous_log_time.has_value() && log_time < *previous_log_time) {
      clock.rebase(log_time, std::chrono::steady_clock::now());
      previous_log_time.reset();
    }
    const auto target = clock.target_for(log_time);
    if (PacingClock::should_sleep(target - std::chrono::steady_clock::now())) {
      return {.sleep_until = target};
    }
    entries_.front().message->target = target;
    BufferedEntry released = std::move(entries_.front());
    entries_.pop_front();
    --size_;
    lock.unlock();
    drained_.notify_all();
    return {.released = std::move(released)};
  }

  // Marks that the reader will never push again (finished its pass, or could not read further).
  // Safe to call more than once. Only an empty buffer reports it (see release_if_due()), so
  // everything already pushed is still released first.
  void mark_reader_finished() {
    const std::lock_guard<std::mutex> lock(mutex_);
    reader_finished_ = true;
  }

  // Whether the reader has marked itself finished (see mark_reader_finished()). Says nothing about
  // what is still buffered.
  [[nodiscard]] bool reader_finished() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return reader_finished_;
  }

  // Drops everything still buffered. Called once a run has ended, so decoded messages that will
  // never be sent do not stay resident for as long as the graph holds the context.
  void clear() {
    const std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    size_ = 0;
  }

  // Refuses every further push and wakes anyone already waiting in one.
  void close() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    drained_.notify_all();
  }

  [[nodiscard]] std::size_t size() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return size_;
  }

 private:
  const std::size_t max_messages_;

  mutable std::mutex mutex_;
  std::condition_variable drained_;
  std::deque<BufferedEntry> entries_;
  std::size_t size_ = 0;
  bool closed_ = false;
  bool reader_finished_ = false;
};

// Everything the prefetch and pacing threads share for one run.
struct ReplayContext {
  ReplayContext(std::size_t max_messages, double playback_speed)
      : buffer(max_messages), stats(playback_speed) {}

  // Every topic being replayed, by the recording's channel id for it. Filled during construction;
  // afterwards only attached is set (during compose(), before the run's threads exist) and
  // subscribed (by the pacing thread alone), so no lock guards it.
  std::unordered_map<mcap::ChannelId, ReplayTopic> topics;
  ReplayBuffer buffer;
  ReplayStats stats;
  // Messages the prefetch thread has read from the recording, whether or not they were buffered.
  // Progress for prime() to watch, since skipping unattached topics and unreadable messages
  // reads without growing the buffer.
  std::atomic<std::uint64_t> messages_read{0};
};

}  // namespace isaac::os::mcap_replayer
