/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"

#include "common/mcap_replayer/message.h"
#include "common/mcap_replayer/pacing.h"
#include "common/mcap_replayer/replay_context.h"

namespace {

using isaac::os::mcap_replayer::BufferedEntry;
using isaac::os::mcap_replayer::Message;
using isaac::os::mcap_replayer::PacingClock;
using isaac::os::mcap_replayer::ReplayBuffer;

// Every entry carries this same recorded time. The clock's first target_for() latches it to the
// instant the clock was constructed, which is already in the past by the time anything asks, so
// every entry is due the moment it reaches the front and no test here has to wait on real time.
constexpr std::uint64_t kLogTimeNs = 1000;

// A buffered entry that is never published: these tests only move entries through the buffer. The
// entries are told apart by their channel ids.
BufferedEntry make_entry(mcap::ChannelId channel_id) {
  auto message = std::make_unique<Message>();
  message->log_time_ns = kLogTimeNs;
  return BufferedEntry{.message = std::move(message), .channel_id = channel_id};
}

}  // namespace

TEST(ReplayBuffer, AnEmptyBufferDoesNotReportTheReaderFinishedUntilItIsMarked) {
  ReplayBuffer buffer(4);
  PacingClock clock(1.0);
  std::optional<std::uint64_t> previous_log_time;

  const auto outcome = buffer.release_if_due(clock, previous_log_time);
  EXPECT_FALSE(outcome.released.has_value());
  EXPECT_FALSE(outcome.sleep_until.has_value());
  EXPECT_FALSE(outcome.reader_finished);
}

// The reader marks itself finished right after its last push, and the timing side must release
// every entry pushed before that and only then learn the reader is done. Here the finish is marked
// while an entry is still buffered -- and a further entry is pushed even after it -- to show the
// buffer itself guarantees this, independent of the order its caller happens to use: a buffer
// that still holds anything always releases it rather than reporting the reader finished, and the
// finish is only ever reported by the same locked look that found the buffer empty.
TEST(ReplayBuffer, ReaderFinishedIsOnlyReportedOnceEveryPushedEntryHasBeenReleased) {
  ReplayBuffer buffer(4);
  PacingClock clock(1.0);
  std::optional<std::uint64_t> previous_log_time;

  ASSERT_TRUE(buffer.push(make_entry(1)));
  ASSERT_TRUE(buffer.push(make_entry(2)));

  // Ordinary draining: both come back in order, and the emptied buffer does not yet claim the
  // reader has finished.
  for (const mcap::ChannelId expected_channel : {1, 2}) {
    auto outcome = buffer.release_if_due(clock, previous_log_time);
    ASSERT_TRUE(outcome.released.has_value()) << "entry " << expected_channel;
    EXPECT_EQ(outcome.released->channel_id, expected_channel);
    EXPECT_FALSE(outcome.reader_finished);
    previous_log_time = outcome.released->message->log_time_ns;
  }
  {
    const auto outcome = buffer.release_if_due(clock, previous_log_time);
    EXPECT_FALSE(outcome.released.has_value());
    EXPECT_FALSE(outcome.sleep_until.has_value());
    EXPECT_FALSE(outcome.reader_finished);
  }

  // The reader finishes, and one more entry lands anyway.
  buffer.mark_reader_finished();
  ASSERT_TRUE(buffer.push(make_entry(3)));

  // That entry is released, not dropped behind a report that the reader is done.
  {
    auto outcome = buffer.release_if_due(clock, previous_log_time);
    ASSERT_TRUE(outcome.released.has_value());
    EXPECT_EQ(outcome.released->channel_id, 3);
    EXPECT_FALSE(outcome.sleep_until.has_value());
    EXPECT_FALSE(outcome.reader_finished);
    previous_log_time = outcome.released->message->log_time_ns;
  }

  // Only now, with the buffer empty, is the finish reported -- and it stays reported.
  for (int look = 0; look < 2; ++look) {
    const auto outcome = buffer.release_if_due(clock, previous_log_time);
    EXPECT_FALSE(outcome.released.has_value()) << "look " << look;
    EXPECT_FALSE(outcome.sleep_until.has_value()) << "look " << look;
    EXPECT_TRUE(outcome.reader_finished) << "look " << look;
  }
  EXPECT_EQ(buffer.size(), 0U);
}

TEST(ReplayBuffer, MarkingTheReaderFinishedMoreThanOnceIsHarmless) {
  ReplayBuffer buffer(4);
  PacingClock clock(1.0);
  std::optional<std::uint64_t> previous_log_time;

  buffer.mark_reader_finished();
  buffer.mark_reader_finished();

  const auto outcome = buffer.release_if_due(clock, previous_log_time);
  EXPECT_FALSE(outcome.released.has_value());
  EXPECT_FALSE(outcome.sleep_until.has_value());
  EXPECT_TRUE(outcome.reader_finished);
}
