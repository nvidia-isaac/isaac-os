/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

#include "common/mcap_replayer/replay_stats.h"

namespace isaac::os::mcap_replayer {
namespace {

using namespace std::chrono_literals;

// True when any line of summary contains needle, so the assertions below pin the information a
// reader needs without pinning the exact wording of a sentence.
bool mentions(const std::vector<std::string>& summary, std::string_view needle) {
  for (const std::string& line : summary) {
    if (line.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

double milliseconds_of(std::chrono::nanoseconds value) {
  return std::chrono::duration<double, std::milli>(value).count();
}

// The recorded spacing the tests below replay at unless they say otherwise: ten milliseconds
// between messages, which is a hundred a second.
constexpr std::uint64_t kIntervalNs = 10'000'000;

// The recorded time of the index-th message of a stream at that spacing.
std::uint64_t recorded_at(std::uint64_t index) { return index * kIntervalNs; }

// An arbitrary fixed instant the fabricated emit times below count from, so every test is
// deterministic and none has to sleep.
constexpr std::chrono::steady_clock::time_point kStart{};

// The instant the index-th message goes out when a run keeps the recorded spacing exactly.
std::chrono::steady_clock::time_point emitted_at(std::uint64_t index) {
  return kStart + std::chrono::nanoseconds(index * kIntervalNs);
}

TEST(ReplayStats, CountsEmittedAndRefusedSeparatelyPerTopic) {
  ReplayStats stats;
  stats.record_emitted(stats.register_topic("pose"), 1ms, recorded_at(0), emitted_at(0));
  stats.record_emitted(stats.register_topic("pose"), 1ms, recorded_at(1), emitted_at(1));
  stats.record_refused_emit(stats.register_topic("pose"));
  stats.record_refused_emit(stats.register_topic("twist"));

  EXPECT_EQ(stats.emitted("pose"), 2U);
  EXPECT_EQ(stats.refused("pose"), 1U);
  EXPECT_EQ(stats.emitted("twist"), 0U);
  EXPECT_EQ(stats.refused("twist"), 1U);
}

TEST(ReplayStats, AnUnreportedTopicReadsAsZero) {
  const ReplayStats stats;

  EXPECT_EQ(stats.emitted("absent"), 0U);
  EXPECT_EQ(stats.refused("absent"), 0U);
  EXPECT_EQ(stats.worst_lag("absent"), 0ns);
  EXPECT_EQ(stats.mean_lag("absent"), 0ns);
}

TEST(ReplayStats, KeepsTheWorstLagApartFromTheMean) {
  ReplayStats stats;
  stats.record_emitted(stats.register_topic("pose"), 10ms, recorded_at(0), emitted_at(0));
  stats.record_emitted(stats.register_topic("pose"), 20ms, recorded_at(1), emitted_at(1));
  stats.record_emitted(stats.register_topic("pose"), 30ms, recorded_at(2), emitted_at(2));

  EXPECT_EQ(stats.worst_lag("pose"), 30ms);
  EXPECT_EQ(stats.mean_lag("pose"), 20ms);
}

// A lag that counts as late.
constexpr std::chrono::nanoseconds kLate = ReplayStats::kLateThreshold + 5ms;

// Runs action and returns what it logged. The warnings are logged by the stats themselves, so
// their rate limiting is only visible in the output.
template <typename Action>
std::string logged_by(Action&& action) {
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  action();
  return testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();
}

std::size_t occurrences_of(std::string_view text, std::string_view needle) {
  std::size_t count = 0;
  for (std::size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + needle.size())) {
    ++count;
  }
  return count;
}

TEST(ReplayStats, WarnsOnTheFirstRefusalThenLimitsEachTopicToOneWarningPerInterval) {
  ReplayStats stats;
  const ReplayStats::TopicId pose = stats.register_topic("pose");
  const ReplayStats::TopicId twist = stats.register_topic("twist");

  const std::string first =
      logged_by([&stats, pose] { stats.record_refused_emit(pose, emitted_at(0)); });
  EXPECT_EQ(occurrences_of(first, "Topic 'pose' could not publish"), 1U)
      << "the first refusal on a topic must always be reported; output was:\n"
      << first;

  const std::string flood = logged_by([&stats, pose] {
    for (int attempt = 1; attempt <= 20; ++attempt) {
      stats.record_refused_emit(pose, emitted_at(0) + attempt * 1ms);
    }
  });
  EXPECT_EQ(flood, "") << "a topic refusing at message rate must not log every refusal";
  EXPECT_EQ(stats.refused("pose"), 21U) << "every refusal is counted whether or not it is logged";

  // The allowance is per topic, so a topic that has not misbehaved yet still gets its own first
  // warning however badly another topic is behaving.
  const std::string other =
      logged_by([&stats, twist] { stats.record_refused_emit(twist, emitted_at(0)); });
  EXPECT_EQ(occurrences_of(other, "Topic 'twist' could not publish"), 1U) << other;

  // A problem that carries on is reported again once the interval has passed, and says how many
  // refusals the quiet stretch held.
  const std::string again = logged_by([&stats, pose] {
    stats.record_refused_emit(pose, emitted_at(0) + ReplayStats::kWarnInterval);
  });
  EXPECT_EQ(occurrences_of(again, "Topic 'pose' could not publish"), 1U) << again;
  EXPECT_NE(again.find("20 more like it"), std::string::npos)
      << "the warning must say how many it stood in for; output was:\n"
      << again;
}

TEST(ReplayStats, EachKindOfProblemKeepsItsOwnWarningWindowPerTopic) {
  ReplayStats stats;
  const ReplayStats::TopicId pose = stats.register_topic("pose");
  stats.record_refused_emit(pose, emitted_at(0));
  ASSERT_EQ(logged_by([&stats, pose] { stats.record_refused_emit(pose, emitted_at(0)); }), "");

  // A refused-emit problem (no subscriber connected, or a genuine publish failure) and a
  // behind-schedule problem (a message reached its consumer late) tell a reader different things
  // -- one says messages were lost, the other says the messages that did arrive no longer arrived
  // when the recording had them -- and neither is caused by the other under the current
  // kDropOldest default, where a slow consumer instead loses messages silently and uncounted.
  // Sharing one window would still let whichever kind fired first silence the other, which is
  // what this asserts against.
  const std::string late = logged_by([&stats, pose] {
    stats.record_emitted(pose, kLate, recorded_at(1), emitted_at(0));
    stats.record_emitted(pose, 1ms, recorded_at(0), emitted_at(1));  // the next pass starts
  });
  EXPECT_EQ(occurrences_of(late, "Topic 'pose': 1 message went out more than"), 1U) << late;
  const std::string unreadable =
      logged_by([&stats, pose] { stats.record_decode_failure(pose, "bad bytes", emitted_at(0)); });
  EXPECT_EQ(occurrences_of(unreadable, "could not be read back"), 1U) << unreadable;
  EXPECT_NE(unreadable.find("bad bytes"), std::string::npos)
      << "the decoder's reason must reach the warning; output was:\n"
      << unreadable;
}

// Delivers `lags` as one pass over the recording, one message each, the first at recorded time 0.
void deliver_pass(ReplayStats& stats, ReplayStats::TopicId topic,
                  const std::vector<std::chrono::nanoseconds>& lags, std::uint64_t first_emit = 0) {
  for (std::uint64_t index = 0; index < lags.size(); ++index) {
    stats.record_emitted(topic, lags[index], recorded_at(index), emitted_at(first_emit + index));
  }
}

// The one summary line that mentions how many went out late, or an empty string.
std::string late_line_of(const std::vector<std::string>& summary) {
  for (const std::string& line : summary) {
    if (line.find("went out more than") != std::string::npos) {
      return line;
    }
  }
  return "";
}

TEST(ReplayStats, DoesNotLogWhenAMessageGoesOutLateSoTheRunIsNotSlowedDown) {
  ReplayStats stats;
  const ReplayStats::TopicId pose = stats.register_topic("pose");

  const std::string output =
      logged_by([&stats, pose] { deliver_pass(stats, pose, {1ms, kLate, kLate, 1ms}); });

  EXPECT_EQ(output, "") << "late messages are listed in the summary, not logged one by one";
  EXPECT_EQ(stats.late_arrivals("pose"), 2U);
}

TEST(ReplayStats, SummaryListsTheNumbersOfTheMessagesThatWentOutLate) {
  ReplayStats stats;
  deliver_pass(stats, stats.register_topic("pose"), {kLate, 1ms, 1ms, kLate, 1ms});

  const std::string line = late_line_of(stats.summary_lines());

  EXPECT_NE(line.find("2 went out more than"), std::string::npos) << line;
  EXPECT_NE(line.find("(messages #1, #4)"), std::string::npos)
      << "the summary must say which messages were late; line was: " << line;
}

TEST(ReplayStats, SummaryNamesASingleLateMessageAsOne) {
  ReplayStats stats;
  deliver_pass(stats, stats.register_topic("pose"), {1ms, kLate, 1ms});

  const std::string line = late_line_of(stats.summary_lines());

  EXPECT_NE(line.find("1 went out more than"), std::string::npos) << line;
  EXPECT_NE(line.find("(message #2)"), std::string::npos) << line;
}

TEST(ReplayStats, SummaryCollapsesConsecutiveLateMessagesAndCapsTheList) {
  ReplayStats stats;
  const ReplayStats::TopicId pose = stats.register_topic("pose");
  // Late: 1-3 together, then every other message from 5 to 29, which is more runs than are kept.
  std::vector<std::chrono::nanoseconds> lags(30, 1ms);
  for (const std::size_t position : {1U, 2U, 3U}) {
    lags[position - 1] = kLate;
  }
  for (std::size_t position = 5; position <= 29; position += 2) {
    lags[position - 1] = kLate;
  }
  deliver_pass(stats, pose, lags);

  const std::string line = late_line_of(stats.summary_lines());

  EXPECT_NE(line.find("16 went out more than"), std::string::npos) << line;
  EXPECT_NE(line.find("#1-#3, #5, #7, #9, #11, #13, #15, #17, #19, #21,"), std::string::npos)
      << "the first ten runs are listed, consecutive messages collapsed; line was: " << line;
  EXPECT_NE(line.find("and 4 more)"), std::string::npos)
      << "the rest are counted; line was: " << line;
}

TEST(ReplayStats, SummaryHasNoListWhenNothingWentOutLate) {
  ReplayStats stats;
  deliver_pass(stats, stats.register_topic("pose"), {1ms, 2ms, 1ms});

  EXPECT_EQ(late_line_of(stats.summary_lines()), "");
}

TEST(ReplayStats, LoggingTheLateMessagesOfAPassWhenALoopingRecordingStartsOver) {
  ReplayStats stats;
  const ReplayStats::TopicId pose = stats.register_topic("pose");

  // The first pass has two late messages, and the second starts over with an on-time one.
  const std::string output = logged_by([&stats, pose] {
    deliver_pass(stats, pose, {1ms, kLate, kLate, 1ms});
    deliver_pass(stats, pose, {1ms, 1ms}, /*first_emit=*/4);
  });

  EXPECT_EQ(occurrences_of(output, "Topic 'pose': 2 messages went out more than"), 1U) << output;
  EXPECT_NE(output.find("(messages #2-#3)"), std::string::npos) << output;
  EXPECT_NE(output.find("in the last pass over the recording"), std::string::npos) << output;
}

TEST(ReplayStats, StaysQuietWhenAPassHadNoLateMessagesAtTheLoop) {
  ReplayStats stats;
  const ReplayStats::TopicId pose = stats.register_topic("pose");

  const std::string output = logged_by([&stats, pose] {
    deliver_pass(stats, pose, {1ms, 1ms, 1ms});
    deliver_pass(stats, pose, {1ms, 1ms, 1ms}, /*first_emit=*/3);
  });

  EXPECT_EQ(output, "");
}

TEST(ReplayStats, AFinalPassSummaryAccountsForEarlierPassesThatWereAlreadyLogged) {
  ReplayStats stats;
  const ReplayStats::TopicId pose = stats.register_topic("pose");

  // Two late messages in the first pass (logged when it ended), one in the second.
  static_cast<void>(logged_by([&stats, pose] {
    deliver_pass(stats, pose, {kLate, kLate, 1ms});
    deliver_pass(stats, pose, {1ms, 1ms, kLate}, /*first_emit=*/3);
  }));

  const std::string line = late_line_of(stats.summary_lines());

  EXPECT_NE(line.find("3 went out more than"), std::string::npos)
      << "the total covers every pass; line was: " << line;
  EXPECT_NE(line.find("(message #3 in the final pass)"), std::string::npos)
      << "only the final pass's numbers are listed; line was: " << line;
}

TEST(ReplayStats, LimitsTheLoopLinesToOnePerIntervalAndCountsTheLateMessagesItHeldBack) {
  ReplayStats stats;
  const ReplayStats::TopicId pose = stats.register_topic("pose");

  // Six passes of a late message then an on-time one, all inside one interval: each wrap reports
  // the pass before it, but only the first of those reports is logged.
  const std::string burst = logged_by([&stats, pose] {
    for (std::uint64_t pass = 0; pass < 6; ++pass) {
      stats.record_emitted(pose, kLate, recorded_at(0), emitted_at(2 * pass));
      stats.record_emitted(pose, 1ms, recorded_at(1), emitted_at(2 * pass + 1));
    }
  });
  EXPECT_EQ(occurrences_of(burst, "went out more than"), 1U)
      << "a topic that wraps at message rate must not log every pass; output was:\n"
      << burst;

  // The next report after the interval says how many late messages the quiet stretch held.
  const std::string later = logged_by([&stats, pose] {
    stats.record_emitted(pose, kLate, recorded_at(0), emitted_at(2) + ReplayStats::kWarnInterval);
  });
  EXPECT_EQ(occurrences_of(later, "went out more than"), 1U) << later;
  EXPECT_NE(later.find("4 more like it"), std::string::npos) << later;
}

TEST(ReplayStats, CallsAMessageLateOnlyWhenItWentOutMoreThanTheThresholdAfterItWasDue) {
  ReplayStats stats;

  stats.record_emitted(stats.register_topic("pose"), ReplayStats::kLateThreshold, recorded_at(0),
                       emitted_at(0));
  EXPECT_EQ(stats.late_arrivals("pose"), 0U) << "exactly the threshold is still on time";

  stats.record_emitted(stats.register_topic("pose"), ReplayStats::kLateThreshold + 1ns,
                       recorded_at(1), emitted_at(1));
  EXPECT_EQ(stats.late_arrivals("pose"), 1U);
}

// The threshold is about the replay's own timing, so neither the topic's rate, nor the playback
// speed, nor whether it is the first message changes what counts as late.
TEST(ReplayStats, JudgesLatenessTheSameWhateverTheRateTheSpeedOrTheMessage) {
  ReplayStats fast_replay(10.0);
  ReplayStats slow_replay(0.1);

  fast_replay.record_emitted(fast_replay.register_topic("pose"), 5s, recorded_at(0), emitted_at(0));
  slow_replay.record_emitted(slow_replay.register_topic("pose"), 5s, recorded_at(0), emitted_at(0));
  fast_replay.record_emitted(fast_replay.register_topic("slow_topic"), 9ms, 0, kStart);
  fast_replay.record_emitted(fast_replay.register_topic("slow_topic"), 9ms, 1'000'000'000,
                             kStart + 100ms);

  EXPECT_EQ(fast_replay.late_arrivals("pose"), 1U) << "the first message counts too";
  EXPECT_EQ(slow_replay.late_arrivals("pose"), 1U);
  EXPECT_EQ(fast_replay.late_arrivals("slow_topic"), 0U);
}

TEST(ReplayStats, SummarisesAHealthyRunRatherThanStayingSilent) {
  ReplayStats stats;
  stats.record_emitted(stats.register_topic("pose"), 1ms, recorded_at(0), emitted_at(0));
  stats.record_emitted(stats.register_topic("pose"), 3ms, recorded_at(1), emitted_at(1));

  const std::vector<std::string> summary = stats.summary_lines();

  ASSERT_FALSE(summary.empty());
  EXPECT_TRUE(mentions(summary, "pose"));
  EXPECT_TRUE(mentions(summary, "delivered 2 messages"));
  EXPECT_TRUE(mentions(summary, "on schedule"))
      << "a run that lost nothing must still say so, so a reader has a healthy baseline";
}

TEST(ReplayStats, SummarisesNothingAsNothingRatherThanAsAHealthyRun) {
  const ReplayStats stats;

  const std::vector<std::string> summary = stats.summary_lines();

  ASSERT_FALSE(summary.empty()) << "a run that delivered nothing must still report that it did";
  EXPECT_FALSE(mentions(summary, "on schedule"))
      << "a run that delivered nothing kept no schedule, so claiming it did would be true only "
         "because there was nothing to be wrong about";
}

// A run that stopped because a consumer never subscribed also delivered nothing, and must not be
// summarised as one whose recording held nothing: that is the one confusion this state exists to
// rule out.
TEST(ReplayStats, SummarisesARunWhoseConsumerNeverSubscribedAsOneThatNeverBegan) {
  ReplayStats stats;
  stats.record_never_subscribed(stats.register_topic("pose"));

  const std::vector<std::string> summary = stats.summary_lines();

  EXPECT_TRUE(stats.never_subscribed("pose"));
  EXPECT_FALSE(stats.never_subscribed("twist"));
  EXPECT_EQ(stats.refused("pose"), 0U) << "a run that never began refused nothing";
  EXPECT_TRUE(mentions(summary, "Its consumer never subscribed"));
  EXPECT_TRUE(mentions(summary, "stopped before replaying anything"));
  EXPECT_FALSE(mentions(summary, "The recording held none"));
  EXPECT_FALSE(mentions(summary, "on schedule"));
}

// Like a consumer that never subscribed, a stalled start-up delivered nothing and must not read as
// an empty recording.
TEST(ReplayStats, SummarisesARunWhoseStartUpStalledAsOneThatNeverBegan) {
  ReplayStats stats;
  EXPECT_FALSE(stats.start_stalled());
  stats.record_start_stalled();

  const std::vector<std::string> summary = stats.summary_lines();

  EXPECT_TRUE(stats.start_stalled());
  EXPECT_TRUE(mentions(summary, "reading the recording stalled"));
  EXPECT_FALSE(mentions(summary, "The recording held none"));
  EXPECT_FALSE(mentions(summary, "on schedule"));
}

TEST(ReplayStats, SummarisesARunWhoseReadingFailedPartwayAsOneThatEndedEarly) {
  ReplayStats stats;
  stats.record_emitted(stats.register_topic("pose"), 1ms, recorded_at(0), emitted_at(0));
  EXPECT_FALSE(stats.read_failed());
  stats.record_read_failed();

  const std::vector<std::string> summary = stats.summary_lines();

  EXPECT_TRUE(stats.read_failed());
  EXPECT_TRUE(mentions(summary, "Reading the recording failed partway through"));
  EXPECT_FALSE(mentions(summary, "on schedule"))
      << "a run cut short by a failed read must not be reported as a healthy one";
}

TEST(ReplayStats, SummaryReportsHowManyMessagesTheConsumerCouldNotAccept) {
  ReplayStats stats;
  stats.record_emitted(stats.register_topic("pose"), 1ms, recorded_at(0), emitted_at(0));
  stats.record_refused_emit(stats.register_topic("pose"));
  stats.record_refused_emit(stats.register_topic("pose"));

  const std::vector<std::string> summary = stats.summary_lines();

  EXPECT_TRUE(mentions(summary, "pose"));
  EXPECT_TRUE(mentions(summary, "delivered 1 message"));
  EXPECT_TRUE(mentions(summary, "2 were skipped"));
  EXPECT_FALSE(mentions(summary, "on schedule"))
      << "a run that skipped messages must not be reported as a healthy one";
}

TEST(ReplayStats, SummaryReportsMessagesThatCouldNotBeReadBackFromTheRecording) {
  ReplayStats stats;
  stats.record_emitted(stats.register_topic("pose"), 1ms, recorded_at(0), emitted_at(0));
  stats.record_decode_failure(stats.register_topic("pose"), "unreadable");
  stats.record_decode_failure(stats.register_topic("pose"), "unreadable");

  const std::vector<std::string> summary = stats.summary_lines();

  EXPECT_EQ(stats.decode_failures("pose"), 2U);
  EXPECT_TRUE(mentions(summary, "pose"));
  EXPECT_TRUE(mentions(summary, "2 were skipped"));
  EXPECT_TRUE(mentions(summary, "could not be read back"));
  EXPECT_FALSE(mentions(summary, "delivered 3 messages"))
      << "the summary never sees how many messages the recording holds, so it must count what "
         "this run delivered rather than folding the unreadable ones back into a total it "
         "presents as the recording's";
  EXPECT_FALSE(mentions(summary, "on schedule"))
      << "a run that lost messages to a recording it could not read must not be reported as a "
         "healthy one";
}

TEST(ReplayStats, SummaryDoesNotCallARunHealthyWhenItRanBehindTheRecording) {
  ReplayStats stats;
  stats.record_emitted(stats.register_topic("pose"), 1ms, recorded_at(0), emitted_at(0));
  stats.record_emitted(stats.register_topic("pose"), 40ms, recorded_at(1), emitted_at(1));
  stats.record_emitted(stats.register_topic("pose"), 40ms, recorded_at(2), emitted_at(2));

  const std::vector<std::string> summary = stats.summary_lines();

  EXPECT_EQ(stats.late_arrivals("pose"), 2U);
  EXPECT_TRUE(mentions(summary, "behind"));
  EXPECT_FALSE(mentions(summary, "on schedule"))
      << "a run that delivered every message but delivered them later than the late threshold did "
         "not keep "
         "the recording's schedule, and saying it did is the one thing a timing report must not "
         "get wrong";
}

// A stream recorded twenty times a second, fed to stats as if emitted at the given spacing: the
// index-th message carries the index-th recorded time and goes out index spacings after the first.
// Every lag is passed as zero, which is what a run meeting each of its targets reports whatever
// those targets were.
void emit_twenty_hertz(ReplayStats& stats, std::string_view topic, std::size_t count,
                       std::chrono::nanoseconds spacing) {
  constexpr std::uint64_t kTwentyHertzNs = 50'000'000;
  for (std::size_t index = 0; index < count; ++index) {
    stats.record_emitted(stats.register_topic(topic), 0ns, index * kTwentyHertzNs,
                         kStart + spacing * static_cast<std::int64_t>(index));
  }
}

TEST(ReplayStats, AReplayKeepingTheRecordedSpacingReadsAsTheRecordedRateWithNoScheduleError) {
  ReplayStats stats;
  emit_twenty_hertz(stats, "pose", 40, 50ms);

  ASSERT_TRUE(stats.recorded_hz("pose").has_value());
  ASSERT_TRUE(stats.replayed_hz("pose").has_value());
  EXPECT_DOUBLE_EQ(*stats.recorded_hz("pose"), 20.0);
  EXPECT_DOUBLE_EQ(*stats.replayed_hz("pose"), 20.0);
  EXPECT_EQ(stats.worst_schedule_error("pose"), 0ns);
  EXPECT_EQ(stats.typical_schedule_error("pose"), 0ns);
}

TEST(ReplayStats, ScalesTheReplayedRateBackToRecordedTimeAtAnyPlaybackSpeed) {
  ReplayStats stats(2.0);

  // Twice the recorded speed: 20 Hz recorded goes out 25 ms apart, which is the recording kept.
  emit_twenty_hertz(stats, "pose", 40, 25ms);

  ASSERT_TRUE(stats.replayed_hz("pose").has_value());
  EXPECT_DOUBLE_EQ(*stats.replayed_hz("pose"), *stats.recorded_hz("pose"));
  EXPECT_EQ(stats.worst_schedule_error("pose"), 0ns);
}

// The reason these measures exist. A run that meets every target it is given reports no lag at
// all, so a target computed wrongly -- here, one that runs the recording at twice its speed --
// looks exactly like a perfect run to the lag. Measured against the recording's own times
// instead, the same run shows up as both the wrong rate and a growing error.
TEST(ReplayStats, NoticesAWrongScheduleThatTheLagCannotSee) {
  ReplayStats stats;
  emit_twenty_hertz(stats, "short", 11, 25ms);
  emit_twenty_hertz(stats, "long", 41, 25ms);

  EXPECT_EQ(stats.mean_lag("long"), 0ns);
  EXPECT_EQ(stats.worst_lag("long"), 0ns);
  ASSERT_TRUE(stats.replayed_hz("long").has_value());
  EXPECT_NEAR(*stats.replayed_hz("long") / *stats.recorded_hz("long"), 2.0, 1e-9);
  // Each message lands 25 ms further ahead of where the recording's spacing puts it than the one
  // before it, so the errors ramp from 0 down to 250 ms ahead after ten intervals and a second
  // ahead after forty. The spread of a ramp is its whole length, and its standard deviation is
  // that of evenly spaced values: 25 ms times the square root of (n * n - 1) / 12 over n errors.
  EXPECT_EQ(stats.worst_schedule_error("short"), 250ms);
  EXPECT_EQ(stats.worst_schedule_error("long"), 1000ms);
  ASSERT_TRUE(stats.typical_schedule_error("short").has_value());
  ASSERT_TRUE(stats.typical_schedule_error("long").has_value());
  EXPECT_NEAR(milliseconds_of(*stats.typical_schedule_error("short")),
              25.0 * std::sqrt((11.0 * 11.0 - 1.0) / 12.0), 1e-3);
  EXPECT_NEAR(milliseconds_of(*stats.typical_schedule_error("long")),
              25.0 * std::sqrt((41.0 * 41.0 - 1.0) / 12.0), 1e-3);
}

// One message out of line with the rest is one message's error, however early in the pass it
// comes. The first message is the one every other is placed against, so a delay on it alone must
// not read as every later message being early.
TEST(ReplayStats, ChargesALateFirstMessageToThatMessageAlone) {
  ReplayStats stats;
  constexpr std::uint64_t kMessages = 40;
  stats.record_emitted(stats.register_topic("pose"), 5ms, recorded_at(0), emitted_at(0) + 5ms);
  for (std::uint64_t index = 1; index < kMessages; ++index) {
    stats.record_emitted(stats.register_topic("pose"), 0ns, recorded_at(index), emitted_at(index));
  }

  EXPECT_EQ(stats.worst_schedule_error("pose"), 5ms);
  // Thirty-nine errors of -5 ms and one of zero: a standard deviation of 5 ms times the square
  // root of (1/40)(39/40), about 0.78 ms.
  ASSERT_TRUE(stats.typical_schedule_error("pose").has_value());
  EXPECT_LT(*stats.typical_schedule_error("pose"), 1ms);
  EXPECT_NEAR(milliseconds_of(*stats.typical_schedule_error("pose")),
              5.0 * std::sqrt((1.0 / 40.0) * (39.0 / 40.0)), 1e-3);
}

// A delay shared by every message of a pass, the first included, moves the pass without changing
// its spacing, which is the one thing these measures are about. Lateness against the due time is
// what reports a delay like that.
TEST(ReplayStats, DoesNotCountADelayCommonToAWholePass) {
  ReplayStats stats;
  const std::vector<std::chrono::nanoseconds> pass_delays{0ms, 5ms, 12ms};
  auto pass_start = kStart;
  for (const std::chrono::nanoseconds delay : pass_delays) {
    for (std::uint64_t index = 0; index < 5; ++index) {
      stats.record_emitted(stats.register_topic("pose"), delay, recorded_at(index),
                           pass_start + delay + std::chrono::nanoseconds(index * kIntervalNs));
    }
    pass_start += 100ms;
  }

  EXPECT_EQ(stats.worst_schedule_error("pose"), 0ns);
  EXPECT_EQ(stats.typical_schedule_error("pose"), 0ns);
  EXPECT_EQ(stats.worst_lag("pose"), 12ms);
}

TEST(ReplayStats, StartsTheScheduleOverWhereTheRecordingDoes) {
  ReplayStats stats;
  // One pass of five messages kept exactly, then the recording starts over and its first message
  // goes out 30 ms after the last -- a pause no recorded interval accounts for, and one that
  // does not count against either pass -- followed by the second pass kept exactly.
  for (std::uint64_t index = 0; index < 5; ++index) {
    stats.record_emitted(stats.register_topic("pose"), 0ns, recorded_at(index), emitted_at(index));
  }
  const auto second_pass = emitted_at(4) + 30ms;
  for (std::uint64_t index = 0; index < 5; ++index) {
    stats.record_emitted(stats.register_topic("pose"), 0ns, recorded_at(index),
                         second_pass + std::chrono::nanoseconds(index * kIntervalNs));
  }

  EXPECT_EQ(stats.worst_schedule_error("pose"), 0ns)
      << "the pause at the start of a new pass is not an error in either pass's spacing";
  ASSERT_TRUE(stats.replayed_hz("pose").has_value());
  EXPECT_DOUBLE_EQ(*stats.recorded_hz("pose"), 100.0);
  EXPECT_DOUBLE_EQ(*stats.replayed_hz("pose"), 100.0)
      << "the step back to the start of the recording is not an interval of either rate";
}

TEST(ReplayStats, AMessageThatWasNeverDeliveredDoesNotSkewTheRateComparison) {
  ReplayStats stats;
  for (std::uint64_t index = 0; index < 20; ++index) {
    if (index == 7) {
      stats.record_refused_emit(stats.register_topic("pose"));
      continue;
    }
    stats.record_emitted(stats.register_topic("pose"), 0ns, recorded_at(index), emitted_at(index));
  }

  ASSERT_TRUE(stats.replayed_hz("pose").has_value());
  EXPECT_DOUBLE_EQ(*stats.replayed_hz("pose") / *stats.recorded_hz("pose"), 1.0);
  EXPECT_EQ(stats.worst_schedule_error("pose"), 0ns);
}

TEST(ReplayStats, HasNoRateOrScheduleErrorToReportWithoutAPairOfMessages) {
  ReplayStats stats;
  stats.record_emitted(stats.register_topic("single"), 0ns, recorded_at(0), emitted_at(0));
  // Every message at one recorded instant: a recording with no cadence has no rate.
  stats.record_emitted(stats.register_topic("flat"), 0ns, 0, emitted_at(0));
  stats.record_emitted(stats.register_topic("flat"), 0ns, 0, emitted_at(1));

  for (const char* topic : {"single", "flat", "absent"}) {
    EXPECT_FALSE(stats.recorded_hz(topic).has_value()) << topic;
    EXPECT_FALSE(stats.replayed_hz(topic).has_value()) << topic;
  }
  // A recording with no cadence has no spacing to keep. Its recorded time never steps backwards
  // either, so under a looping replay its passes cannot be told apart and any error taken on it
  // would grow by a whole pass on every repeat.
  for (const char* topic : {"single", "flat", "absent"}) {
    EXPECT_FALSE(stats.worst_schedule_error(topic).has_value()) << topic;
    EXPECT_FALSE(stats.typical_schedule_error(topic).has_value()) << topic;
  }
}

TEST(ReplayStats, SummaryGivesTheReplayedAndRecordedRateOfEachTopic) {
  ReplayStats stats;
  emit_twenty_hertz(stats, "pose", 10, 50ms);

  const std::vector<std::string> summary = stats.summary_lines();

  EXPECT_TRUE(mentions(summary, "Replayed at 20.00 Hz, recorded at 20.00 Hz"));
  EXPECT_TRUE(mentions(summary, "within 0.0 ms typically and 0.0 ms at worst"));
  EXPECT_TRUE(mentions(summary, "on schedule"))
      << "reporting the rate must not change when a run is called healthy";
}

TEST(ReplayStats, SummaryGivesBothTheRecordedAndTheDeliveredRateAtAnyOtherSpeed) {
  ReplayStats stats(2.0);

  emit_twenty_hertz(stats, "pose", 10, 25ms);

  const std::vector<std::string> summary = stats.summary_lines();

  EXPECT_TRUE(mentions(summary, "Replayed at 20.00 Hz in recorded time"));
  EXPECT_TRUE(mentions(summary, "40.00 Hz as delivered at 2x speed"));
  EXPECT_TRUE(mentions(summary, "as delivered, messages kept the recording's own spacing"))
      << "the spacing is measured in delivered time, and the sentence must not leave that to be "
         "guessed from the rate beside it";
}

TEST(ReplayStats, SummaryGivesTheSpacingAloneWhenThereIsNoReplayedRate) {
  ReplayStats stats;
  // Recorded 10 ms apart but all sent at one instant: a spacing to have kept, and no replayed
  // interval to take a rate over.
  for (std::uint64_t index = 0; index < 3; ++index) {
    stats.record_emitted(stats.register_topic("pose"), 0ns, recorded_at(index), kStart);
  }

  const std::vector<std::string> summary = stats.summary_lines();

  EXPECT_FALSE(stats.replayed_hz("pose").has_value());
  EXPECT_TRUE(mentions(summary, "  Messages kept the recording's own spacing to within"));
  EXPECT_TRUE(mentions(summary, "20.0 ms at worst"));
  EXPECT_FALSE(mentions(summary, "Hz"));
}

TEST(ReplayStats, SummarySaysNothingAboutSpacingForARecordingWithNoCadence) {
  ReplayStats stats;
  for (std::uint64_t index = 0; index < 3; ++index) {
    stats.record_emitted(stats.register_topic("static_transform"), 0ns, 0, emitted_at(index));
  }

  const std::vector<std::string> summary = stats.summary_lines();

  EXPECT_TRUE(mentions(summary, "delivered 3 messages"));
  EXPECT_FALSE(mentions(summary, "spacing"));
  EXPECT_FALSE(mentions(summary, "Hz"));
}

}  // namespace
}  // namespace isaac::os::mcap_replayer
