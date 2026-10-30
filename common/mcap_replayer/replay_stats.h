/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace isaac::os::mcap_replayer {

// What a replay run can go wrong in. Each kind is rate limited separately, so a problem at
// message rate cannot use up the allowance another kind needs to be reported once.
enum class ReplayProblem {
  // A message could not be published at all: nothing is subscribed to the topic, or the publish
  // failed outright. Not "the consumer was too slow": under attach()'s kDropOldest policy a slow
  // consumer loses messages silently inside the transport, invisible here.
  kRefusedEmit,
  // Recorded bytes could not be read back as the type their topic is replayed as.
  kUnreadableMessage,
  // Messages went out more than ReplayStats::kLateThreshold after they were due, which points at
  // the machine: something delayed the replay between waking up and publishing. Reported when a
  // pass over a looping recording ends, and in the summary.
  kBehindSchedule,
  // Not a problem: the number of kinds, which sizes the per-kind allowance.
  kCount,
};

// Per-topic counters and reporting for one replay run.
//
// A message a run reads but does not publish is attributable to exactly one of two causes, both
// counted here: its bytes could not be decoded as the topic's type, or the publish failed. A full
// buffer is not a third cause, since the prefetch thread blocks rather than discarding.
//
// Published is not consumed. Under attach()'s kDropOldest policy, a consumer slower than the
// recording loses messages inside the transport after they were counted as published, and no
// counter here can tell that from a consumer that read every one.
//
// Every method is safe to call from several threads at once without external synchronization.
class ReplayStats {
 public:
  // A topic's index in this object, from register_topic(). Recording by index keeps the pacing
  // thread from building and hashing a topic name for every message.
  using TopicId = std::uint32_t;

  // A message that goes out this much after it was due is late. A fixed amount rather than a
  // share of the topic's message spacing, because it is meant to catch the replay's own timing
  // going wrong between waking up and publishing, whatever the recording's rate or speed.
  static constexpr std::chrono::milliseconds kLateThreshold{10};

  // The warning for one kind of problem on one topic is logged at most this often. A problem
  // that happens once tends to recur at message rate, and logging each time would flood the
  // terminal and slow the path already under strain. A time window rather than a fixed number of
  // warnings, so a burst at start-up cannot use up the reports of a later problem. Per topic, to
  // bound a single unhappy topic, and per kind, so one problem cannot silence another (a strained
  // run commonly has both refused and late messages, and the first report of each matters most).
  static constexpr std::chrono::seconds kWarnInterval{1};

  // playback_speed is recorded seconds per wall-clock second, which turns recorded gaps into the
  // wall-clock gaps a faithful replay would have.
  explicit ReplayStats(double playback_speed = 1.0) : playback_speed_(playback_speed) {}

  // The index to record topic's events under, the same one each time it is asked for. Topics are
  // registered before the run's threads start.
  [[nodiscard]] TopicId register_topic(std::string_view topic);

  // Records one delivered message: its lag (how late it went out relative to its due time), the
  // time it carried in the recording, and the instant it went out.
  //
  // The recorded time and emit instant measure the run against the recording alone (spacing kept,
  // rates) without using the due time that lag is taken against: a wrongly computed due time would
  // be met just as punctually, and only a comparison that never consults it can notice. Passed in
  // so the caller's one clock reading serves both measures. A lag over kLateThreshold is counted
  // as a late arrival, and its position among the messages delivered in the current pass is kept
  // for the summary, so a reader can tell a few late messages at start-up from lateness through the
  // whole run. When this message starts a new pass of a looping recording, the late messages of the
  // pass that just ended are logged, at most once per kWarnInterval per topic.
  void record_emitted(TopicId topic, std::chrono::nanoseconds schedule_lag,
                      std::uint64_t recorded_time_ns,
                      std::chrono::steady_clock::time_point emitted_at);

  // Records one message that could not be published -- see ReplayProblem::kRefusedEmit for what
  // that does and does not cover. Logs a warning, at most once per kWarnInterval per topic. now is
  // when the refusal happened.
  void record_refused_emit(
      TopicId topic, std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

  // Records one recorded message that could not be read back as the type its topic is replayed
  // as, and so never became a message to deliver at all. reason is the decoder's explanation,
  // included in the warning, which is logged at most once per kWarnInterval per topic. now is when
  // the failure happened.
  void record_decode_failure(
      TopicId topic, std::string_view reason,
      std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

  // Records that a topic's consumer never subscribed before the start-up wait ran out, which ends
  // the run before it replays anything. Separate from the refused count: a refusal is one message
  // that found nobody, while this is a run that never began, and the summary must distinguish "the
  // consumer never connected" from "the recording holds nothing on this topic".
  void record_never_subscribed(TopicId topic);

  // Records that reading the recording stalled while the run was starting, which ends the run
  // before it replays anything. Run-wide, not per topic, because the reader is shared by all.
  void record_start_stalled();
  [[nodiscard]] bool start_stalled() const;

  // Records that reading the recording failed partway through, so the run ends before the whole
  // recording has been replayed. Run-wide, like record_start_stalled().
  void record_read_failed();
  [[nodiscard]] bool read_failed() const;

  // Logs the end-of-run summary. Always produces output, including for a healthy run, so there is
  // a baseline to compare a bad run against.
  void log_summary() const;

  // The same text log_summary() writes, one entry per line.
  [[nodiscard]] std::vector<std::string> summary_lines() const;

  [[nodiscard]] std::size_t emitted(std::string_view topic) const;
  [[nodiscard]] std::size_t refused(std::string_view topic) const;
  [[nodiscard]] std::size_t decode_failures(std::string_view topic) const;

  // Whether the run stopped before replaying anything because this topic's consumer never
  // subscribed; see record_never_subscribed().
  [[nodiscard]] bool never_subscribed(std::string_view topic) const;

  // How many delivered messages on this topic went out more than kLateThreshold after they were
  // due.
  [[nodiscard]] std::size_t late_arrivals(std::string_view topic) const;

  // The largest lag seen on this topic and the mean over all delivered messages (the summary's
  // "late on average" figure). Both are zero for a topic that delivered nothing. The worst is kept
  // beside the mean because after one long stall the mean describes no message that happened.
  [[nodiscard]] std::chrono::nanoseconds worst_lag(std::string_view topic) const;
  [[nodiscard]] std::chrono::nanoseconds mean_lag(std::string_view topic) const;

  // How well delivered messages kept the recording's spacing. Within a pass over the recording,
  // each message has a signed error: its delivered time since the pass's first message, less its
  // recorded time since that message divided by the playback speed (zero for the first message).
  //
  // Both readings use the spread of that series within a pass, not its level, so a delay common
  // to a whole pass (including a late first message alone) is not counted. A clock running at the
  // wrong speed still shows, because its errors ramp across the pass.
  //
  // worst_schedule_error() is the largest peak-to-peak spread of one pass: from the message that
  // landed earliest relative to the recording to the one that landed latest.
  // typical_schedule_error() is the standard deviation about each pass's own mean, pooled over
  // all passes. Both are in delivered (wall-clock) time: at twice the recording's speed, 1 ms is
  // 2 ms of recorded time.
  //
  // A pass ends where recorded time steps backwards (a looping recording starting over). Empty
  // until some pass has delivered two messages, and for a topic with no cadence (every message at
  // one recorded time, or one message per pass of a looping replay, whose passes cannot then be
  // told apart).
  [[nodiscard]] std::optional<std::chrono::nanoseconds> worst_schedule_error(
      std::string_view topic) const;
  [[nodiscard]] std::optional<std::chrono::nanoseconds> typical_schedule_error(
      std::string_view topic) const;

  // Messages per second the topic was recorded at and replayed at, over the same pairs of
  // consecutive delivered messages so the two are comparable (a message that failed to publish
  // between two widens both gaps alike). The replayed rate is in recorded time (divided by the
  // playback speed), so a faithful run reads the same number twice at any speed. The pair across
  // a loop wrap is excluded from both.
  //
  // Empty for fewer than two delivered messages, or a recording with no cadence.
  [[nodiscard]] std::optional<double> recorded_hz(std::string_view topic) const;
  [[nodiscard]] std::optional<double> replayed_hz(std::string_view topic) const;

 private:
  // One warning window per ReplayProblem kind.
  static constexpr std::size_t kProblemKinds = static_cast<std::size_t>(ReplayProblem::kCount);

  // A run of consecutive late messages within one pass, by their number among the messages
  // delivered on the topic in the pass, starting at 1. A message that was skipped as unreadable or
  // refused at publish is not delivered, so it takes no number and later numbers fall short of
  // the recording's own order by the count skipped.
  struct LateRun {
    std::size_t first = 0;
    std::size_t last = 0;
  };

  // How many runs of late messages are kept per pass: enough to read, and bounded for a run that
  // is late throughout.
  static constexpr std::size_t kMaxLateRuns = 10;

  // The late messages of a pass that has ended.
  struct LatePass {
    std::vector<LateRun> runs;
    std::size_t count = 0;
  };

  // Where one kind of problem on one topic stands in its warning window.
  struct WarnWindow {
    bool warned = false;
    std::chrono::steady_clock::time_point last_warned_at{};
    // Occurrences since the last warning that were counted but not logged.
    std::size_t suppressed = 0;
  };

  struct PerTopic {
    std::size_t emitted = 0;
    std::size_t refused = 0;
    std::size_t decode_failures = 0;
    std::size_t late_arrivals = 0;
    // The late messages of the current pass, by delivered number in it: the runs kept (see
    // kMaxLateRuns)
    // and how many there were in all.
    std::vector<LateRun> late_runs;
    std::size_t pass_late = 0;
    bool never_subscribed = false;
    std::chrono::nanoseconds worst_lag{0};
    std::chrono::nanoseconds total_lag{0};
    // The last delivered message's recorded time.
    std::uint64_t last_recorded_time_ns = 0;
    bool has_recorded_time = false;
    // The current pass's first delivered message, the reference for the rest; see
    // worst_schedule_error().
    std::uint64_t pass_first_recorded_time_ns = 0;
    std::chrono::steady_clock::time_point pass_first_emitted_at{};
    std::chrono::steady_clock::time_point last_emitted_at{};
    // Running statistics (Welford) of the current pass's signed schedule errors in nanoseconds,
    // including the first message's zero: count, mean, sum of squared deviations, min and max.
    // Doubles so a long run cannot overflow.
    std::size_t pass_n = 0;
    double pass_mean = 0.0;
    double pass_m2 = 0.0;
    double pass_min = 0.0;
    double pass_max = 0.0;
    // Finished passes of two or more messages, folded together: error count, squared deviations
    // each about its own pass's mean, and the widest single-pass spread.
    std::size_t pooled_n = 0;
    double pooled_m2 = 0.0;
    double worst_range = 0.0;
    // Consecutive delivered pairs within a pass, with their recorded and replayed gaps summed;
    // see recorded_hz().
    std::size_t paired = 0;
    std::uint64_t recorded_gap_sum_ns = 0;
    std::chrono::nanoseconds emit_gap_sum{0};
    std::array<WarnWindow, kProblemKinds> warn_windows{};
  };

  // What worst_schedule_error() and typical_schedule_error() report, in nanoseconds. Folds the
  // in-progress pass into a copy so reading changes nothing. Empty where those accessors are.
  struct ScheduleSpread {
    double worst_ns = 0.0;
    double typical_ns = 0.0;
  };
  [[nodiscard]] static std::optional<ScheduleSpread> schedule_spread_of(const PerTopic& entry);

  // What the closing lines of the summary depend on, summed over every topic that reported.
  struct SummaryTotals {
    std::size_t emitted = 0;
    std::size_t refused = 0;
    std::size_t decode_failures = 0;
    std::size_t late_arrivals = 0;
    bool any_never_subscribed = false;
  };

  // Decides whether this report of problem on entry is logged. Returns empty when it falls in the
  // window of a warning logged less than kWarnInterval ago, and counts its `occurrences` as
  // suppressed. Otherwise returns how many occurrences were suppressed since the last warning, and
  // opens a new window at now. Callers hold mutex_ and log after releasing it.
  [[nodiscard]] static std::optional<std::size_t> take_warning_locked(
      PerTopic& entry, ReplayProblem problem, std::chrono::steady_clock::time_point now,
      std::size_t occurrences = 1);

  // The counting body of record_emitted(). Returns the late messages of the pass this message
  // ended, if it started a new pass and the old one had any. Callers hold mutex_.
  [[nodiscard]] std::optional<LatePass> record_emitted_locked(
      PerTopic& entry, std::chrono::nanoseconds schedule_lag, std::uint64_t recorded_time_ns,
      std::chrono::steady_clock::time_point emitted_at);

  // The positions of `count` late messages kept as `runs`, as "messages #1, #4-#6, and 3 more".
  [[nodiscard]] static std::string describe_late_messages(const std::vector<LateRun>& runs,
                                                          std::size_t count);

  // The pieces of summary_lines(). Callers hold mutex_.
  void append_topic_lines(const std::string& topic, const PerTopic& entry,
                          std::vector<std::string>& lines) const;
  // The line on rate and spacing for a topic that has a recorded cadence, worded for the playback
  // speed.
  [[nodiscard]] std::optional<std::string> spacing_line(const PerTopic& entry) const;
  void append_closing_lines(const SummaryTotals& totals, std::vector<std::string>& lines) const;

  // The entry for topic by name, or null if none was registered. Callers hold mutex_.
  [[nodiscard]] const PerTopic* find(std::string_view topic) const;

  // The unlocked bodies of recorded_hz() and replayed_hz(), for callers already holding the lock.
  [[nodiscard]] static std::optional<double> recorded_hz_of(const PerTopic& entry);
  [[nodiscard]] std::optional<double> replayed_hz_of(const PerTopic& entry) const;

  mutable std::mutex mutex_;
  // Indexed by TopicId, with each topic's name beside it in names_ and the reverse in ids_ (used
  // when reading by name, never per message).
  std::vector<PerTopic> topics_;
  std::vector<std::string> names_;
  std::unordered_map<std::string, TopicId> ids_;
  // Recorded seconds per wall-clock second.
  const double playback_speed_;
  bool start_stalled_ = false;
  bool read_failed_ = false;
};

}  // namespace isaac::os::mcap_replayer
