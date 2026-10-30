/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/mcap_replayer/replay_stats.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

#include "holoscan/logger/logger.hpp"
#include <fmt/format.h>

namespace isaac::os::mcap_replayer {

namespace {

double to_milliseconds(std::chrono::nanoseconds value) {
  return std::chrono::duration<double, std::milli>(value).count();
}

// The tail of a warning that stood in for others held back by the rate limit, empty when none were.
std::string suppressed_note(std::size_t suppressed) {
  if (suppressed == 0) {
    return "";
  }
  return fmt::format(" {} more like it on this topic were not logged since the last warning.",
                     suppressed);
}

}  // namespace

ReplayStats::TopicId ReplayStats::register_topic(std::string_view topic) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto [entry, added] =
      ids_.try_emplace(std::string(topic), static_cast<TopicId>(topics_.size()));
  if (added) {
    topics_.emplace_back();
    names_.emplace_back(topic);
  }
  return entry->second;
}

const ReplayStats::PerTopic* ReplayStats::find(std::string_view topic) const {
  const auto entry = ids_.find(std::string(topic));
  return entry == ids_.end() ? nullptr : &topics_[entry->second];
}

void ReplayStats::record_emitted(TopicId topic, std::chrono::nanoseconds schedule_lag,
                                 std::uint64_t recorded_time_ns,
                                 std::chrono::steady_clock::time_point emitted_at) {
  // Logged after the lock is released: a terminal write must not hold up other readers of the
  // stats. The topic name and positions are only copied out when there is something to log.
  struct PassReport {
    std::string topic;
    std::string late_messages;
    std::size_t count;
    std::size_t suppressed;
  };
  std::optional<PassReport> report;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    PerTopic& entry = topics_.at(topic);
    if (const auto ended =
            record_emitted_locked(entry, schedule_lag, recorded_time_ns, emitted_at)) {
      if (const auto suppressed =
              take_warning_locked(entry, ReplayProblem::kBehindSchedule, emitted_at, ended->count);
          suppressed.has_value()) {
        report = PassReport{names_.at(topic), describe_late_messages(ended->runs, ended->count),
                            ended->count, *suppressed};
      }
    }
  }
  if (report) {
    HOLOSCAN_LOG_WARN(
        "Topic '{}': {} went out more than {} ms after they were due in the last pass over the "
        "recording ({}). Their contents are unaffected. If these are only the first few messages "
        "of the run, this may be start-up noise; otherwise a lower playback speed, a faster "
        "consumer or a less busy machine would keep timing closer.{}",
        report->topic, report->count == 1 ? "1 message" : fmt::format("{} messages", report->count),
        kLateThreshold.count(), report->late_messages, suppressed_note(report->suppressed));
  }
}

std::optional<ReplayStats::LatePass> ReplayStats::record_emitted_locked(
    PerTopic& entry, std::chrono::nanoseconds schedule_lag, std::uint64_t recorded_time_ns,
    std::chrono::steady_clock::time_point emitted_at) {
  ++entry.emitted;
  if (schedule_lag > kLateThreshold) {
    ++entry.late_arrivals;
  }
  entry.worst_lag = std::max(entry.worst_lag, schedule_lag);
  entry.total_lag += schedule_lag;
  std::optional<LatePass> ended;

  // Measured from the recording and emit instants alone, never the due time that lag uses, so a
  // wrong due time cannot hide itself here. Recorded time stepping backwards is a loop wrap: that
  // message opens a new pass and becomes its reference, and the step is neither a recorded nor a
  // replayed interval. Two messages at one recorded instant are an ordinary pair with a zero
  // recorded gap.
  const bool starts_pass =
      !entry.has_recorded_time || recorded_time_ns < entry.last_recorded_time_ns;
  if (starts_pass) {
    // A pass of one message has no spread, so it adds nothing to fold.
    if (entry.pass_n >= 2) {
      entry.pooled_m2 += entry.pass_m2;
      entry.pooled_n += entry.pass_n;
      entry.worst_range = std::max(entry.worst_range, entry.pass_max - entry.pass_min);
    }
    // The late messages of the pass that is ending are handed back to be reported; this message
    // is the first of the next pass.
    if (entry.pass_late > 0) {
      ended = LatePass{std::move(entry.late_runs), entry.pass_late};
    }
    entry.late_runs.clear();
    entry.pass_late = 0;
    entry.pass_first_recorded_time_ns = recorded_time_ns;
    entry.pass_first_emitted_at = emitted_at;
    // The reference message's zero error is a sample like any other; omitting it would hide a pass
    // whose first message went out late.
    entry.pass_n = 1;
    entry.pass_mean = 0.0;
    entry.pass_m2 = 0.0;
    entry.pass_min = 0.0;
    entry.pass_max = 0.0;
  } else {
    ++entry.paired;
    entry.recorded_gap_sum_ns += recorded_time_ns - entry.last_recorded_time_ns;
    entry.emit_gap_sum +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(emitted_at - entry.last_emitted_at);

    const double replayed_since_first_ns =
        std::chrono::duration<double, std::nano>(emitted_at - entry.pass_first_emitted_at).count();
    const double recorded_since_first_ns =
        static_cast<double>(recorded_time_ns - entry.pass_first_recorded_time_ns) / playback_speed_;
    // Signed, so early and late messages land on opposite sides of the spread.
    const double error_ns = replayed_since_first_ns - recorded_since_first_ns;
    ++entry.pass_n;
    const double delta = error_ns - entry.pass_mean;
    entry.pass_mean += delta / static_cast<double>(entry.pass_n);
    entry.pass_m2 += delta * (error_ns - entry.pass_mean);
    entry.pass_min = std::min(entry.pass_min, error_ns);
    entry.pass_max = std::max(entry.pass_max, error_ns);
  }
  entry.last_emitted_at = emitted_at;

  // pass_n now counts this message, so it is its number among the messages delivered in the pass.
  if (schedule_lag > kLateThreshold) {
    ++entry.pass_late;
    const std::size_t position = entry.pass_n;
    if (!entry.late_runs.empty() && entry.late_runs.back().last + 1 == position) {
      entry.late_runs.back().last = position;
    } else if (entry.late_runs.size() < kMaxLateRuns) {
      entry.late_runs.push_back(LateRun{position, position});
    }
  }

  entry.last_recorded_time_ns = recorded_time_ns;
  entry.has_recorded_time = true;
  return ended;
}

void ReplayStats::record_refused_emit(TopicId topic, std::chrono::steady_clock::time_point now) {
  std::optional<std::string> name;
  std::size_t suppressed = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    PerTopic& entry = topics_.at(topic);
    ++entry.refused;
    if (const auto held_back = take_warning_locked(entry, ReplayProblem::kRefusedEmit, now);
        held_back.has_value()) {
      name = names_.at(topic);
      suppressed = *held_back;
    }
  }
  if (name) {
    HOLOSCAN_LOG_WARN(
        "Topic '{}' could not publish a replayed message. The usual cause is that nothing is "
        "subscribed to it -- either it was never wired to a consumer, or its consumer's graph "
        "stopped while the run was still going. The end-of-run summary reports how many messages "
        "this cost.{}",
        *name, suppressed_note(suppressed));
  }
}

void ReplayStats::record_decode_failure(TopicId topic, std::string_view reason,
                                        std::chrono::steady_clock::time_point now) {
  std::optional<std::string> name;
  std::size_t suppressed = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    PerTopic& entry = topics_.at(topic);
    ++entry.decode_failures;
    if (const auto held_back = take_warning_locked(entry, ReplayProblem::kUnreadableMessage, now);
        held_back.has_value()) {
      name = names_.at(topic);
      suppressed = *held_back;
    }
  }
  if (name) {
    HOLOSCAN_LOG_WARN(
        "A recorded message on topic '{}' could not be read back and was left out of the "
        "replay: {}.{}",
        *name, reason, suppressed_note(suppressed));
  }
}

void ReplayStats::record_never_subscribed(TopicId topic) {
  const std::lock_guard<std::mutex> lock(mutex_);
  topics_.at(topic).never_subscribed = true;
}

std::optional<std::size_t> ReplayStats::take_warning_locked(
    PerTopic& entry, ReplayProblem problem, std::chrono::steady_clock::time_point now,
    std::size_t occurrences) {
  WarnWindow& window = entry.warn_windows.at(static_cast<std::size_t>(problem));
  if (window.warned && now - window.last_warned_at < kWarnInterval) {
    window.suppressed += occurrences;
    return std::nullopt;
  }
  const std::size_t suppressed = window.suppressed;
  window.warned = true;
  window.last_warned_at = now;
  window.suppressed = 0;
  return suppressed;
}

std::string ReplayStats::describe_late_messages(const std::vector<LateRun>& runs,
                                                std::size_t count) {
  std::string list;
  std::size_t listed = 0;
  for (const LateRun& run : runs) {
    if (!list.empty()) {
      list += ", ";
    }
    list += run.first == run.last ? fmt::format("#{}", run.first)
                                  : fmt::format("#{}-#{}", run.first, run.last);
    listed += run.last - run.first + 1;
  }
  if (listed < count) {
    list += fmt::format(", and {} more", count - listed);
  }
  const bool several = listed < count || runs.size() > 1 || runs.front().first != runs.front().last;
  return (several ? "messages " : "message ") + list;
}

std::size_t ReplayStats::emitted(std::string_view topic) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const PerTopic* const it = find(topic);
  return it == nullptr ? 0U : it->emitted;
}

std::size_t ReplayStats::refused(std::string_view topic) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const PerTopic* const it = find(topic);
  return it == nullptr ? 0U : it->refused;
}

std::size_t ReplayStats::decode_failures(std::string_view topic) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const PerTopic* const it = find(topic);
  return it == nullptr ? 0U : it->decode_failures;
}

void ReplayStats::record_start_stalled() {
  const std::lock_guard<std::mutex> lock(mutex_);
  start_stalled_ = true;
}

bool ReplayStats::start_stalled() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return start_stalled_;
}

void ReplayStats::record_read_failed() {
  const std::lock_guard<std::mutex> lock(mutex_);
  read_failed_ = true;
}

bool ReplayStats::read_failed() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return read_failed_;
}

bool ReplayStats::never_subscribed(std::string_view topic) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const PerTopic* const it = find(topic);
  return it != nullptr && it->never_subscribed;
}

std::size_t ReplayStats::late_arrivals(std::string_view topic) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const PerTopic* const it = find(topic);
  return it == nullptr ? 0U : it->late_arrivals;
}

std::chrono::nanoseconds ReplayStats::worst_lag(std::string_view topic) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const PerTopic* const it = find(topic);
  return it == nullptr ? std::chrono::nanoseconds{0} : it->worst_lag;
}

std::chrono::nanoseconds ReplayStats::mean_lag(std::string_view topic) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const PerTopic* const it = find(topic);
  if (it == nullptr || it->emitted == 0) {
    return std::chrono::nanoseconds{0};
  }
  return it->total_lag / static_cast<std::chrono::nanoseconds::rep>(it->emitted);
}

std::optional<ReplayStats::ScheduleSpread> ReplayStats::schedule_spread_of(const PerTopic& entry) {
  // With no recorded gap there is no spacing to keep, and passes that never step recorded time
  // backwards cannot be told apart, so repeats would read as one pass's worth of error.
  if (entry.recorded_gap_sum_ns == 0) {
    return std::nullopt;
  }
  std::size_t pooled_n = entry.pooled_n;
  double pooled_m2 = entry.pooled_m2;
  double worst_range = entry.worst_range;
  if (entry.pass_n >= 2) {
    pooled_n += entry.pass_n;
    pooled_m2 += entry.pass_m2;
    worst_range = std::max(worst_range, entry.pass_max - entry.pass_min);
  }
  if (pooled_n == 0) {
    return std::nullopt;
  }
  return ScheduleSpread{.worst_ns = worst_range,
                        .typical_ns = std::sqrt(pooled_m2 / static_cast<double>(pooled_n))};
}

std::optional<std::chrono::nanoseconds> ReplayStats::worst_schedule_error(
    std::string_view topic) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const PerTopic* const it = find(topic);
  const std::optional<ScheduleSpread> spread =
      it == nullptr ? std::nullopt : schedule_spread_of(*it);
  if (!spread.has_value()) {
    return std::nullopt;
  }
  return std::chrono::nanoseconds(std::llround(spread->worst_ns));
}

std::optional<std::chrono::nanoseconds> ReplayStats::typical_schedule_error(
    std::string_view topic) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const PerTopic* const it = find(topic);
  const std::optional<ScheduleSpread> spread =
      it == nullptr ? std::nullopt : schedule_spread_of(*it);
  if (!spread.has_value()) {
    return std::nullopt;
  }
  return std::chrono::nanoseconds(std::llround(spread->typical_ns));
}

std::optional<double> ReplayStats::recorded_hz_of(const PerTopic& entry) {
  if (entry.recorded_gap_sum_ns == 0) {
    return std::nullopt;
  }
  return static_cast<double>(entry.paired) * 1e9 / static_cast<double>(entry.recorded_gap_sum_ns);
}

std::optional<double> ReplayStats::replayed_hz_of(const PerTopic& entry) const {
  // Both rates or neither, so one is never shown without the other to compare against.
  if (entry.recorded_gap_sum_ns == 0 || entry.emit_gap_sum <= std::chrono::nanoseconds::zero()) {
    return std::nullopt;
  }
  return static_cast<double>(entry.paired) * 1e9 /
         (static_cast<double>(entry.emit_gap_sum.count()) * playback_speed_);
}

std::optional<double> ReplayStats::recorded_hz(std::string_view topic) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const PerTopic* const it = find(topic);
  if (it == nullptr || !replayed_hz_of(*it).has_value()) {
    return std::nullopt;
  }
  return recorded_hz_of(*it);
}

std::optional<double> ReplayStats::replayed_hz(std::string_view topic) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const PerTopic* const it = find(topic);
  return it == nullptr ? std::nullopt : replayed_hz_of(*it);
}

std::vector<std::string> ReplayStats::summary_lines() const {
  const std::lock_guard<std::mutex> lock(mutex_);

  std::vector<std::string> lines;
  SummaryTotals totals;
  for (std::size_t index = 0; index < topics_.size(); ++index) {
    const PerTopic& entry = topics_[index];
    // A topic that was registered but never had anything happen reports nothing.
    if (entry.emitted == 0 && entry.refused == 0 && entry.decode_failures == 0 &&
        !entry.never_subscribed) {
      continue;
    }
    totals.emitted += entry.emitted;
    totals.refused += entry.refused;
    totals.decode_failures += entry.decode_failures;
    totals.late_arrivals += entry.late_arrivals;
    totals.any_never_subscribed = totals.any_never_subscribed || entry.never_subscribed;
    append_topic_lines(names_[index], entry, lines);
  }
  append_closing_lines(totals, lines);
  return lines;
}

void ReplayStats::append_topic_lines(const std::string& topic, const PerTopic& entry,
                                     std::vector<std::string>& lines) const {
  // Counts of what this run did, not a reconciliation against the recording, whose total message
  // count a streaming replay never learns.
  lines.push_back(
      fmt::format("Topic '{}' delivered {} messages to the consumer.", topic, entry.emitted));
  if (entry.never_subscribed) {
    lines.emplace_back("  Its consumer never subscribed to it.");
  }
  if (entry.refused > 0) {
    lines.push_back(
        fmt::format("  {} were skipped because the topic could not publish them.", entry.refused));
  }
  if (entry.decode_failures > 0) {
    lines.push_back(
        fmt::format("  {} were skipped because they could not be read back from the recording "
                    "as the type this topic is being replayed as.",
                    entry.decode_failures));
  }
  if (entry.emitted > 0) {
    const std::chrono::nanoseconds average =
        entry.total_lag / static_cast<std::chrono::nanoseconds::rep>(entry.emitted);
    lines.push_back(
        fmt::format("  Messages arrived {:.1f} ms late on average and {:.1f} ms "
                    "late at worst.",
                    to_milliseconds(average), to_milliseconds(entry.worst_lag)));
  }
  if (std::optional<std::string> spacing = spacing_line(entry)) {
    lines.push_back(std::move(*spacing));
  }
  if (entry.late_arrivals > 0) {
    std::string line = fmt::format("  {} went out more than {} ms after they were due",
                                   entry.late_arrivals, kLateThreshold.count());
    // The late messages of the pass the run ended in, which is the whole run unless it looped.
    // Earlier passes' were logged as each ended.
    if (entry.pass_late > 0) {
      line += fmt::format(" ({}{})", describe_late_messages(entry.late_runs, entry.pass_late),
                          entry.pass_late == entry.late_arrivals ? "" : " in the final pass");
    }
    lines.push_back(line + ".");
  }
}

std::optional<std::string> ReplayStats::spacing_line(const PerTopic& entry) const {
  // A topic with no recorded cadence has neither a spread nor a rate, so gets no line.
  const std::optional<ScheduleSpread> spread = schedule_spread_of(entry);
  if (!spread.has_value()) {
    return std::nullopt;
  }
  const std::string kept_spacing = fmt::format(
      "kept the recording's own spacing to within {:.1f} ms typically and {:.1f} ms at worst.",
      spread->typical_ns / 1e6, spread->worst_ns / 1e6);
  const std::optional<double> recorded = recorded_hz_of(entry);
  const std::optional<double> replayed = replayed_hz_of(entry);
  if (!recorded.has_value() || !replayed.has_value()) {
    return fmt::format("  Messages {}", kept_spacing);
  }
  if (playback_speed_ == 1.0) {
    return fmt::format("  Replayed at {:.2f} Hz, recorded at {:.2f} Hz; messages {}", *replayed,
                       *recorded, kept_spacing);
  }
  // The replayed rate is given in recorded time to read against the recorded rate, with the
  // delivered rate beside it. The spacing is in delivered time, and the sentence says so.
  return fmt::format(
      "  Replayed at {:.2f} Hz in recorded time ({:.2f} Hz as delivered at {:g}x speed), "
      "recorded at {:.2f} Hz; as delivered, messages {}",
      *replayed, *replayed * playback_speed_, playback_speed_, *recorded, kept_spacing);
}

void ReplayStats::append_closing_lines(const SummaryTotals& totals,
                                       std::vector<std::string>& lines) const {
  if (totals.refused > 0) {
    lines.emplace_back(
        "Some topics could not publish every message this run read. The usual cause is a topic "
        "with nothing subscribed to it -- one never wired to a consumer, or one whose consumer's "
        "graph stopped while the run was still going.");
  }
  if (totals.late_arrivals > 0) {
    lines.emplace_back(
        "Replay fell behind the recorded timing during this run, so some messages did not reach "
        "the consumer as they did when they were recorded. Their contents are unaffected. Check "
        "the late messages listed for each topic above: if there are few and they are the first "
        "messages on that topic, this is likely start-up noise. If there are many, or they are "
        "spread through the run, a lower playback speed, a faster consumer or a less busy "
        "machine would keep it closer.");
  }
  if (totals.decode_failures > 0) {
    lines.emplace_back(
        "Some recorded messages could not be read back and never reached the consumer. The "
        "recording may be damaged, or may not hold what the topics above are being replayed as.");
  }
  if (read_failed_) {
    lines.emplace_back(
        "Reading the recording failed partway through, so only the part read before the failure "
        "was replayed. Check that the file is intact and the storage holding it is responding.");
  }

  // At most one of these: the run never began, ended early, replayed nothing, or went well.
  if (start_stalled_) {
    // Like a consumer that never subscribed, this replaces the closing lines below: the run never
    // began, which is neither an empty recording nor a healthy run.
    lines.emplace_back(
        "The run stopped before replaying anything because reading the recording stalled while it "
        "was starting. Check that the file is readable and the storage holding it is responding.");
  } else if (totals.any_never_subscribed) {
    // Replaces both closing lines below: the run never began, which is neither an empty recording
    // nor a healthy run.
    lines.emplace_back(
        "The run stopped before replaying anything because a topic wired to a consumer never had "
        "that consumer subscribe. Check that the graph holding the consumer was started and is "
        "still running.");
  } else if (read_failed_) {
    // The line above already says why this was cut short, which is neither an empty recording nor
    // a healthy run.
  } else if (totals.emitted == 0 && totals.refused == 0 && totals.decode_failures == 0) {
    // Without this, the healthy line below would be vacuously true and read as success for a run
    // that replayed nothing.
    lines.emplace_back(
        "No messages were replayed. The recording held none on the topics that were bound.");
  } else if (totals.refused == 0 && totals.decode_failures == 0 && totals.late_arrivals == 0) {
    // A claim only about the messages this run read, which is all a streaming replay sees.
    lines.emplace_back(
        "Every message read from the recording was delivered to the consumer on schedule.");
  }
}

void ReplayStats::log_summary() const {
  for (const std::string& line : summary_lines()) {
    HOLOSCAN_LOG_INFO("{}", line);
  }
}

}  // namespace isaac::os::mcap_replayer
