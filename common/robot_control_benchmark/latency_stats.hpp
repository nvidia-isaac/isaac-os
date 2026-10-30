/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace isaac::os::benchmark {

/// Track first responses across warmup and measurement without restarting at the boundary.
class TargetResponseTracker {
 public:
  /// Return true only for the first completed command using a real target version.
  bool record(std::int64_t produced_ns, std::uint64_t version) {
    if (produced_ns == 0) {
      return false;
    }
    const bool first = !last_version_.has_value() || version != *last_version_;
    last_version_ = version;
    return first;
  }

 private:
  std::optional<std::uint64_t> last_version_;
};

/// One raw observation retained without allocation or file I/O on the measured path.
struct ControlTickObservation {
  std::int64_t sequence = 0;
  std::int64_t scheduled_ns = 0;
  // HSDK exposes callback entry, not a distinct executor wake coordinate.
  std::int64_t callback_start_ns = 0;
  std::int64_t completion_ns = 0;
  std::int64_t target_produced_ns = 0;
  std::int64_t target_use_ns = 0;
  std::uint64_t target_version = 0;
  bool first_command_for_target = false;
  std::uint64_t coalesced_ticks = 0;
  std::uint64_t dropped_ticks = 0;
  bool late_command = false;
};

/// Tail statistics for one KPI distribution.
struct DistributionSummary {
  std::size_t count = 0;
  double p50_us = 0.0;
  double p99_99_us = 0.0;
  double max_us = 0.0;
};

/// Complete compact result emitted by one benchmark run.
struct KpiSummary {
  DistributionSummary joint_read_callback_entry_to_command_completion;
  DistributionSummary joint_read_callback_interval;
  DistributionSummary scheduled_activation_to_joint_read_callback_entry;
  DistributionSummary scheduled_activation_to_command_completion;
  DistributionSummary target_produced_to_first_command_completion;
  std::uint64_t late_commands = 0;
  std::uint64_t coalesced_ticks = 0;
  std::uint64_t dropped_ticks = 0;
};

/// Collects benchmark observations without allocating on the measured path.
class LatencyStats {
 public:
  /// Initializes every slot before measurement, touching the pages that recording will write.
  explicit LatencyStats(std::size_t sample_capacity) : observations_(sample_capacity) {}

  /// Records one completed command and its preceding-interval correctness counters.
  [[nodiscard]] bool add_control_tick(ControlTickObservation observation) {
    if (count_ == observations_.size() ||
        observation.callback_start_ns < observation.scheduled_ns ||
        observation.completion_ns < observation.callback_start_ns ||
        (count_ != 0U &&
         observation.callback_start_ns <= observations_[count_ - 1U].callback_start_ns) ||
        (observation.target_produced_ns != 0 &&
         (observation.target_use_ns < observation.target_produced_ns ||
          observation.completion_ns < observation.target_use_ns)) ||
        (observation.first_command_for_target && observation.target_produced_ns == 0)) {
      return false;
    }
    if (count_ == 0U) {
      observation.coalesced_ticks = 0U;
      observation.dropped_ticks = 0U;
    }
    observations_[count_++] = observation;
    return true;
  }

  /// Returns the number of measured commands recorded so far.
  [[nodiscard]] std::size_t count() const { return count_; }

  /// Computes P50, P99.99, and maximum for each distribution.
  [[nodiscard]] KpiSummary summary() const {
    std::vector<double> joint_read_callback_entry_to_command_completion_us;
    std::vector<double> joint_read_callback_interval_us;
    std::vector<double> scheduled_activation_to_joint_read_callback_entry_us;
    std::vector<double> scheduled_activation_to_command_completion_us;
    std::vector<double> target_produced_to_first_command_completion_us;
    joint_read_callback_entry_to_command_completion_us.reserve(count_);
    joint_read_callback_interval_us.reserve(count_ == 0U ? 0U : count_ - 1U);
    scheduled_activation_to_joint_read_callback_entry_us.reserve(count_);
    scheduled_activation_to_command_completion_us.reserve(count_);
    target_produced_to_first_command_completion_us.reserve(count_);

    KpiSummary result;
    for (std::size_t index = 0; index < count_; ++index) {
      const ControlTickObservation& observation = observations_[index];
      joint_read_callback_entry_to_command_completion_us.push_back(
          static_cast<double>(observation.completion_ns - observation.callback_start_ns) * 1e-3);
      scheduled_activation_to_joint_read_callback_entry_us.push_back(
          static_cast<double>(observation.callback_start_ns - observation.scheduled_ns) * 1e-3);
      scheduled_activation_to_command_completion_us.push_back(
          static_cast<double>(observation.completion_ns - observation.scheduled_ns) * 1e-3);
      if (observation.first_command_for_target) {
        target_produced_to_first_command_completion_us.push_back(
            static_cast<double>(observation.completion_ns - observation.target_produced_ns) * 1e-3);
      }
      if (index > 0U) {
        joint_read_callback_interval_us.push_back(
            static_cast<double>(observation.callback_start_ns -
                                observations_[index - 1U].callback_start_ns) *
            1e-3);
      }
      result.late_commands += observation.late_command ? 1U : 0U;
      result.coalesced_ticks += observation.coalesced_ticks;
      result.dropped_ticks += observation.dropped_ticks;
    }
    result.joint_read_callback_entry_to_command_completion =
        summarize(std::move(joint_read_callback_entry_to_command_completion_us));
    result.joint_read_callback_interval = summarize(std::move(joint_read_callback_interval_us));
    result.scheduled_activation_to_joint_read_callback_entry =
        summarize(std::move(scheduled_activation_to_joint_read_callback_entry_us));
    result.scheduled_activation_to_command_completion =
        summarize(std::move(scheduled_activation_to_command_completion_us));
    result.target_produced_to_first_command_completion =
        summarize(std::move(target_produced_to_first_command_completion_us));
    return result;
  }

  /// Prints metric distributions, correctness counters, and a v6 result row.
  void report(const std::string& label, const std::string& scenario, double control_rate_hz,
              double target_rate_hz) const {
    const KpiSummary result = summary();
    const auto print_distribution = [](const char* name, const DistributionSummary& metric) {
      if (metric.count == 0U) {
        std::cout << std::format("  {}: samples 0, P50 N/A, P99.99 N/A, max N/A\n", name);
      } else {
        std::cout << std::format(
            "  {}: samples {}, P50 {:.2f} us, P99.99 {:.2f} us, max {:.2f} us\n", name,
            metric.count, metric.p50_us, metric.p99_99_us, metric.max_us);
      }
    };
    std::cout << std::format("\n==== {}: IK control benchmark ====\n", label);
    print_distribution("KPI scheduled activation -> joint-command completion",
                       result.scheduled_activation_to_command_completion);
    print_distribution("KPI scheduled activation -> joint-read callback entry",
                       result.scheduled_activation_to_joint_read_callback_entry);
    print_distribution("KPI target produced -> first completed joint-command using that target",
                       result.target_produced_to_first_command_completion);
    std::cout << std::format(
        "  correctness: late commands={}, coalesced ticks={}, dropped ticks={}\n"
        "  Additional metrics:\n",
        result.late_commands, result.coalesced_ticks, result.dropped_ticks);
    print_distribution("joint-read callback entry -> joint-command completion",
                       result.joint_read_callback_entry_to_command_completion);
    print_distribution("interval between successive joint-read callback entries",
                       result.joint_read_callback_interval);
    std::cout << std::format("BENCHCSV,{},{},{},{}", label, scenario, control_rate_hz,
                             target_rate_hz);
    for (const DistributionSummary* metric :
         {&result.scheduled_activation_to_command_completion,
          &result.scheduled_activation_to_joint_read_callback_entry,
          &result.target_produced_to_first_command_completion,
          &result.joint_read_callback_entry_to_command_completion,
          &result.joint_read_callback_interval}) {
      if (metric->count == 0U) {
        std::cout << ",0,,,";
      } else {
        std::cout << std::format(",{},{:.2f},{:.2f},{:.2f}", metric->count, metric->p50_us,
                                 metric->p99_99_us, metric->max_us);
      }
    }
    std::cout << std::format(",{},{},{},bench-latency/v6\n", result.late_commands,
                             result.coalesced_ticks, result.dropped_ticks);
    std::cout.flush();
  }

  /// Writes integer clock coordinates so every reported distribution is reproducible.
  void write_observations_csv(const std::filesystem::path& path) const {
    std::ofstream output(path);
    if (!output) {
      throw std::ios_base::failure(
          std::format("Unable to open raw benchmark observations at '{}'.", path.string()));
    }
    output << "schema,index,sequence,scheduled_ns,callback_start_ns,completion_ns,"
              "callback_to_completion_ns,scheduled_to_callback_ns,scheduled_to_completion_ns,"
              "callback_period_ns,late_command,coalesced_ticks,dropped_ticks,"
              "target_produced_ns,target_use_ns,target_version,first_command_for_target,"
              "target_produced_to_completion_ns\n";
    for (std::size_t index = 0; index < count_; ++index) {
      const ControlTickObservation& observation = observations_[index];
      output << "bench-observations/v3," << index << ',' << observation.sequence << ','
             << observation.scheduled_ns << ',' << observation.callback_start_ns << ','
             << observation.completion_ns << ','
             << observation.completion_ns - observation.callback_start_ns << ','
             << observation.callback_start_ns - observation.scheduled_ns << ','
             << observation.completion_ns - observation.scheduled_ns << ',';
      if (index > 0U) {
        output << observation.callback_start_ns - observations_[index - 1U].callback_start_ns;
      }
      output << ',' << (observation.late_command ? 1 : 0) << ',' << observation.coalesced_ticks
             << ',' << observation.dropped_ticks << ',';
      if (observation.target_produced_ns != 0) {
        output << observation.target_produced_ns << ',' << observation.target_use_ns << ','
               << observation.target_version << ','
               << (observation.first_command_for_target ? 1 : 0) << ','
               << observation.completion_ns - observation.target_produced_ns;
      } else {
        output << ",,,,";
      }
      output << '\n';
    }
    if (!output) {
      throw std::ios_base::failure(
          std::format("Unable to write raw benchmark observations at '{}'.", path.string()));
    }
  }

 private:
  /// Uses linear interpolation between adjacent ordered samples.
  [[nodiscard]] static double percentile(const std::vector<double>& sorted, double percent) {
    if (sorted.empty()) {
      return 0.0;
    }
    const double rank = (percent / 100.0) * static_cast<double>(sorted.size() - 1U);
    const auto lower = static_cast<std::size_t>(std::floor(rank));
    const std::size_t upper = std::min(lower + 1U, sorted.size() - 1U);
    const double fraction = rank - static_cast<double>(lower);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
  }

  /// Sorts after measurement so the observed path stays constant-time.
  [[nodiscard]] static DistributionSummary summarize(std::vector<double> samples) {
    if (samples.empty()) {
      return {};
    }
    std::ranges::sort(samples);
    return DistributionSummary{
        .count = samples.size(),
        .p50_us = percentile(samples, 50.0),
        .p99_99_us = percentile(samples, 99.99),
        .max_us = samples.back(),
    };
  }

  std::vector<ControlTickObservation> observations_;
  std::size_t count_ = 0;
};

}  // namespace isaac::os::benchmark
