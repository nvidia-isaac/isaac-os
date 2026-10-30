/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/robot_control_benchmark/latency_stats.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace isaac::os::benchmark {
namespace {

TEST(LatencyStats, PreallocatedSlotsAreNotCountedUntilWritten) {
  LatencyStats stats{4U};
  EXPECT_EQ(stats.count(), 0U);
  EXPECT_EQ(stats.summary().joint_read_callback_entry_to_command_completion.count, 0U);
  ASSERT_TRUE(stats.add_control_tick(
      ControlTickObservation{.scheduled_ns = 10, .callback_start_ns = 20, .completion_ns = 30}));
  EXPECT_EQ(stats.count(), 1U);
  EXPECT_EQ(stats.summary().joint_read_callback_entry_to_command_completion.count, 1U);
  EXPECT_EQ(stats.summary().joint_read_callback_interval.count, 0U);
}

TEST(LatencyStats, FullAndZeroCapacityBuffersRejectAdditionalObservations) {
  const ControlTickObservation first{
      .scheduled_ns = 10, .callback_start_ns = 20, .completion_ns = 30};
  LatencyStats empty{0U};
  EXPECT_FALSE(empty.add_control_tick(first));
  EXPECT_EQ(empty.count(), 0U);

  LatencyStats stats{1U};
  ASSERT_TRUE(stats.add_control_tick(first));
  EXPECT_FALSE(stats.add_control_tick(
      ControlTickObservation{.scheduled_ns = 40, .callback_start_ns = 50, .completion_ns = 60}));
  EXPECT_EQ(stats.count(), 1U);
  EXPECT_DOUBLE_EQ(stats.summary().joint_read_callback_entry_to_command_completion.max_us, 0.01);
}

TEST(LatencyStats, InvalidTimingDoesNotConsumeCapacity) {
  LatencyStats stats{1U};
  EXPECT_FALSE(stats.add_control_tick(
      ControlTickObservation{.scheduled_ns = 20, .callback_start_ns = 10, .completion_ns = 30}));
  EXPECT_FALSE(stats.add_control_tick(
      ControlTickObservation{.scheduled_ns = 10, .callback_start_ns = 30, .completion_ns = 20}));
  EXPECT_EQ(stats.count(), 0U);
  EXPECT_TRUE(stats.add_control_tick(
      ControlTickObservation{.scheduled_ns = 10, .callback_start_ns = 20, .completion_ns = 30}));
}

/// Zero warmup still records the first real target, including version zero.
TEST(TargetResponseTracker, FirstRealTargetWithoutWarmup) {
  TargetResponseTracker tracker;
  EXPECT_FALSE(tracker.record(0, 0));
  EXPECT_TRUE(tracker.record(100, 0));
  EXPECT_FALSE(tracker.record(100, 0));
}

/// Warmup responses stay excluded while new targets at the boundary remain observable.
TEST(TargetResponseTracker, MeasurementBoundaryPreservesTargetHistory) {
  TargetResponseTracker tracker;
  EXPECT_TRUE(tracker.record(100, 1));   // First command during warmup.
  EXPECT_FALSE(tracker.record(100, 1));  // First measured command reuses that target.
  EXPECT_TRUE(tracker.record(200, 2));   // A new target's actual first command is retained.
  EXPECT_FALSE(tracker.record(200, 2));
}

/// Verifies P50, P99.99, and maximum on resolved distributions.
TEST(LatencyStats, StatisticsUseLinearInterpolationAndMaximum) {
  constexpr std::size_t kSamples = 10'001;
  constexpr std::int64_t kCallbackSpacingNs = 20'000'000;
  LatencyStats stats{kSamples};

  for (std::size_t index = 0; index < kSamples; ++index) {
    const auto sequence = static_cast<std::int64_t>(index);
    const std::int64_t scheduled_ns = 900'000'000 + sequence * kCallbackSpacingNs;
    const std::int64_t callback_start_ns = 1'000'000'000 + sequence * kCallbackSpacingNs;
    ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
        .sequence = sequence,
        .scheduled_ns = scheduled_ns,
        .callback_start_ns = callback_start_ns,
        .completion_ns = callback_start_ns + sequence * 1'000,
    }));
  }

  const KpiSummary result = stats.summary();
  EXPECT_EQ(result.joint_read_callback_entry_to_command_completion.count, kSamples);
  EXPECT_DOUBLE_EQ(result.joint_read_callback_entry_to_command_completion.p50_us, 5'000.0);
  EXPECT_DOUBLE_EQ(result.joint_read_callback_entry_to_command_completion.p99_99_us, 9'999.0);
  EXPECT_DOUBLE_EQ(result.joint_read_callback_entry_to_command_completion.max_us, 10'000.0);
  EXPECT_EQ(result.joint_read_callback_interval.count, kSamples - 1U);
  EXPECT_DOUBLE_EQ(result.joint_read_callback_interval.p50_us, 20'000.0);
  EXPECT_DOUBLE_EQ(result.joint_read_callback_interval.p99_99_us, 20'000.0);
  EXPECT_DOUBLE_EQ(result.joint_read_callback_interval.max_us, 20'000.0);
  EXPECT_DOUBLE_EQ(result.scheduled_activation_to_command_completion.p50_us, 105'000.0);
}

TEST(LatencyStats, ScheduledCompletionPercentilesUsePerCommandDurations) {
  LatencyStats stats{3U};
  const std::array<std::int64_t, 3> callback_delays_ns = {10'000, 30'000, 20'000};
  for (std::size_t index = 0; index < 3U; ++index) {
    const std::int64_t scheduled_ns = 1'000'000'000 + static_cast<std::int64_t>(index) * 1'000'000;
    ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
        .scheduled_ns = scheduled_ns,
        .callback_start_ns = scheduled_ns + callback_delays_ns[index],
        .completion_ns = scheduled_ns + 40'000,
    }));
  }

  const KpiSummary result = stats.summary();
  EXPECT_EQ(result.scheduled_activation_to_command_completion.count, 3U);
  EXPECT_DOUBLE_EQ(result.scheduled_activation_to_command_completion.p50_us, 40.0);
  EXPECT_DOUBLE_EQ(result.scheduled_activation_to_command_completion.p99_99_us, 40.0);
  EXPECT_DOUBLE_EQ(result.scheduled_activation_to_command_completion.max_us, 40.0);
  EXPECT_DOUBLE_EQ(result.scheduled_activation_to_joint_read_callback_entry.p50_us, 20.0);
  EXPECT_DOUBLE_EQ(result.scheduled_activation_to_joint_read_callback_entry.p99_99_us, 29.998);
  EXPECT_DOUBLE_EQ(result.joint_read_callback_entry_to_command_completion.p99_99_us, 29.998);
}

/// Verifies callback periods stay raw and interval counters retain distinct meanings.
TEST(LatencyStats, PeriodsAndCorrectnessCountersRemainUnfiltered) {
  LatencyStats stats{3U};
  ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
      .scheduled_ns = 999'900'000,
      .callback_start_ns = 1'000'000'000,
      .completion_ns = 1'000'100'000,
      .coalesced_ticks = 9U,
      .dropped_ticks = 9U,
  }));
  ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
      .sequence = 1,
      .scheduled_ns = 1'000'900'000,
      .callback_start_ns = 1'001'000'000,
      .completion_ns = 1'001'200'000,
      .coalesced_ticks = 2U,
  }));
  ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
      .sequence = 2,
      .scheduled_ns = 1'002'900'000,
      .callback_start_ns = 1'003'000'000,
      .completion_ns = 1'003'100'000,
      .dropped_ticks = 3U,
      .late_command = true,
  }));

  const KpiSummary result = stats.summary();
  EXPECT_EQ(result.joint_read_callback_entry_to_command_completion.count, 3U);
  EXPECT_EQ(result.joint_read_callback_interval.count, 2U);
  EXPECT_DOUBLE_EQ(result.joint_read_callback_interval.p50_us, 1'500.0);
  EXPECT_DOUBLE_EQ(result.joint_read_callback_interval.p99_99_us, 1'999.9);
  EXPECT_DOUBLE_EQ(result.joint_read_callback_interval.max_us, 2'000.0);
  EXPECT_EQ(result.late_commands, 1U);
  EXPECT_EQ(result.coalesced_ticks, 2U);
  EXPECT_EQ(result.dropped_ticks, 3U);
}

TEST(LatencyStats, ReportUsesTheStableV6Schema) {
  LatencyStats stats{2U};
  ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
      .scheduled_ns = 999'900'000,
      .callback_start_ns = 1'000'000'000,
      .completion_ns = 1'000'100'000,
  }));
  ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
      .sequence = 1,
      .scheduled_ns = 1'000'900'000,
      .callback_start_ns = 1'001'000'000,
      .completion_ns = 1'001'200'000,
      .coalesced_ticks = 2U,
      .dropped_ticks = 3U,
  }));

  testing::internal::CaptureStdout();
  stats.report("gpu-ik-clean", "clean", 1'000.0, 137.0);
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(std::count(output.begin(), output.end(), '\n'), 10);
  EXPECT_NE(output.find("interval between successive joint-read callback entries: samples 1, "
                        "P50 1000.00 us, P99.99 1000.00 us, "
                        "max 1000.00 us"),
            std::string::npos);
  EXPECT_NE(output.find("KPI scheduled activation -> joint-command completion: samples 2, "
                        "P50 250.00 us, P99.99 299.99 us, max 300.00 us"),
            std::string::npos);
  EXPECT_NE(output.find("KPI target produced -> first completed joint-command using that target: "
                        "samples 0, P50 N/A, P99.99 N/A, max N/A"),
            std::string::npos);
  EXPECT_LT(output.find("KPI scheduled activation -> joint-command completion"),
            output.find("KPI scheduled activation -> joint-read callback entry"));
  EXPECT_LT(output.find("KPI scheduled activation -> joint-read callback entry"),
            output.find("KPI target produced"));
  EXPECT_LT(output.find("KPI target produced"), output.find("correctness:"));
  EXPECT_LT(output.find("correctness:"), output.find("Additional metrics:"));
  EXPECT_EQ(output.find("diagnostic"), std::string::npos);
  const std::string expected_row =
      "BENCHCSV,gpu-ik-clean,clean,1000,137,2,250.00,299.99,300.00,2,100.00,100.00,100.00,"
      "0,,,,2,150.00,199.99,200.00,1,1000.00,1000.00,1000.00,0,2,3,bench-latency/v6";
  const std::size_t row_start = output.find("BENCHCSV,");
  ASSERT_NE(row_start, std::string::npos);
  const std::size_t row_end = output.find('\n', row_start);
  ASSERT_NE(row_end, std::string::npos);
  const std::string actual_row = output.substr(row_start, row_end - row_start);
  EXPECT_EQ(actual_row, expected_row);
  EXPECT_EQ(std::count(actual_row.begin(), actual_row.end(), ','), 28);
}

TEST(LatencyStats, ReportPreservesExactNonintegerRates) {
  LatencyStats stats{0U};
  const std::array<double, 4> control_rates = {1'000.25, 29.97, 0.125,
                                               std::nextafter(1'000.0, 2'000.0)};
  for (const double control_rate_hz : control_rates) {
    const double target_rate_hz = control_rate_hz / 3.0;
    testing::internal::CaptureStdout();
    stats.report("gpu-ik-clean", "clean", control_rate_hz, target_rate_hz);
    const std::string output = testing::internal::GetCapturedStdout();

    const std::size_t row_start = output.find("BENCHCSV,");
    ASSERT_NE(row_start, std::string::npos);
    const std::size_t row_end = output.find('\n', row_start);
    ASSERT_NE(row_end, std::string::npos);
    std::istringstream row{output.substr(row_start, row_end - row_start)};
    std::array<std::string, 29> fields;
    for (std::string& field : fields) {
      ASSERT_TRUE(std::getline(row, field, ','));
    }
    EXPECT_TRUE(row.eof());
    EXPECT_EQ(fields.back(), "bench-latency/v6");
    const double reported_control_rate_hz = std::stod(fields[3]);
    EXPECT_EQ(reported_control_rate_hz, control_rate_hz);
    EXPECT_EQ(std::stod(fields[4]), target_rate_hz);
    ASSERT_GT(reported_control_rate_hz, 0.0);
    EXPECT_EQ(static_cast<std::int64_t>(1e9 / reported_control_rate_hz),
              static_cast<std::int64_t>(1e9 / control_rate_hz));
  }
}

/// Verifies raw integer coordinates are sufficient to recompute every distribution.
TEST(LatencyStats, RawObservationExportPreservesCoordinates) {
  LatencyStats stats{8U};
  ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
      .sequence = 7,
      .scheduled_ns = 900,
      .callback_start_ns = 1'000,
      .completion_ns = 1'100,
      .target_produced_ns = 800,
      .target_use_ns = 950,
      .target_version = 4U,
  }));
  ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
      .sequence = 8,
      .scheduled_ns = 1'900,
      .callback_start_ns = 2'000,
      .completion_ns = 2'250,
      .target_produced_ns = 1'800,
      .target_use_ns = 1'950,
      .target_version = 5U,
      .first_command_for_target = true,
      .coalesced_ticks = 1U,
      .dropped_ticks = 2U,
      .late_command = true,
  }));
  const std::filesystem::path path =
      std::filesystem::path{::testing::TempDir()} / "robot-control-observations.csv";

  stats.write_observations_csv(path);
  std::ifstream input(path);
  const std::string contents{std::istreambuf_iterator<char>{input},
                             std::istreambuf_iterator<char>{}};
  std::filesystem::remove(path);

  EXPECT_EQ(std::count(contents.begin(), contents.end(), '\n'), 3);
  EXPECT_NE(contents.find("schema,index,sequence,scheduled_ns,callback_start_ns,completion_ns"),
            std::string::npos);
  EXPECT_NE(contents.find("bench-observations/v3,0,7,900,1000,1100,100,100,200,,0,0,0,"
                          "800,950,4,0,300"),
            std::string::npos);
  EXPECT_NE(contents.find("bench-observations/v3,1,8,1900,2000,2250,250,100,350,1000,1,1,2,"
                          "1800,1950,5,1,450"),
            std::string::npos);
}

/// Verifies target response includes only the first command using each new target version.
TEST(LatencyStats, TargetResponseUsesFirstCommandOnly) {
  LatencyStats stats{3U};
  ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
      .scheduled_ns = 900,
      .callback_start_ns = 1'000,
      .completion_ns = 1'100,
      .target_produced_ns = 700,
      .target_use_ns = 950,
      .target_version = 4U,
  }));
  ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
      .sequence = 1,
      .scheduled_ns = 1'900,
      .callback_start_ns = 2'000,
      .completion_ns = 2'200,
      .target_produced_ns = 1'600,
      .target_use_ns = 1'950,
      .target_version = 5U,
      .first_command_for_target = true,
  }));
  ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
      .sequence = 2,
      .scheduled_ns = 2'900,
      .callback_start_ns = 3'000,
      .completion_ns = 3'150,
      .target_produced_ns = 1'600,
      .target_use_ns = 2'950,
      .target_version = 5U,
  }));

  const KpiSummary result = stats.summary();
  EXPECT_EQ(result.target_produced_to_first_command_completion.count, 1U);
  EXPECT_DOUBLE_EQ(result.target_produced_to_first_command_completion.p50_us, 0.6);
  EXPECT_DOUBLE_EQ(result.target_produced_to_first_command_completion.p99_99_us, 0.6);
  EXPECT_DOUBLE_EQ(result.target_produced_to_first_command_completion.max_us, 0.6);

  testing::internal::CaptureStdout();
  stats.report("target-test", "clean", 1'000.0, 137.0);
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("KPI target produced -> first completed joint-command using that target: "
                        "samples 1, P50 0.60 us, P99.99 0.60 us, max 0.60 us"),
            std::string::npos);
}

TEST(LatencyStats, SingleCommandHasNoCallbackInterval) {
  LatencyStats stats{1U};
  ASSERT_TRUE(stats.add_control_tick(ControlTickObservation{
      .scheduled_ns = 900,
      .callback_start_ns = 1'000,
      .completion_ns = 1'100,
  }));
  testing::internal::CaptureStdout();
  stats.report("single-command", "clean", 1'000.0, 137.0);
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("interval between successive joint-read callback entries: samples 0, "
                        "P50 N/A, P99.99 N/A, max N/A"),
            std::string::npos);
}

}  // namespace
}  // namespace isaac::os::benchmark
