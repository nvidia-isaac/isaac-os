/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <stdexcept>

#include "gtest/gtest.h"

#include "common/mcap_replayer/pacing.h"

namespace {
using isaac::os::mcap_replayer::PacingClock;
using Steady = std::chrono::steady_clock;
}  // namespace

TEST(PacingClock, ConstructionLatchesTheOriginToNow) {
  const Steady::time_point before = Steady::now();
  PacingClock clock(1.0);
  const Steady::time_point after = Steady::now();

  // The first target_for() call latches 1000 as the zero point, so its delta is zero and the
  // returned target is exactly the origin construction captured.
  const auto target = clock.target_for(1000ULL);
  EXPECT_GE(target, before);
  EXPECT_LE(target, after);
}

TEST(PacingClock, TargetsAreAbsoluteSoRoundingCannotAccumulate) {
  PacingClock clock(1.0);
  // The first call below latches 1'000'000'000ULL as the zero point -- and the origin it maps to,
  // whatever real instant construction captured -- that every later delta is measured against.
  const auto zero = clock.target_for(1'000'000'000ULL);

  // A 3 ns period across many messages: incremental accumulation would drift, absolute cannot.
  for (std::uint64_t index = 1; index < 100000; ++index) {
    const std::uint64_t log_time = 1'000'000'000ULL + index * 3ULL;
    const auto expected = zero + std::chrono::nanoseconds(index * 3ULL);
    EXPECT_EQ(clock.target_for(log_time), expected) << "at index " << index;
  }
}

TEST(PacingClock, PlaybackSpeedScalesTheIntervalNotTheOrigin) {
  PacingClock clock(2.0);
  const auto zero = clock.target_for(500ULL);
  EXPECT_EQ(clock.target_for(1500ULL), zero + std::chrono::nanoseconds(500));
}

TEST(PacingClock, TargetForThrowsWhenAskedAboutTimeBeforeTheOriginWithoutARebase) {
  PacingClock clock(1.0);
  static_cast<void>(clock.target_for(1000ULL));
  EXPECT_THROW(static_cast<void>(clock.target_for(400ULL)), std::logic_error);
}

TEST(PacingClock, SleepIsSkippedBelowTheEpsilon) {
  EXPECT_FALSE(PacingClock::should_sleep(std::chrono::microseconds(0)));
  EXPECT_FALSE(PacingClock::should_sleep(std::chrono::microseconds(199)));
  EXPECT_TRUE(PacingClock::should_sleep(std::chrono::microseconds(201)));
  EXPECT_FALSE(PacingClock::should_sleep(std::chrono::microseconds(-5000)));
}

TEST(PacingClock, RebaseRestartsTheTimelineWhenTheReaderWraps) {
  PacingClock clock(1.0);
  const Steady::time_point second_pass = Steady::time_point{} + std::chrono::seconds(5);
  clock.rebase(1000ULL, second_pass);
  EXPECT_EQ(clock.target_for(1000ULL), second_pass);
  EXPECT_EQ(clock.target_for(2000ULL), second_pass + std::chrono::nanoseconds(1000));
}
