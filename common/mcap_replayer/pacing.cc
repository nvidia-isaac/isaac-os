/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/mcap_replayer/pacing.h"

#include <cmath>
#include <stdexcept>

namespace isaac::os::mcap_replayer {

PacingClock::PacingClock(double playback_speed)
    : playback_speed_(playback_speed), origin_(std::chrono::steady_clock::now()) {}

void PacingClock::rebase(std::uint64_t first_log_time_ns,
                         std::chrono::steady_clock::time_point origin) {
  first_log_time_ns_ = first_log_time_ns;
  origin_ = origin;
}

std::chrono::steady_clock::time_point PacingClock::target_for(std::uint64_t log_time_ns) {
  if (!first_log_time_ns_.has_value()) {
    first_log_time_ns_ = log_time_ns;
  }
  if (log_time_ns < *first_log_time_ns_) {
    // Only a wrap the caller failed to rebase() for can produce this; see the header.
    throw std::logic_error(
        "PacingClock::target_for() was asked about a log_time_ns earlier than its current "
        "origin; the caller should have rebase()'d before this call.");
  }
  const std::uint64_t delta_ns = log_time_ns - *first_log_time_ns_;
  const auto scaled_ns =
      static_cast<std::int64_t>(std::llround(static_cast<double>(delta_ns) / playback_speed_));
  return origin_ + std::chrono::nanoseconds(scaled_ns);
}

bool PacingClock::should_sleep(std::chrono::steady_clock::duration remaining) {
  return remaining > kNoSleepEpsilon;
}

}  // namespace isaac::os::mcap_replayer
