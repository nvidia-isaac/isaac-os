/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace isaac::os::mcap_replayer {

// Pure timing arithmetic for pacing recorded messages back out at their original cadence. Holds
// no thread and no Holoscan dependency, so the timing rules below are unit-testable without
// building a graph.
//
// Every wall-clock target is an absolute offset from first_log_time_ns_ and origin_, never an
// increment on the previously returned target: accumulating deltas would compound rounding error
// into drift that grows with message count.
class PacingClock {
 public:
  // A wake-up costs more to schedule than this interval would spend waiting, so sleeping for less
  // cannot improve where the message lands and only adds scheduling jitter. Sized from a measured
  // wake-up latency of p50 50.7 us / p99 115.6 us.
  static constexpr std::chrono::microseconds kNoSleepEpsilon{200};

  // Latches the origin to the wall-clock instant of construction -- the caller is expected to
  // construct this right when it is ready to start timing, not before.
  explicit PacingClock(double playback_speed);

  // Restarts the timeline for a repeat pass: first_log_time_ns becomes the new zero point and
  // origin the wall-clock instant it maps to.
  void rebase(std::uint64_t first_log_time_ns, std::chrono::steady_clock::time_point origin);

  // Returns the wall-clock instant at which log_time_ns should be emitted. The first call after
  // construction latches first_log_time_ns_ to the log_time_ns it is given, assuming it names the
  // earliest message of the run; later calls are relative to that value until the next rebase().
  //
  // Log-times are non-decreasing within a pass, so one earlier than first_log_time_ns_ means the
  // caller missed a wrap and should have called rebase() first. Throws std::logic_error in that
  // case rather than clamping to the origin, since a clamp would hide the bug behind a
  // plausible-looking target. A caller on a thread with no handler for it must catch it there.
  [[nodiscard]] std::chrono::steady_clock::time_point target_for(std::uint64_t log_time_ns);

  // A negative remaining duration means the message is already overdue and should be posted
  // immediately rather than slept on.
  [[nodiscard]] static bool should_sleep(std::chrono::steady_clock::duration remaining);

 private:
  std::optional<std::uint64_t> first_log_time_ns_;
  double playback_speed_;
  std::chrono::steady_clock::time_point origin_;
};

}  // namespace isaac::os::mcap_replayer
