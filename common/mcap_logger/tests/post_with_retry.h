/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <thread>

#include "holoscan/core/errors.hpp"
#include "holoscan/core/readiness_source.hpp"

namespace isaac::os::mcap_logger::testing {

// A NotificationSender obtained during a generator operator's Arm lifecycle stage (the pattern
// every test in this directory uses, via a *TriggerProbe that publishes the sender once arm()
// runs) can have its first post() -- or, more rarely, a post() shortly after -- fail with
// ErrorCode::kNotReady: per NotificationSender::post()'s own doc comment
// (holoscan/core/readiness_source.hpp), that means "Binding exists but its operator has not
// completed kStart" -- and kStart happens after Arm. The same doc comment documents kNotReady (like
// kBackpressured) as retryable by the caller, not terminal. This helper retries specifically that
// case, bounded so a genuinely terminal error (or an unexpectedly long stall) still surfaces
// instead of hanging.
[[nodiscard]] inline holoscan::expected<void, holoscan::Error> post_with_retry(
    holoscan::NotificationSender& sender) {
  constexpr int kMaxAttempts = 200;
  constexpr auto kRetryInterval = std::chrono::milliseconds(1);
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    auto posted = sender.post();
    if (posted || posted.error().code != holoscan::ErrorCode::kNotReady) {
      return posted;
    }
    std::this_thread::sleep_for(kRetryInterval);
  }
  return holoscan::make_unexpected(
      holoscan::Error{holoscan::ErrorCode::kTimeout,
                      "Timed out waiting for the generator to become ready before posting."});
}

}  // namespace isaac::os::mcap_logger::testing
