/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace isaac::os::mcap_logger {

// How often a repeating failure is reported after its first occurrence: every Nth one.
inline constexpr std::uint64_t kFailureReportInterval = 10;

// Whether the Nth occurrence of a repeating failure, counting from 1, should be reported: the 1st,
// then every tenth. The one rule for every repeating warning in this component.
//
// A failure that repeats on every frame must not be logged every time, but it must not go quiet
// either: a fixed cap leaves a topic recording nothing for the rest of a run under a log that has
// announced it will say no more. Every tenth keeps the problem in front of the reader, each message
// carrying the running total. At 1000 failures a second that is still 100 lines a second, on
// purpose; raise kFailureReportInterval if it is too loud.
[[nodiscard]] constexpr bool IsFailureReportDue(std::uint64_t occurrence) noexcept {
  return occurrence == 1 || (occurrence != 0 && occurrence % kFailureReportInterval == 0);
}

}  // namespace isaac::os::mcap_logger
