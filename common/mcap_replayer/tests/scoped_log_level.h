/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "holoscan/logger/logger.hpp"

namespace isaac::os::mcap_replayer {

// Sets the global log level for its lifetime, restoring the previous one on scope exit. Tests
// that deliberately drive a warning path wrap it in one of these: without that, a passing run
// still prints the warning to the test's own output, training anyone reading it to skim past
// warnings on a green run -- exactly how a real one gets missed later.
class ScopedLogLevel {
 public:
  explicit ScopedLogLevel(holoscan::LogLevel level) : previous_(holoscan::log_level()) {
    holoscan::set_log_level(level);
  }
  ~ScopedLogLevel() { holoscan::set_log_level(previous_); }

  ScopedLogLevel(const ScopedLogLevel&) = delete;
  ScopedLogLevel& operator=(const ScopedLogLevel&) = delete;

 private:
  holoscan::LogLevel previous_;
};

}  // namespace isaac::os::mcap_replayer
