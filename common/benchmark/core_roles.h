/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#if defined(__linux__)
#include <sched.h>
#endif

namespace isaac::os::benchmark {

inline constexpr int kGr00tFifoPriority = 5;
inline constexpr int kWbcFifoPriority = 10;

struct CoreRoles {
  std::optional<std::uint32_t> controller;
  std::optional<std::uint32_t> background;
  std::optional<std::uint32_t> gr00t;
  std::optional<std::uint32_t> wbc;

  [[nodiscard]] bool enabled() const {
    return controller.has_value() && background.has_value() && gr00t.has_value() && wbc.has_value();
  }

  void validate() const {
    const std::array fields{controller, background, gr00t, wbc};
    const auto provided = std::count_if(fields.begin(), fields.end(),
                                        [](const auto& core) { return core.has_value(); });
    if (provided != 0 && provided != fields.size()) {
      throw std::invalid_argument(
          "--benchmark-controller-core, --benchmark-background-core, "
          "--benchmark-gr00t-core, and --benchmark-wbc-core must be provided together.");
    }
    if (!enabled()) return;

    const std::array roles{
        std::pair<std::string_view, std::uint32_t>{"controller", *controller},
        std::pair<std::string_view, std::uint32_t>{"background", *background},
        std::pair<std::string_view, std::uint32_t>{"gr00t", *gr00t},
        std::pair<std::string_view, std::uint32_t>{"wbc", *wbc},
    };
    for (std::size_t index = 0; index < roles.size(); ++index) {
      for (std::size_t other = index + 1; other < roles.size(); ++other) {
        if (roles[index].second == roles[other].second) {
          throw std::invalid_argument("Benchmark roles '" + std::string{roles[index].first} +
                                      "' and '" + std::string{roles[other].first} +
                                      "' cannot use the same CPU core.");
        }
      }
    }

#if defined(__linux__)
    cpu_set_t available;
    CPU_ZERO(&available);
    if (sched_getaffinity(0, sizeof(available), &available) != 0) {
      throw std::runtime_error("Unable to inspect the benchmark process CPU affinity.");
    }
    for (const auto& [role, core] : roles) {
      if (core >= CPU_SETSIZE || !CPU_ISSET(static_cast<int>(core), &available)) {
        throw std::invalid_argument("Benchmark " + std::string{role} + " core " +
                                    std::to_string(core) + " is not available to this process.");
      }
    }
#else
    throw std::runtime_error("Benchmark CPU placement requires Linux.");
#endif
  }
};

inline void pin_current_thread(std::uint32_t core) {
#if defined(__linux__)
  cpu_set_t requested;
  CPU_ZERO(&requested);
  CPU_SET(static_cast<int>(core), &requested);
  if (sched_setaffinity(0, sizeof(requested), &requested) != 0) {
    throw std::runtime_error("Unable to pin the benchmark controller thread.");
  }
#else
  (void)core;
  throw std::runtime_error("Benchmark CPU placement requires Linux.");
#endif
}

}  // namespace isaac::os::benchmark
