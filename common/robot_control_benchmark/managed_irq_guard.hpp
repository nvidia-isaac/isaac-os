/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace isaac::os::benchmark {

/// Compare interrupt totals on selected CPUs for sources containing any requested substring.
class ManagedIrqGuard {
 public:
  // The explicit path supports procfs fixtures in tests; applications use the default.
  ManagedIrqGuard(std::vector<unsigned> cpus, std::vector<std::string> source_substrings,
                  std::filesystem::path proc_root = "/proc")
      : proc_root_(std::move(proc_root)),
        cpus_(std::move(cpus)),
        source_substrings_(std::move(source_substrings)) {
    try {
      if (cpus_.empty()) throw CaptureError("No CPUs were selected for IRQ capture.");
      if (source_substrings_.empty() ||
          std::ranges::any_of(source_substrings_,
                              [](const auto& source) { return source.empty(); })) {
        throw CaptureError("Select at least one nonempty interrupt source substring.");
      }
      std::ranges::sort(cpus_);
      const auto duplicates = std::ranges::unique(cpus_);
      cpus_.erase(duplicates.begin(), duplicates.end());
      before_ = capture();
    } catch (const CaptureError& error) {
      initial_error_ = error.what();
    }
  }

  /// Warn on observed activity or unavailable evidence; unchanged counters produce no output.
  void check() const {
    if (!initial_error_.empty()) {
      warn_unavailable(initial_error_);
      return;
    }
    try {
      const auto after = capture();
      if (after < before_) {
        throw CaptureError("The matching interrupt total decreased; the run cannot be verified.");
      }
      const auto interrupts = after - before_;
      if (interrupts == 0) return;
      std::string cpus;
      for (const auto cpu : cpus_) cpus += std::format(" {}", cpu);
      std::string sources;
      for (const auto& source : source_substrings_) sources += std::format(" '{}'", source);
      std::cerr << std::format(
          "Warning: Observed {} interrupts on CPUs{} during the run. Source substrings:{}. "
          "Interrupt handling can delay benchmark work and increase the measured latency. "
          "Managed IRQs are routed by the kernel and may still target isolated CPUs despite "
          "CPU isolation settings: isolcpus=managed_irq provides best-effort isolation and "
          "cannot redirect an IRQ if its affinity mask contains no online non-isolated CPU.\n",
          interrupts, cpus, sources);
    } catch (const CaptureError& error) {
      warn_unavailable(error.what());
    }
  }

 private:
  class CaptureError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
  };

  static void warn_unavailable(const std::string& detail) {
    std::cerr << std::format("Warning: Unable to check interrupt activity. {}\n", detail);
  }

  static unsigned cpu_number(std::string_view token) {
    unsigned cpu{};
    if (!token.starts_with("CPU")) {
      throw CaptureError("The interrupt table has an invalid CPU header.");
    }
    const auto [end, error] = std::from_chars(token.data() + 3, token.data() + token.size(), cpu);
    if (error != std::errc{} || end != token.data() + token.size()) {
      throw CaptureError("The interrupt table has an invalid CPU header.");
    }
    return cpu;
  }

  static void add_count(std::uint64_t& total, std::uint64_t count) {
    if (count > std::numeric_limits<std::uint64_t>::max() - total) {
      throw CaptureError("The matching interrupt total overflowed; the run cannot be verified.");
    }
    total += count;
  }

  bool matches_source(std::string_view source) const {
    return std::ranges::any_of(source_substrings_, [source](const auto& substring) {
      return !std::ranges::search(source, substring).empty();
    });
  }

  std::uint64_t capture() const {
    const auto path = proc_root_ / "interrupts";
    std::ifstream input(path);
    if (!input)
      throw CaptureError(std::format("Unable to read IRQ evidence from {}.", path.string()));
    std::string line;
    std::getline(input, line);
    std::istringstream header(line);
    std::vector<unsigned> cpus;
    std::string token;
    while (header >> token) cpus.push_back(cpu_number(token));
    for (const auto cpu : cpus_) {
      if (std::ranges::count(cpus, cpu) != 1) {
        throw CaptureError(
            std::format("The interrupt table must include CPU {} exactly once.", cpu));
      }
    }

    std::uint64_t total = 0;
    bool found_matching = false;
    while (std::getline(input, line)) {
      // Match only the interrupt source in the last field, not controller names or counters.
      const auto end = line.find_last_not_of(" \t\r");
      if (end == std::string::npos) continue;
      line.resize(end + 1);
      if (const auto action = line.find_last_of(" \t");
          action == std::string::npos || !matches_source(std::string_view(line).substr(action + 1)))
        continue;
      std::istringstream fields(line);
      fields >> token;  // IRQ label; identities are not compared between snapshots.
      for (const auto cpu : cpus) {
        std::uint64_t count{};
        if (!(fields >> token)) {
          throw CaptureError("The matching interrupt row has missing counters.");
        }
        const auto [last, error] =
            std::from_chars(token.data(), token.data() + token.size(), count);
        if (error != std::errc{} || last != token.data() + token.size()) {
          throw CaptureError("The matching interrupt row has an invalid counter.");
        }
        if (std::ranges::find(cpus_, cpu) != cpus_.end()) add_count(total, count);
      }
      found_matching = true;
    }
    if (input.bad()) throw CaptureError(std::format("Unable to finish reading {}.", path.string()));
    if (!found_matching) {
      throw CaptureError("The interrupt table has no matching interrupt counters.");
    }
    return total;
  }

  std::filesystem::path proc_root_;
  std::vector<unsigned> cpus_;
  std::vector<std::string> source_substrings_;
  std::uint64_t before_ = 0;
  std::string initial_error_;
};

}  // namespace isaac::os::benchmark
