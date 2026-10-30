/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include "holoscan/core/compile.hpp"
#include "holoscan/core/diagnostic.hpp"
#include "holoscan/core/run.hpp"

#include "common/mcap_replayer/replay_stats.h"
#include "replay_graph.h"

namespace isaac::os::mcap_replay_reference {
namespace {

struct Options {
  std::filesystem::path input_path;
  double playback_speed = 1.0;
  bool loop = false;
  bool visualize = false;
  bool help = false;
};

// The whole value must be a finite number above zero: "2x" is not 2, and an infinite speed would
// collapse every pacing interval to nothing.
double parse_playback_speed(const std::string& value) {
  constexpr const char* kMessage = "--playback-speed must be a finite number greater than zero.";
  std::size_t consumed = 0;
  double speed = 0.0;
  try {
    speed = std::stod(value, &consumed);
  } catch (const std::logic_error&) {
    throw std::invalid_argument(kMessage);
  }
  if (consumed != value.size() || !std::isfinite(speed) || !(speed > 0.0)) {
    throw std::invalid_argument(kMessage);
  }
  return speed;
}

Options parse_options(std::span<char* const> arguments) {
  Options options;
  for (std::size_t index = 1; index < arguments.size(); ++index) {
    const std::string_view argument{arguments[index]};
    if (argument == "--help" || argument == "-h") {
      options.help = true;
    } else if (argument == "--visualize") {
      options.visualize = true;
    } else if (argument == "--loop") {
      options.loop = true;
    } else if (argument.starts_with("--input-path=")) {
      options.input_path = argument.substr(std::string_view{"--input-path="}.size());
    } else if (argument.starts_with("--playback-speed=")) {
      const std::string value{argument.substr(std::string_view{"--playback-speed="}.size())};
      options.playback_speed = parse_playback_speed(value);
    } else {
      throw std::invalid_argument("Unknown argument: " + std::string{argument} + ".");
    }
  }
  // Checked before the required argument, so --help alone is not rejected for lacking it.
  if (options.help) {
    return options;
  }
  if (options.input_path.empty()) {
    throw std::invalid_argument("--input-path is required.");
  }
  return options;
}

void print_usage(std::string_view executable) {
  std::cout << "Usage: " << executable
            << " --input-path=<recorded .mcap file>"
               " [--playback-speed=<positive number, default 1.0>]"
               " [--loop] [--visualize]\n";
}

// True once the replayer has given up on a run because a topic's consumer never subscribed or
// reading the recording stalled. It sends nothing after that and has already logged why, and the
// graph is still running, so neither wait below would otherwise end.
bool replay_never_began(const isaac::os::mcap_replayer::ReplayStats& stats) {
  return stats.start_stalled() || std::ranges::any_of(kReplayedTopics, [&stats](const char* topic) {
           return stats.never_subscribed(topic);
         });
}

constexpr const char* kNeverBegan = "The replay never began, so nothing was replayed.";
constexpr const char* kReadFailed =
    "Reading the recording failed partway through, so the replay is incomplete.";
constexpr const char* kGraphStopped =
    "The graph stopped before the replay was complete, so the replay is incomplete.";

// Why the replayer gave up on the run, or nothing if it did not. It has already logged the cause,
// and its summary says how far the run got.
const char* reason_replay_gave_up(const isaac::os::mcap_replayer::ReplayStats& stats) {
  if (replay_never_began(stats)) {
    return kNeverBegan;
  }
  return stats.read_failed() ? kReadFailed : nullptr;
}

// Set by handle_sigint. A looping replay never ends on its own, so Ctrl+C ends it. A
// function-local static, not a global, and first used before the handler is installed so it is
// already constructed when a signal arrives.
std::atomic<bool>& interrupted() {
  static std::atomic flag{false};
  return flag;
}

void handle_sigint(int /*signal*/) { interrupted().store(true); }

// A wait ends with nothing when the run is done, or with the reason it cannot be: the replayer
// gave up, or the graph stopped on its own. Nothing in this graph ends by itself, so a graph that
// has stopped failed.
using Failure = const char*;

// Waits for Ctrl+C. A looping replay has no end of its own.
Failure wait_until_interrupted(const isaac::os::mcap_replayer::McapReplayer& replayer,
                               const holoscan::RunSession& run_session) {
  interrupted();  // constructed here, not inside the handler
  std::signal(SIGINT, handle_sigint);
  while (!interrupted().load()) {
    if (const Failure reason = reason_replay_gave_up(replayer.stats())) {
      return reason;
    }
    if (!run_session.is_running()) {
      return kGraphStopped;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return nullptr;
}

// Waits for the replayer to report that the run has ended, which for a recording played once is
// when everything it read has gone out. It also ends when the replayer gave up, which the check
// afterwards tells apart.
Failure wait_until_replayed(const isaac::os::mcap_replayer::McapReplayer& replayer,
                            const holoscan::RunSession& run_session) {
  while (!replayer.finished()) {
    if (!run_session.is_running()) {
      return kGraphStopped;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return reason_replay_gave_up(replayer.stats());
}

int run(const Options& options) {
  const isaac::os::mcap_replayer::ReplayOptions replay_options{
      .playback_speed = options.playback_speed,
      .loop = options.loop,
  };
  const std::optional<std::string> skip_reason =
      encoded_topic_skip_reason(options.input_path, replay_options);
  if (skip_reason) {
    HOLOSCAN_LOG_INFO("The encoded counter stream is not replayed. {}", *skip_reason);
  }
  isaac::os::mcap_replayer::McapReplayer replayer =
      make_replayer(options.input_path, replay_options, !skip_reason);
  ReplayGraph graph(replayer, options.visualize, !skip_reason);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << "Unable to compile the mcap_replay_reference graph.\n";
    for (const holoscan::Diagnostic& diagnostic : plan.diagnostics()) {
      std::cerr << "  " << diagnostic.token << " (service " << diagnostic.service << ")\n";
    }
    return 1;
  }

  holoscan::RunSession run_session = holoscan::run_async(plan);
  replayer.start(plan.endpoint_catalog());

  std::cout << "Replaying " << options.input_path << " at " << options.playback_speed << "x";
  std::cout << (options.loop ? ", looping until interrupted (Ctrl+C)...\n" : "...\n") << std::flush;
  Failure failure = options.loop ? wait_until_interrupted(replayer, run_session)
                                 : wait_until_replayed(replayer, run_session);
  if (failure == nullptr) {
    // The final messages are still in flight to their consumers when the last one is emitted, so
    // let them land, then check the graph survived them.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    failure = run_session.is_running() ? nullptr : kGraphStopped;
  }

  replayer.stop();
  run_session.request_stop();
  run_session.wait();
  if (failure != nullptr) {
    std::cerr << failure << '\n';
    return 1;
  }
  return 0;
}

}  // namespace
}  // namespace isaac::os::mcap_replay_reference

int main(int argc, char** argv) {
  try {
    const isaac::os::mcap_replay_reference::Options options =
        isaac::os::mcap_replay_reference::parse_options(
            std::span<char* const>(argv, static_cast<std::size_t>(argc)));
    if (options.help) {
      isaac::os::mcap_replay_reference::print_usage(argv[0]);
      return 0;
    }
    return isaac::os::mcap_replay_reference::run(options);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    isaac::os::mcap_replay_reference::print_usage(argv[0]);
    return 1;
  }
}
