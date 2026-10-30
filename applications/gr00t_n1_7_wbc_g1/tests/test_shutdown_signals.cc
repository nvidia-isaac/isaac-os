/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "applications/gr00t_n1_7_wbc_g1/shutdown_signals.h"

namespace isaac::os {
namespace {

using namespace std::chrono_literals;

struct ProbeState {
  std::atomic<bool> computed{false};
  std::atomic<bool> stopped{false};
  bool complete{};
  bool fail{};
};

class ProbeStateHandle {
 public:
  [[nodiscard]] ProbeState* operator->() const { return state_.get(); }

 private:
  std::shared_ptr<ProbeState> state_ = std::make_shared<ProbeState>();
};

class ShutdownProbeOp final : public holoscan::Operator<> {
 public:
  explicit ShutdownProbeOp(ProbeStateHandle state) : state_(std::move(state)) {}

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 1ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    state_->computed.store(true);
    if (state_->fail) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    if (state_->complete) {
      context.request_completion();
    }
    return {};
  }

  void stop() override { state_->stopped.store(true); }

 private:
  ProbeStateHandle state_;
};

holoscan::ExecutionPlan make_plan(ProbeStateHandle state = {}) {
  holoscan::Graph graph{"shutdown-test"};
  graph.op<ShutdownProbeOp>("probe", std::move(state));
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  return holoscan::compile(graph);
}

int stop_with_existing_thread(int signal) {
  alarm(10);
  const ProbeStateHandle state;
  std::atomic<bool> peer_ready{false};
  std::jthread peer([state, &peer_ready, signal](std::stop_token stop) {
    peer_ready.store(true);
    while (!state->computed.load() && !stop.stop_requested()) {
      std::this_thread::sleep_for(1ms);
    }
    if (!stop.stop_requested()) {
      // Deliver on this thread, which predates signal setup, as a profiler thread can.
      raise(signal);
    }
  });
  while (!peer_ready.load()) {
    std::this_thread::yield();
  }

  const ShutdownSignals signals;
  const auto plan = make_plan(state);
  if (!plan.ok()) {
    return 1;
  }
  auto session = holoscan::run_async(plan);
  signals.wait(session);
  peer.join();
  return signals.stop_requested() && state->stopped.load() &&
                 session.termination() == holoscan::RunTermination::kStopped
             ? 0
             : 1;
}

class ShutdownSignalsTest : public ::testing::TestWithParam<int> {};

TEST_P(ShutdownSignalsTest, StopsAndCleansUpWithExistingThread) {
  const int signal = GetParam();
  EXPECT_EXIT(_exit(stop_with_existing_thread(signal)), ::testing::ExitedWithCode(0), "");
}

TEST_P(ShutdownSignalsTest, RetainsSignalReceivedBeforeSessionStarts) {
  const int signal = GetParam();
  EXPECT_EXIT(
      {
        alarm(10);
        const ShutdownSignals signals;
        raise(signal);
        auto session = holoscan::run_async(make_plan());
        signals.wait(session);
        _exit(signals.stop_requested() &&
                      session.termination() == holoscan::RunTermination::kStopped
                  ? 0
                  : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_P(ShutdownSignalsTest, RepeatedSignalForcesTermination) {
  const int signal = GetParam();
  EXPECT_EXIT(
      {
        alarm(10);
        const ShutdownSignals signals;
        raise(signal);
        if (!signals.stop_requested()) {
          _exit(1);
        }
        raise(signal);
        _exit(1);
      },
      ::testing::KilledBySignal(signal), "");
}

INSTANTIATE_TEST_SUITE_P(InterruptAndTerminate, ShutdownSignalsTest,
                         ::testing::Values(SIGINT, SIGTERM));

int finish_without_signal(bool fail) {
  alarm(10);
  const ShutdownSignals signals;
  const ProbeStateHandle state;
  state->complete = !fail;
  state->fail = fail;
  auto session = holoscan::run_async(make_plan(state));
  signals.wait(session);
  const auto expected =
      fail ? holoscan::RunTermination::kFailed : holoscan::RunTermination::kRequestedCompleted;
  return !signals.stop_requested() && state->stopped.load() && session.termination() == expected
             ? 0
             : 1;
}

TEST(ShutdownSignals, CompletesWithoutSignal) {
  EXPECT_EXIT(_exit(finish_without_signal(false)), ::testing::ExitedWithCode(0), "");
}

TEST(ShutdownSignals, PreservesRuntimeFailure) {
  EXPECT_EXIT(_exit(finish_without_signal(true)), ::testing::ExitedWithCode(0), "");
}

int restore_process_signals() {
  alarm(10);
  struct sigaction ignored{};
  ignored.sa_handler = SIG_IGN;
  sigemptyset(&ignored.sa_mask);
  if (sigaction(SIGINT, &ignored, nullptr) != 0 || sigaction(SIGTERM, &ignored, nullptr) != 0) {
    return 1;
  }
  sigset_t blocked;
  sigemptyset(&blocked);
  sigaddset(&blocked, SIGTERM);
  if (pthread_sigmask(SIG_BLOCK, &blocked, nullptr) != 0) {
    return 1;
  }
  {
    const ShutdownSignals signals;
    // Setup unblocks the creator even when the launcher blocked termination signals.
    raise(SIGTERM);
    if (!signals.stop_requested()) {
      return 1;
    }
    try {
      const ShutdownSignals duplicate;
      return 1;
    } catch (const std::logic_error&) {
      // Rejecting a second signal owner is the expected outcome.
    }
    if (!signals.stop_requested()) {
      return 1;
    }
  }
  struct sigaction interrupt_action{};
  struct sigaction terminate_action{};
  sigset_t restored;
  if (sigaction(SIGINT, nullptr, &interrupt_action) != 0 ||
      sigaction(SIGTERM, nullptr, &terminate_action) != 0 ||
      pthread_sigmask(SIG_BLOCK, nullptr, &restored) != 0 ||
      interrupt_action.sa_handler != SIG_IGN || terminate_action.sa_handler != SIG_IGN ||
      sigismember(&restored, SIGTERM) != 1) {
    return 1;
  }
  const ShutdownSignals fresh;
  return fresh.stop_requested() ? 1 : 0;
}

TEST(ShutdownSignals, RestoresDispositionsMaskAndOwnership) {
  EXPECT_EXIT(_exit(restore_process_signals()), ::testing::ExitedWithCode(0), "");
}

}  // namespace
}  // namespace isaac::os
