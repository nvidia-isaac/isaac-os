/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Verify the two physical coordinates used by the benchmark: the OnClock scheduled activation
// and the callback's direct clock observation. TagTime is deliberately excluded because it is
// logical ordering evidence rather than a physical timestamp.

#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::int64_t kPeriodNs = 1'000'000;
constexpr std::size_t kActivations = 200;

// One activation's scheduled and observed coordinates, exactly as the runtime reported them.
struct Coordinates {
  std::int64_t activation_time_ns{};
  std::int64_t clock_now_ns{};
  std::uint64_t activation_clock_id{};
  std::uint64_t clock_id{};
};

class CoordinateCapture {
 public:
  /// Clears samples retained by the preceding probe run.
  static void reset() {
    const std::scoped_lock lock(mutex_);
    recorded_.clear();
  }

  /// Retains one activation until the requested sample count is reached.
  static void record(const Coordinates& coordinates) {
    const std::scoped_lock lock(mutex_);
    if (recorded_.size() < kActivations) {
      recorded_.push_back(coordinates);
    }
  }

  /// Returns the number of activation samples currently retained.
  static std::size_t count() {
    const std::scoped_lock lock(mutex_);
    return recorded_.size();
  }

  /// Returns a snapshot of all retained activation samples.
  static std::vector<Coordinates> take() {
    const std::scoped_lock lock(mutex_);
    return recorded_;
  }

 private:
  static std::mutex mutex_;
  static std::vector<Coordinates> recorded_;
};

std::mutex CoordinateCapture::mutex_;
std::vector<Coordinates> CoordinateCapture::recorded_;

}  // namespace

namespace isaac::os::ops {

class CoordinateProbeOp final : public holoscan::Operator<> {
 public:
  /// Declares the probe as a portless source operator.
  void setup(holoscan::OperatorSpec&) override {
    // This source operator has no ports.
  }

  /// Activates the probe on the benchmark's one-millisecond clock period.
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::nanoseconds{kPeriodNs}});
    return result;
  }

  /// Captures the scheduled activation and direct clock observation for one callback.
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    const auto clock = context.clock();
    const auto now = clock.now();
    if (!now) {
      return holoscan::make_unexpected(std::move(now).error());
    }
    const holoscan::TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kProtocolError, "The scheduled activation time must be valid."});
    }
    CoordinateCapture::record(Coordinates{
        .activation_time_ns = activation.timestamp_ns,
        .clock_now_ns = now->timestamp_ns,
        .activation_clock_id = activation.clock_id.value,
        .clock_id = now->clock_id.value,
    });
    if (CoordinateCapture::count() >= kActivations) {
      context.request_completion();
    }
    return {};
  }
};

}  // namespace isaac::os::ops

namespace {

/// Runs a real scheduled graph and returns the requested activation coordinates.
std::vector<Coordinates> run_probe() {
  CoordinateCapture::reset();
  holoscan::Graph graph{"activation-coordinates"};
  const auto probe = graph.op<isaac::os::ops::CoordinateProbeOp>("probe");
  graph.partition("probe").add(probe);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  EXPECT_TRUE(plan.ok()) << plan.json();
  if (!plan.ok()) {
    return {};
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (CoordinateCapture::count() < kActivations && std::chrono::steady_clock::now() < deadline &&
         session.termination() == holoscan::RunTermination::kRunning) {
    std::this_thread::sleep_for(2ms);
  }
  session.request_stop();
  session.wait();
  return CoordinateCapture::take();
}

/// Verifies that scheduled and observed coordinates use the admitted graph clock.
TEST(ActivationCoordinates, ScheduledActivationUsesTheAdmittedClock) {
  const auto recorded = run_probe();
  ASSERT_EQ(recorded.size(), kActivations);

  // Both coordinates live in the admitted activation clock, so no cross-clock projection is
  // needed before computing wake or end-to-end latency.
  for (const Coordinates& coordinates : recorded) {
    EXPECT_EQ(coordinates.activation_clock_id, coordinates.clock_id);
  }
}

/// Verifies that runtime schedule gaps remain whole multiples of the configured period.
TEST(ActivationCoordinates, TheScheduleIsAnExactPeriodGridUnaffectedByDispatchDelay) {
  const auto recorded = run_probe();
  ASSERT_EQ(recorded.size(), kActivations);

  // Coalesced ticks leave a gap in the schedule, so consecutive scheduled activations are a
  // whole number of periods apart.
  for (std::size_t index = 1; index < recorded.size(); ++index) {
    const std::int64_t advance =
        recorded[index].activation_time_ns - recorded[index - 1].activation_time_ns;
    EXPECT_GT(advance, 0);
    EXPECT_EQ(advance % kPeriodNs, 0) << "tick " << index << " advanced by " << advance << " ns";
  }
}

/// Verifies that the direct clock observation adds information beyond the scheduled coordinate.
TEST(ActivationCoordinates, ClockNowAddsCallbackEntryTiming) {
  const auto recorded = run_probe();
  ASSERT_EQ(recorded.size(), kActivations);

  std::size_t delayed = 0;
  for (const Coordinates& coordinates : recorded) {
    if (coordinates.clock_now_ns != coordinates.activation_time_ns) {
      ++delayed;
    }
  }
  // A callback is not expected to start exactly on its scheduled coordinate for every activation.
  // The sign is intentionally unconstrained because early dispatch is valid.
  EXPECT_GT(delayed, 0U);
}

}  // namespace
