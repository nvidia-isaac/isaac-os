/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/so101_driver/tests/feetech_test_support.h"
#include "operators/so101/so101_io_ops.h"

namespace isaac::os::ops {
namespace {
using namespace std::chrono_literals;
using so101::testing::CalibrationFile;
using so101::testing::FakeFeetechBus;
using so101::testing::options_for;

class StateCaptureOp final : public holoscan::Operator<> {
 public:
  explicit StateCaptureOp(std::shared_ptr<std::vector<isaac::StateSampleCpuT>> samples)
      : samples_(std::move(samples)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(state, "in_state"); }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{state});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto sample = state.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());
    samples_->push_back(sample->data);
    // Finish only after telemetry has recovered from two distinct failure streaks.
    if (samples_->back().fault_epoch > samples_->front().fault_epoch) {
      context.request_completion();
    }
    return {};
  }
  holoscan::Input<isaac::StateSampleCpuT> state;

 private:
  std::shared_ptr<std::vector<isaac::StateSampleCpuT>> samples_;
};

void run_source(const std::shared_ptr<so101::FeetechSerialDriver>& driver,
                const std::shared_ptr<std::vector<isaac::StateSampleCpuT>>& samples) {
  holoscan::Graph graph{"so101-disconnect-test"};
  auto source = graph.op<So101StateSourceOp>("source", driver);
  auto sink = graph.op<StateCaptureOp>("sink", samples);
  graph.add_flow(source->state, sink->state);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  auto session = holoscan::run_async(plan);
  session.wait();
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kRequestedCompleted);
}

TEST(So101StateSourceOp, SustainedReadFailureCompletesWithoutEmittingInvalidState) {
  const CalibrationFile calibration{"so101_source_disconnect.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  bus->max_successful_sync_reads = 0;
  auto driver = std::make_shared<so101::FeetechSerialDriver>(std::move(bus));
  driver->initialize(options_for(calibration));
  auto samples = std::make_shared<std::vector<isaac::StateSampleCpuT>>();
  const auto start = std::chrono::steady_clock::now();
  run_source(driver, samples);
  EXPECT_GE(std::chrono::steady_clock::now() - start, 2s);
  EXPECT_TRUE(samples->empty());
}

TEST(So101StateSourceOp, SuccessfulReadResetsTheFailureDeadline) {
  const CalibrationFile calibration{"so101_source_recovery.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  auto* bus_view = bus.get();
  std::optional<std::chrono::steady_clock::time_point> first_read;
  bus->on_sync_read = [&first_read, bus_view] {
    const auto now = std::chrono::steady_clock::now();
    if (!first_read) first_read = now;
    const auto elapsed = now - *first_read;
    const bool unavailable = elapsed < 1100ms || (elapsed >= 1400ms && elapsed < 2600ms);
    bus_view->max_successful_sync_reads = unavailable ? 0 : -1;
  };
  auto driver = std::make_shared<so101::FeetechSerialDriver>(std::move(bus));
  driver->initialize(options_for(calibration));
  auto samples = std::make_shared<std::vector<isaac::StateSampleCpuT>>();
  run_source(driver, samples);
  ASSERT_GE(samples->size(), 2U);
  EXPECT_GT(samples->front().fault_epoch, 0U);
  EXPECT_GT(samples->back().fault_epoch, samples->front().fault_epoch);
  EXPECT_GT(samples->back().source_sequence, samples->front().source_sequence);
}
}  // namespace
}  // namespace isaac::os::ops
