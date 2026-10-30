/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "applications/so101_reach/sim/sim_adapters.h"

namespace isaac::os::ops {
namespace {

TEST(So101ReachSim, StatePreservesJointValuesAndSequence) {
  isaac::JointStateCpuT input;
  input.position = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};
  input.velocity.assign(6U, 0.25F);
  const auto sample = make_so101_sim_state(input, 42U);
  EXPECT_EQ(sample.position, input.position);
  EXPECT_EQ(sample.velocity, input.velocity);
  EXPECT_EQ(sample.source_sequence, 42U);
  EXPECT_EQ(sample.fault_epoch, 0U);
}

TEST(So101ReachSim, StateRejectsWrongShapeAndNonfiniteValues) {
  isaac::JointStateCpuT input;
  input.position.assign(6U, 0.0F);
  input.velocity.assign(6U, 0.0F);
  input.position.pop_back();
  EXPECT_THROW(make_so101_sim_state(input, 1U), std::invalid_argument);
  input.position.assign(6U, 0.0F);
  input.velocity.pop_back();
  EXPECT_THROW(make_so101_sim_state(input, 1U), std::invalid_argument);
  input.velocity.assign(6U, 0.0F);
  input.position[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(make_so101_sim_state(input, 1U), std::invalid_argument);
  input.position[0] = 0.0F;
  input.velocity[0] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(make_so101_sim_state(input, 1U), std::invalid_argument);
}

TEST(So101ReachSim, StateRejectsUnexpectedJointOrder) {
  isaac::JointStateCpuT input;
  input.position.assign(6U, 0.0F);
  input.velocity.assign(6U, 0.0F);
  for (const auto name : so101::kSo101JointNames) input.name.emplace_back(name);
  EXPECT_NO_THROW(make_so101_sim_state(input, 1U));
  std::swap(input.name[0], input.name[1]);
  EXPECT_THROW(make_so101_sim_state(input, 1U), std::invalid_argument);
}

class SimStateSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(state, "state"); }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{20}});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    isaac::JointStateCpuT sample;
    sample.position.assign(6U, 0.2F);
    sample.velocity.assign(6U, 0.0F);
    auto now = context.clock().now();
    if (!now) return holoscan::make_unexpected(now.error());
    return state.emit(sample, {.capture_time = *now});
  }
  holoscan::Output<isaac::JointStateCpuT> state;
};

class SimStateProbeOp final : public holoscan::Operator<> {
 public:
  explicit SimStateProbeOp(std::shared_ptr<std::size_t> count) : count_(std::move(count)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(state, "state"); }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{state});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto sample = state.receive();
    if (!sample) return holoscan::make_unexpected(sample.error());
    ++*count_;
    EXPECT_EQ(sample->data.source_sequence, *count_);
    EXPECT_EQ(sample->data.position, std::vector<float>(6U, 0.2F));
    EXPECT_GT(sample->metadata.capture_time().timestamp_ns, 0);
    if (*count_ >= 3U) context.request_completion();
    return {};
  }
  holoscan::Input<isaac::StateSampleCpuT> state;

 private:
  std::shared_ptr<std::size_t> count_;
};

TEST(So101ReachSim, StateAdapterPublishesIncreasingSequences) {
  auto count = std::make_shared<std::size_t>(0U);
  holoscan::Graph graph{"sim-state-adapter"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto source = graph.op<SimStateSourceOp>("source");
  const auto adapter = graph.op<So101SimStateOp>("adapter", 0U);
  const auto probe = graph.op<SimStateProbeOp>("probe", count);
  graph.add_flow(source->state, adapter->joints);
  graph.add_flow(adapter->state, probe->state);
  graph.partition("source").add(source);
  graph.partition("adapter").add(adapter);
  graph.partition("probe").add(probe);
  const auto plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = holoscan::run_async(plan);
  session.wait();
  ASSERT_EQ(session.termination(), holoscan::RunTermination::kRequestedCompleted);
  EXPECT_EQ(*count, 3U);
}

}  // namespace
}  // namespace isaac::os::ops
