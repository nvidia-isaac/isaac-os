/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "operators/so101_reach/so101_reach_fused_op.h"

namespace isaac::os::ops {
namespace {

using namespace std::chrono_literals;

constexpr std::uint64_t kTraceId = 17U;

So101ReachParams valid_params() {
  return {
      .onnx_path = SO101_REACH_MODEL_PATH,
  };
}

class StateSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(state, "out_state"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 10ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto capture_time = context.clock().now();
    if (!capture_time) {
      return holoscan::make_unexpected(std::move(capture_time).error());
    }
    auto output = state.allocate();
    if (!output) {
      return holoscan::make_unexpected(std::move(output).error());
    }
    isaac::StateSampleCpuT& message = output->value();
    message.position = {0.0F, 0.0F, 0.0F, 1.57000005F, 0.0F, 0.0F};
    message.velocity = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    message.source_sequence = static_cast<std::uint64_t>(capture_time->timestamp_ns);
    message.fault_epoch = 3U;
    return state.emit(std::move(*output), holoscan::EmitOptions{
                                              .capture_time = *capture_time,
                                              .trace = holoscan::TraceContext::root(kTraceId),
                                          });
  }

  holoscan::Output<isaac::StateSampleCpuT> state;
};

struct CapturedDelta {
  isaac::PolicyDeltaCpuT delta;
  std::int64_t capture_time{};
  std::uint64_t trace_id{};
};

struct DeltaCapture {
  static void reset() {
    const std::lock_guard lock(mutex);
    captured.reset();
  }

  static void record(isaac::PolicyDeltaCpuT delta, std::int64_t capture_time,
                     std::uint64_t trace_id) {
    {
      const std::lock_guard lock(mutex);
      captured = CapturedDelta{
          .delta = std::move(delta),
          .capture_time = capture_time,
          .trace_id = trace_id,
      };
    }
    ready.notify_all();
  }

  static bool wait_for_delta() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 30s, [] { return captured.has_value(); });
  }

  static std::optional<CapturedDelta> take() {
    const std::lock_guard lock(mutex);
    return std::move(captured);
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::optional<CapturedDelta> captured;
};

class DeltaSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(delta, "in_delta"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{delta});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = delta.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    DeltaCapture::record(std::move(received->data), received->metadata.capture_timestamp_ns,
                         received->metadata.trace_id);
    return {};
  }

  holoscan::Input<isaac::PolicyDeltaCpuT> delta;
};

std::optional<CapturedDelta> run_policy(float action_clip) {
  DeltaCapture::reset();
  auto params = valid_params();
  params.action_clip = action_clip;

  holoscan::Graph graph{"so101-reach-fused-op-test"};
  const auto source = graph.op<StateSource>("source");
  auto policy = graph.op<So101ReachFusedOp>("policy");
  const auto sink = graph.op<DeltaSink>("sink");
  policy.set_params(std::move(params));
  graph.add_flow(source->state, policy->state,
                 holoscan::ConnectionOptions{
                     .queue_policy = holoscan::QueuePolicy::kDropOldest,
                     .queue_depth = 1U,
                 });
  graph.add_flow(policy->delta, sink->delta);
  graph.partition("state_source").add(source);
  graph.partition("reach_policy").add(policy);
  graph.partition("delta_sink").add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    return std::nullopt;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool received_delta = DeltaCapture::wait_for_delta();
  session.request_stop();
  session.wait();
  if (!received_delta || session.termination() != holoscan::RunTermination::kStopped) {
    return std::nullopt;
  }
  return DeltaCapture::take();
}

TEST(So101ReachParams, RejectsInvalidPolicyConfiguration) {
  EXPECT_TRUE(validate_so101_reach_params(valid_params()));

  auto params = valid_params();
  params.onnx_path.clear();
  EXPECT_FALSE(validate_so101_reach_params(params));

  params = valid_params();
  params.default_joint_position.pop_back();
  EXPECT_FALSE(validate_so101_reach_params(params));

  params = valid_params();
  params.goal_pos_y_range[0U] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(validate_so101_reach_params(params));

  params = valid_params();
  params.goal_grid_ny = 0;
  EXPECT_FALSE(validate_so101_reach_params(params));

  params = valid_params();
  params.goal_hold_time = 0.0;
  EXPECT_FALSE(validate_so101_reach_params(params));

  params = valid_params();
  params.action_clip = 0.0;
  EXPECT_FALSE(validate_so101_reach_params(params));

  params = valid_params();
  params.inference_period = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(validate_so101_reach_params(params));
}

TEST(So101ReachGoals, UsesExpectedNineGoalGridAndOrder) {
  const auto goals = build_so101_reach_goal_grid(valid_params());

  ASSERT_EQ(goals.size(), 9U);
  EXPECT_EQ(goals[0U], (std::array<float, 3U>{0.2F, -0.112F, 0.136F}));
  EXPECT_EQ(goals[1U], (std::array<float, 3U>{0.2F, 0.0F, 0.136F}));
  EXPECT_EQ(goals[2U], (std::array<float, 3U>{0.2F, 0.112F, 0.136F}));
  EXPECT_EQ(goals[3U], (std::array<float, 3U>{0.2F, -0.112F, 0.2F}));
  EXPECT_EQ(goals[4U], (std::array<float, 3U>{0.2F, 0.0F, 0.2F}));
  EXPECT_EQ(goals[8U], (std::array<float, 3U>{0.2F, 0.112F, 0.264F}));
}

TEST(So101ReachObservation, PreservesTheTrainedNonActuatedJointContract) {
  const So101ReachParams params = valid_params();
  const std::array<float, 6U> joint_position{0.2F, -0.3F, 0.4F, 1.2F, 0.7F, -0.175F};
  const std::array<float, 3U> goal_position{0.2F, -0.112F, 0.136F};
  const std::array<float, 4U> last_action{0.1F, 0.2F, 0.3F, 0.4F};

  const auto observation =
      build_so101_reach_observation(joint_position, params, goal_position, last_action);

  EXPECT_FLOAT_EQ(observation[0U], 0.2F);
  EXPECT_FLOAT_EQ(observation[1U], -0.3F);
  EXPECT_FLOAT_EQ(observation[2U], 0.4F);
  EXPECT_FLOAT_EQ(observation[3U], 1.2F - 1.5700000524520874F);
  EXPECT_FLOAT_EQ(observation[4U], 0.0F);
  EXPECT_FLOAT_EQ(observation[5U], 0.0F);
  EXPECT_FLOAT_EQ(observation[9U], goal_position[0U]);
  EXPECT_FLOAT_EQ(observation[10U], goal_position[1U]);
  EXPECT_FLOAT_EQ(observation[11U], goal_position[2U]);
  EXPECT_EQ(std::vector<float>(observation.begin() + 16U, observation.end()),
            std::vector<float>(last_action.begin(), last_action.end()));
}

TEST(So101ReachAction, RejectsTheEntireActionWhenAnyElementIsInvalid) {
  constexpr float kClip = 0.5F;
  const std::array<float, 4U> invalid_action{std::numeric_limits<float>::quiet_NaN(), 0.8F, -0.7F,
                                             0.3F};
  EXPECT_EQ(sanitize_so101_reach_action(invalid_action, kClip),
            (std::array<float, 4U>{0.0F, 0.0F, 0.0F, 0.0F}));

  const std::array<float, 4U> finite_action{0.1F, 0.8F, -0.7F, 0.3F};
  EXPECT_EQ(sanitize_so101_reach_action(finite_action, kClip),
            (std::array<float, 4U>{0.1F, 0.5F, -0.5F, 0.3F}));
}

TEST(So101ReachFusedOp, RunsPolicyClipsActionsAndPreservesLineage) {
  const auto unclipped = run_policy(1.0F);
  ASSERT_TRUE(unclipped);
  ASSERT_EQ(unclipped->delta.action.size(), 4U);
  const auto maximum_action = std::max_element(
      unclipped->delta.action.begin(), unclipped->delta.action.end(),
      [](float first, float second) { return std::abs(first) < std::abs(second); });
  ASSERT_NE(maximum_action, unclipped->delta.action.end());
  const float maximum_magnitude = std::abs(*maximum_action);
  ASSERT_GT(maximum_magnitude, 0.0F);

  const float action_clip = maximum_magnitude / 2.0F;
  const auto clipped = run_policy(action_clip);
  ASSERT_TRUE(clipped);
  ASSERT_EQ(clipped->delta.action.size(), unclipped->delta.action.size());
  for (std::size_t index = 0; index < clipped->delta.action.size(); ++index) {
    EXPECT_FLOAT_EQ(clipped->delta.action[index],
                    std::clamp(unclipped->delta.action[index], -action_clip, action_clip));
  }
  EXPECT_EQ(clipped->delta.source_sequence, static_cast<std::uint64_t>(clipped->capture_time));
  EXPECT_EQ(clipped->delta.fault_epoch, 3U);
  EXPECT_GT(clipped->capture_time, 0);
  EXPECT_EQ(clipped->trace_id, kTraceId);
}

}  // namespace
}  // namespace isaac::os::ops
