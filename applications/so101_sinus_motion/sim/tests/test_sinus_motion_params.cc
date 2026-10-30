/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "applications/so101_sinus_motion/sim/sinus_motion_op.h"

namespace isaac::os::apps::so101_sinus_motion_sim {
namespace {

struct CommandField {
  const char* name;
  std::vector<double> SinusMotionParams::* member;
};

class SimCommandFieldTest : public testing::TestWithParam<CommandField> {};

TEST_P(SimCommandFieldTest, RejectsInvalidFloatPayloadValues) {
  const double max_float = std::numeric_limits<float>::max();
  const double outside_float = std::nextafter(max_float, std::numeric_limits<double>::infinity());
  for (const double value :
       {1e40, -1e40, outside_float, -outside_float, std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), 1e-100,
        -1e-100}) {
    SCOPED_TRACE(value);
    SinusMotionParams params;
    params.num_joints = 2;
    // Exercise a later element, so checking only the first entry cannot pass.
    params.*GetParam().member = {0.0, value};
    try {
      const SinusMotionOp op{params};
      FAIL() << "Invalid command value was accepted.";
    } catch (const holoscan::RuntimeError& error) {
      const auto entry = std::format("{}[1] = {:.{}g}", GetParam().name, value,
                                     std::numeric_limits<double>::max_digits10);
      EXPECT_NE(std::string{error.what()}.find(entry), std::string::npos);
    }
  }
}

TEST_P(SimCommandFieldTest, AcceptsRoundedDecimalsAndFloatBoundaries) {
  for (const double value :
       {0.0, -0.0, 0.05, -0.05, 0.1, 30.1, static_cast<double>(std::numeric_limits<float>::max()),
        static_cast<double>(std::numeric_limits<float>::lowest()),
        static_cast<double>(std::numeric_limits<float>::denorm_min()),
        -static_cast<double>(std::numeric_limits<float>::denorm_min())}) {
    SCOPED_TRACE(value);
    SinusMotionParams params;
    params.*GetParam().member = {value};
    EXPECT_NO_THROW((void)SinusMotionOp{params});
  }
}

// Supply two simulator timestamps one second apart, independent of wall-clock scheduling.
class StateSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(state, "state"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    const auto tick = ticks_++;
    if (tick < 2) {
      return state.emit(isaac::JointStateCpuT{},
                        holoscan::EmitOptions{.capture_time = context.clock().at_ns(
                                                  tick * std::int64_t{1000000000})});
    }
    // Bound the test even if an expected command is missing.
    if (tick == 1000) context.request_completion();
    return {};
  }

  holoscan::Output<isaac::JointStateCpuT> state;

 private:
  std::int64_t ticks_{};
};

class CommandSink final : public holoscan::Operator<> {
 public:
  explicit CommandSink(std::shared_ptr<std::vector<isaac::JointCommandCpuT>> commands)
      : commands_(std::move(commands)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(command, "command"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{command});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto received = command.receive();
    if (!received) return holoscan::make_unexpected(std::move(received).error());
    commands_->push_back(received->data);
    if (commands_->size() == 2U) context.request_completion();
    return {};
  }

  holoscan::Input<isaac::JointCommandCpuT> command;

 private:
  std::shared_ptr<std::vector<isaac::JointCommandCpuT>> commands_;
};

void check_computed_position(double center, double amplitude, double frequency, bool expect_failure,
                             float expected_position = 0.0F) {
  const auto commands = std::make_shared<std::vector<isaac::JointCommandCpuT>>();
  holoscan::Graph graph{"sim-position-validation"};
  const auto source = graph.op<StateSource>("source");
  const auto sinus =
      graph.op<SinusMotionOp>("sinus", SinusMotionParams{.sin_frequency = frequency,
                                                         .sin_amplitude = amplitude,
                                                         .num_joints = 2,
                                                         .default_position = {0.0, center},
                                                         .kp = {30.1},
                                                         .kd = {0.1}});
  const auto sink = graph.op<CommandSink>("sink", commands);
  graph.add_flow(source->state, sinus->joint_state_in);
  graph.add_flow(sinus->joint_command_out, sink->command);
  graph.partition("source").add(source);
  graph.partition("sinus").add(sinus);
  graph.partition("sink").add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  auto session = holoscan::run_async(plan);
  session.wait();
  if (expect_failure) {
    EXPECT_EQ(session.termination(), holoscan::RunTermination::kFailed);
    EXPECT_LT(commands->size(), 2U);
    bool found_position_error = false;
    for (const auto& diagnostic : session.diagnostics()) {
      if (diagnostic.detail && diagnostic.detail->error.message.find(
                                   "Invalid value in position[") != std::string::npos) {
        found_position_error = true;
      }
    }
    EXPECT_TRUE(found_position_error);
  } else {
    EXPECT_EQ(session.termination(), holoscan::RunTermination::kRequestedCompleted);
    ASSERT_EQ(commands->size(), 2U);
    EXPECT_EQ(commands->back().position.at(1), expected_position);
    EXPECT_EQ(commands->back().kp, (std::vector<float>{30.1F, 30.1F}));
    EXPECT_EQ(commands->back().kd, (std::vector<float>{0.1F, 0.1F}));
  }
}

TEST(SimComputedPosition, RejectsOverflowOfSumWithValidFloatOperands) {
  const double value = 0.75 * std::numeric_limits<float>::max();
  check_computed_position(value, value, 0.25, true);
  check_computed_position(-value, value, 0.75, true);
}

TEST(SimComputedPosition, RejectsAmplitudeOverflowAndUnderflow) {
  check_computed_position(0.0, std::numeric_limits<double>::max(), 0.25, true);
  check_computed_position(0.0, 1e-100, 0.25, true);
}

TEST(SimComputedPosition, RejectsNonfiniteAmplitudeAndPhase) {
  check_computed_position(0.0, std::numeric_limits<double>::infinity(), 0.25, true);
  check_computed_position(0.0, std::numeric_limits<double>::quiet_NaN(), 0.25, true);
  check_computed_position(0.0, 0.2, std::numeric_limits<double>::max(), true);
  check_computed_position(0.0, 0.2, std::numeric_limits<double>::infinity(), true);
  check_computed_position(0.0, 0.2, std::numeric_limits<double>::quiet_NaN(), true);
}

TEST(SimComputedPosition, AcceptsFloatBoundaries) {
  const float max_float = std::numeric_limits<float>::max();
  check_computed_position(0.0, max_float, 0.25, false, max_float);
  check_computed_position(0.0, max_float, 0.75, false, -max_float);
  const float min_float = std::numeric_limits<float>::denorm_min();
  check_computed_position(0.0, min_float, 0.25, false, min_float);
}

TEST(SimComputedPosition, AcceptsDecimalCenterWithZeroAmplitudeOrFrequency) {
  check_computed_position(0.05, 0.0, 0.25, false, 0.05F);
  check_computed_position(0.05, 0.2, 0.0, false, 0.05F);
}

INSTANTIATE_TEST_SUITE_P(
    CommandFields, SimCommandFieldTest,
    testing::Values(CommandField{"default_position", &SinusMotionParams::default_position},
                    CommandField{"kp", &SinusMotionParams::kp},
                    CommandField{"kd", &SinusMotionParams::kd}),
    [](const testing::TestParamInfo<CommandField>& info) { return info.param.name; });

}  // namespace
}  // namespace isaac::os::apps::so101_sinus_motion_sim
