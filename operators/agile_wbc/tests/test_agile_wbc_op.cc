/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "operators/agile_wbc/agile_wbc_op.h"

namespace isaac::os::ops {
namespace {

using namespace std::chrono_literals;
inline constexpr std::size_t kTestSdkJointCount = 43U;

WbcParams make_test_params() {
  WbcParams params;
  params.onnx_path = ISAAC_WBC_TEST_MODEL;

  for (std::size_t index = 0; index < kTestSdkJointCount; ++index) {
    const std::string joint = "joint_" + std::to_string(index);
    params.sdk_joint_order.push_back(joint);
    params.default_joint_positions.emplace(joint, 0.5);
    params.default_kp.emplace(joint, 40.0);
    params.default_kd.emplace(joint, 4.0);
    if (index < 29U) {
      params.policy_joint_order.push_back(joint);
    }
    if (index < 14U) {
      params.action_joint_order.push_back(joint);
    }
  }

  params.velocity_command_lin_x = 0.25;
  params.velocity_command_lin_y = -0.125;
  params.velocity_command_ang_z = 0.5;
  params.history_terms = {
      {.input_name = "base_ang_vel_history",
       .output_name = "base_ang_vel_history_out",
       .dim = 3,
       .length = 5},
      {.input_name = "projected_gravity_history",
       .output_name = "projected_gravity_history_out",
       .dim = 3,
       .length = 5},
      {.input_name = "velocity_commands_history",
       .output_name = "velocity_commands_history_out",
       .dim = 3,
       .length = 5},
      {.input_name = "controlled_joint_pos_history",
       .output_name = "controlled_joint_pos_history_out",
       .dim = 14,
       .length = 5},
      {.input_name = "controlled_joint_vel_history",
       .output_name = "controlled_joint_vel_history_out",
       .dim = 14,
       .length = 5},
      {.input_name = "actions_history",
       .output_name = "actions_history_out",
       .dim = 14,
       .length = 5},
      {.input_name = "last_actions", .output_name = "last_actions_out", .dim = 14},
  };
  return params;
}

struct CommandCapture {
  static void reset() {
    const std::lock_guard lock(mutex);
    count = 0U;
    last.reset();
  }

  static void record(const isaac::JointCommandCpuT& command) {
    {
      const std::lock_guard lock(mutex);
      last = command;
      ++count;
    }
    ready.notify_all();
  }

  static std::optional<isaac::JointCommandCpuT> wait_for(std::size_t expected_count,
                                                         std::chrono::milliseconds timeout = 30s) {
    std::unique_lock lock(mutex);
    if (!ready.wait_for(lock, timeout, [expected_count] { return count >= expected_count; })) {
      return std::nullopt;
    }
    return last;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t count{};
  static inline std::optional<isaac::JointCommandCpuT> last;
};

class StateSource final : public holoscan::Operator<> {
 public:
  explicit StateSource(std::size_t joint_count = kTestSdkJointCount) : joint_count_(joint_count) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(joint_state, "out_joint_state");
    spec.output(imu, "out_imu");
  }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 20ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    isaac::JointStateCpuT joint_state_message{};
    holoscan::schema::ImuT imu_message{};
    joint_state_message.position.resize(joint_count_);
    joint_state_message.velocity.resize(joint_count_);
    for (std::size_t index = 0; index < joint_count_; ++index) {
      joint_state_message.position[index] = static_cast<float>(index + 1U) * 0.01F;
      joint_state_message.velocity[index] = -static_cast<float>(index + 1U) * 0.02F;
    }
    imu_message.orientation = holoscan::schema::Quaternionf{1.0F, 0.0F, 0.0F, 0.0F};
    auto capture_time = context.clock().now();
    if (!capture_time) {
      return holoscan::make_unexpected(std::move(capture_time).error());
    }
    const holoscan::EmitOptions options{.capture_time = *capture_time};
    auto emitted = imu.emit(imu_message, options);
    if (!emitted) {
      return emitted;
    }
    return joint_state.emit(joint_state_message, options);
  }

  holoscan::Output<isaac::JointStateCpuT> joint_state;
  holoscan::Output<holoscan::schema::ImuT> imu;

 private:
  std::size_t joint_count_;
};

class CommandSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(command, "command"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{command});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = command.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    CommandCapture::record(received->data);
    return {};
  }

  holoscan::Input<isaac::JointCommandCpuT> command;
};

class VelocitySource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(velocity_commands, "out_velocity_commands");
  }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    // Slower than the control loop, like the gamepad it stands in for.
    contract.trigger(holoscan::OnClock{.period = 100ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    isaac::TwistCpuT twist{};
    twist.linear = holoscan::schema::Vector3d{0.75, -0.375, 2.0};
    twist.angular = holoscan::schema::Vector3d{3.0, 4.0, -0.25};
    auto capture_time = context.clock().now();
    if (!capture_time) {
      return holoscan::make_unexpected(std::move(capture_time).error());
    }
    return velocity_commands.emit(twist, holoscan::EmitOptions{.capture_time = *capture_time});
  }

  holoscan::Output<isaac::TwistCpuT> velocity_commands;
};

bool has_cuda_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

TEST(AgileWbc, InvokesParameterValidation) {
  AgileWbcOp op;
  EXPECT_THROW(op.start(), std::invalid_argument);
}

TEST(AgileWbc, EmitsExpectedCommand) {
  const WbcParams params = make_test_params();
  const std::unordered_set<std::string> action_joints(params.action_joint_order.begin(),
                                                      params.action_joint_order.end());
  CommandCapture::reset();

  holoscan::Graph graph{"agile-wbc-test"};
  const auto source = graph.op<StateSource>("source");
  auto wbc = graph.op<AgileWbcOp>("agile_wbc");
  const auto sink = graph.op<CommandSink>("sink");
  wbc.set_params(params);
  graph.add_flow(source->joint_state, wbc->joint_state);
  graph.add_flow(source->imu, wbc->imu);
  graph.add_flow(wbc->command, sink->command);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto result = CommandCapture::wait_for(4U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kStopped);
  ASSERT_EQ(result->position.size(), kTestSdkJointCount);
  ASSERT_EQ(result->velocity.size(), kTestSdkJointCount);
  ASSERT_EQ(result->effort.size(), kTestSdkJointCount);
  ASSERT_EQ(result->kp.size(), kTestSdkJointCount);
  ASSERT_EQ(result->kd.size(), kTestSdkJointCount);
  // The stub exposes the three policy velocity inputs as its first three positions.
  EXPECT_FLOAT_EQ(result->position[0], params.velocity_command_lin_x);
  EXPECT_FLOAT_EQ(result->position[1], params.velocity_command_lin_y);
  EXPECT_FLOAT_EQ(result->position[2], params.velocity_command_ang_z);
  for (std::size_t index = 0; index < kTestSdkJointCount; ++index) {
    SCOPED_TRACE(index);
    EXPECT_TRUE(std::isfinite(result->position[index]));
    EXPECT_TRUE(std::isfinite(result->kp[index]));
    EXPECT_TRUE(std::isfinite(result->kd[index]));
    EXPECT_FLOAT_EQ(result->velocity[index], 0.0F);
    EXPECT_FLOAT_EQ(result->effort[index], 0.0F);
    if (!action_joints.contains(params.sdk_joint_order[index])) {
      EXPECT_FLOAT_EQ(result->position[index], 0.5F);
      EXPECT_FLOAT_EQ(result->kp[index], 40.0F);
      EXPECT_FLOAT_EQ(result->kd[index], 4.0F);
    }
  }
}

TEST(AgileWbc, PassesTeleopVelocityAxesToThePolicy) {
  const WbcParams params = make_test_params();
  CommandCapture::reset();

  holoscan::Graph graph{"agile-wbc-velocity-test"};
  const auto source = graph.op<StateSource>("source");
  const auto velocity_source = graph.op<VelocitySource>("velocity_source");
  auto wbc = graph.op<AgileWbcOp>("agile_wbc", /*velocity_from_teleop=*/true);
  const auto sink = graph.op<CommandSink>("sink");
  wbc.set_params(params);
  graph.add_flow(source->joint_state, wbc->joint_state);
  graph.add_flow(source->imu, wbc->imu);
  graph.add_flow(velocity_source->velocity_commands, wbc->velocity_commands);
  graph.add_flow(wbc->command, sink->command);
  // The teleop source runs on its own partition so it cannot stall the control chain.
  graph.partition("teleop").add(velocity_source);
  graph.partition("control").add(source).add(wbc).add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  holoscan::RunSession session = holoscan::run_async(plan);
  // More control ticks than teleop ticks, so this also covers the ticks with no fresh command.
  const auto result = CommandCapture::wait_for(8U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kStopped);
  ASSERT_EQ(result->position.size(), kTestSdkJointCount);
  EXPECT_FLOAT_EQ(result->position[0], 0.75F);
  EXPECT_FLOAT_EQ(result->position[1], -0.375F);
  EXPECT_FLOAT_EQ(result->position[2], -0.25F);
  for (std::size_t index = 0; index < kTestSdkJointCount; ++index) {
    SCOPED_TRACE(index);
    EXPECT_TRUE(std::isfinite(result->position[index]));
  }
}

TEST(AgileWbc, AcceptsConfiguredJointCount) {
  WbcParams params = make_test_params();
  params.sdk_joint_order.pop_back();
  CommandCapture::reset();

  holoscan::Graph graph{"agile-wbc-configured-joint-count-test"};
  const auto source = graph.op<StateSource>("source", params.sdk_joint_order.size());
  auto wbc = graph.op<AgileWbcOp>("agile_wbc");
  const auto sink = graph.op<CommandSink>("sink");
  wbc.set_params(params);
  graph.add_flow(source->joint_state, wbc->joint_state);
  graph.add_flow(source->imu, wbc->imu);
  graph.add_flow(wbc->command, sink->command);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto result = CommandCapture::wait_for(1U, 1s);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kStopped);
  EXPECT_EQ(result->position.size(), params.sdk_joint_order.size());
}

TEST(AgileWbc, RejectsModelInputSizeMismatch) {
  WbcParams params = make_test_params();
  params.policy_joint_order.pop_back();
  CommandCapture::reset();

  holoscan::Graph graph{"agile-wbc-model-input-mismatch-test"};
  const auto source = graph.op<StateSource>("source");
  auto wbc = graph.op<AgileWbcOp>("agile_wbc");
  const auto sink = graph.op<CommandSink>("sink");
  wbc.set_params(params);
  graph.add_flow(source->joint_state, wbc->joint_state);
  graph.add_flow(source->imu, wbc->imu);
  graph.add_flow(wbc->command, sink->command);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  holoscan::RunSession session = holoscan::run_async(plan);
  (void)CommandCapture::wait_for(1U, 1s);
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), holoscan::RunTermination::kFailed);
}

}  // namespace
}  // namespace isaac::os::ops

int main(int argc, char** argv) {
  if (!isaac::os::ops::has_cuda_device()) {
    return ISAAC_CTEST_SKIP_RETURN_CODE;
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
