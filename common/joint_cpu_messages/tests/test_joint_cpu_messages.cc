/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
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

#include "joint_command_cpu_schema_traits.h"
#include "joint_state_cpu_schema_traits.h"

namespace isaac::os {
namespace {

using namespace std::chrono_literals;
inline constexpr std::size_t kTestJointCount = 3U;

template <typename Message>
struct Observation {
  static void reset() {
    const std::lock_guard lock(mutex);
    value.reset();
  }

  static void record(const Message& message) {
    {
      const std::lock_guard lock(mutex);
      value = message;
    }
    ready.notify_all();
  }

  static std::optional<Message> wait() {
    std::unique_lock lock(mutex);
    if (!ready.wait_for(lock, 5s, [] { return value.has_value(); })) {
      return std::nullopt;
    }
    return value;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::optional<Message> value;
};

template <typename Message>
class Source final : public holoscan::Operator<> {
 public:
  explicit Source(Message message) : message_(std::move(message)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 1ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return output.emit(message_);
  }

  holoscan::Output<Message> output;

 private:
  Message message_;
};

template <typename Message>
class Sink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{input});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    Observation<Message>::record(sample->data);
    return {};
  }

  holoscan::Input<Message> input;
};

template <typename Message>
std::optional<Message> transport_one(Message message) {
  Observation<Message>::reset();

  holoscan::Graph graph{"joint-cpu-message-test"};
  const auto source = graph.op<Source<Message>>("source", std::move(message));
  const auto sink = graph.op<Sink<Message>>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    return std::nullopt;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  auto value = Observation<Message>::wait();
  session.request_stop();
  session.wait();
  if (session.termination() != holoscan::RunTermination::kStopped) {
    return std::nullopt;
  }
  return value;
}

TEST(JointCpuMessages, TransportsJointState) {
  isaac::JointStateCpuT message{};
  message.position.resize(kTestJointCount);
  message.velocity.resize(kTestJointCount);
  message.position.front() = 1.0F;
  message.velocity.back() = -2.0F;
  message.name = {"joint_0", "joint_1"};
  message.effort.resize(kTestJointCount);
  message.effort.front() = 0.5F;

  const auto value = transport_one(std::move(message));

  ASSERT_TRUE(value.has_value());
  ASSERT_EQ(value->position.size(), kTestJointCount);
  ASSERT_EQ(value->velocity.size(), kTestJointCount);
  EXPECT_FLOAT_EQ(value->position.front(), 1.0F);
  EXPECT_FLOAT_EQ(value->velocity.back(), -2.0F);
  ASSERT_EQ(value->name.size(), 2U);
  EXPECT_EQ(value->name[0], "joint_0");
  EXPECT_EQ(value->name[1], "joint_1");
  ASSERT_EQ(value->effort.size(), kTestJointCount);
  EXPECT_FLOAT_EQ(value->effort.front(), 0.5F);
}

TEST(JointCpuMessages, TransportsCommand) {
  isaac::JointCommandCpuT message{};
  message.position.resize(kTestJointCount);
  message.velocity.resize(kTestJointCount);
  message.effort.resize(kTestJointCount);
  message.kp.resize(kTestJointCount);
  message.kd.resize(kTestJointCount);
  message.position.front() = 1.0F;
  message.velocity.front() = 2.0F;
  message.effort.front() = 3.0F;
  message.kp.front() = 4.0F;
  message.kd.front() = 5.0F;

  const auto value = transport_one(std::move(message));

  ASSERT_TRUE(value.has_value());
  ASSERT_EQ(value->position.size(), kTestJointCount);
  ASSERT_EQ(value->velocity.size(), kTestJointCount);
  ASSERT_EQ(value->effort.size(), kTestJointCount);
  ASSERT_EQ(value->kp.size(), kTestJointCount);
  ASSERT_EQ(value->kd.size(), kTestJointCount);
  EXPECT_FLOAT_EQ(value->position.front(), 1.0F);
  EXPECT_FLOAT_EQ(value->velocity.front(), 2.0F);
  EXPECT_FLOAT_EQ(value->effort.front(), 3.0F);
  EXPECT_FLOAT_EQ(value->kp.front(), 4.0F);
  EXPECT_FLOAT_EQ(value->kd.front(), 5.0F);
}

}  // namespace
}  // namespace isaac::os
