/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>

#include <flatbuffers/flatbuffer_builder.h>
#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/joint_command_chunk_cpu/joint_command_chunk_cpu.h"
#include "joint_command_chunk_cpu_schema_traits.h"

namespace isaac::os {
namespace {

using namespace std::chrono_literals;
inline constexpr std::uint32_t kJointCount = 43U;
inline constexpr std::uint32_t kActionCount = 40U;
inline constexpr std::size_t kValueCount = kJointCount * kActionCount;

class Source final : public holoscan::Operator<> {
 public:
  explicit Source(isaac::JointCommandChunkCpuT message) : message_(std::move(message)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 1ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    // This test transports one chunk. Repeated clock ticks can fill the queue before the
    // sink runs when a delayed scheduler catches up, testing backpressure instead.
    if (sent_) return {};
    sent_ = true;
    return output.emit(message_);
  }

  holoscan::Output<isaac::JointCommandChunkCpuT> output;

 private:
  isaac::JointCommandChunkCpuT message_;
  bool sent_{};
};

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
    {
      const std::lock_guard lock(mutex_);
      value_ = sample->data;
    }
    ready_.notify_all();
    return {};
  }

  static void reset() {
    const std::lock_guard lock(mutex_);
    value_.reset();
  }

  static std::optional<isaac::JointCommandChunkCpuT> wait() {
    std::unique_lock lock(mutex_);
    if (!ready_.wait_for(lock, 5s, [] { return value_.has_value(); })) {
      return std::nullopt;
    }
    return value_;
  }

  holoscan::Input<isaac::JointCommandChunkCpuT> input;

 private:
  static inline std::mutex mutex_;
  static inline std::condition_variable ready_;
  static inline std::optional<isaac::JointCommandChunkCpuT> value_;
};

isaac::JointCommandChunkCpuT make_chunk() {
  isaac::JointCommandChunkCpuT message;
  message.joint_count = kJointCount;
  message.action_count = kActionCount;
  message.position.resize(kValueCount);
  message.effort.resize(kValueCount);
  message.kp.resize(kJointCount);
  message.kd.resize(kJointCount);
  message.position[0] = 1.0F;
  message.position[42U * kActionCount + 15U] = 2.0F;
  message.effort[2U * kActionCount + 3U] = -3.0F;
  message.kp[2U] = 40.0F;
  message.kd[2U] = 1.5F;
  message.image_capture_time_ns = 123456789LL;
  return message;
}

TEST(JointCommandChunkCpu, TransportsJointMajorActionChunk) {
  Sink::reset();
  holoscan::Graph graph{"joint-command-chunk-cpu-test"};
  const auto source = graph.op<Source>("source", make_chunk());
  const auto sink = graph.op<Sink>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  holoscan::RunSession session = holoscan::run_async(plan);
  const auto received = Sink::wait();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kStopped);
  ASSERT_TRUE(validate_joint_command_chunk(*received));
  EXPECT_FLOAT_EQ(received->position[0], 1.0F);
  EXPECT_FLOAT_EQ(received->position[42U * kActionCount + 15U], 2.0F);
  EXPECT_FLOAT_EQ(received->effort[2U * kActionCount + 3U], -3.0F);
  EXPECT_FLOAT_EQ(received->kp[2U], 40.0F);
  EXPECT_FLOAT_EQ(received->kd[2U], 1.5F);
  EXPECT_EQ(received->image_capture_time_ns, 123456789LL);
}

TEST(JointCommandChunkCpu, RejectsInvalidDimensions) {
  isaac::JointCommandChunkCpuT message = make_chunk();
  message.position.pop_back();
  EXPECT_FALSE(validate_joint_command_chunk(message));

  message = make_chunk();
  message.kp.pop_back();
  EXPECT_FALSE(validate_joint_command_chunk(message));

  message = make_chunk();
  message.kd.pop_back();
  EXPECT_FALSE(validate_joint_command_chunk(message));

  message = make_chunk();
  message.joint_count = 0U;
  EXPECT_FALSE(validate_joint_command_chunk(message));

  message = make_chunk();
  message.action_count = 0U;
  EXPECT_FALSE(validate_joint_command_chunk(message));

  message = make_chunk();
  message.joint_count = std::numeric_limits<std::uint32_t>::max();
  message.action_count = std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(validate_joint_command_chunk(message));
}

TEST(JointCommandChunkCpu, RepresentativeChunkFitsDeclaredBound) {
  const isaac::JointCommandChunkCpuT message = make_chunk();
  flatbuffers::FlatBufferBuilder builder;
  const auto root = isaac::JointCommandChunkCpu::Pack(builder, &message);
  builder.Finish(root);

  EXPECT_LE(
      builder.GetSize(),
      holoscan::schema::FlatBufferSchemaBinding<isaac::JointCommandChunkCpu>::max_serialized_size);
}

}  // namespace
}  // namespace isaac::os
