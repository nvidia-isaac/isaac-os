/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
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
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "operators/sequential_action_handler/sequential_action_handler_op.h"

namespace isaac::os::ops {
namespace {

using namespace std::chrono_literals;

isaac::JointCommandChunkCpuT make_chunk(float base) {
  constexpr std::uint32_t kJointCount = 2U;
  constexpr std::uint32_t kActionCount = 3U;
  isaac::JointCommandChunkCpuT chunk;
  chunk.joint_count = kJointCount;
  chunk.action_count = kActionCount;
  chunk.position = {base + 0.0F,  base + 1.0F,  base + 2.0F,
                    base + 10.0F, base + 11.0F, base + 12.0F};
  chunk.effort = {base + 20.0F, base + 21.0F, base + 22.0F,
                  base + 30.0F, base + 31.0F, base + 32.0F};
  chunk.kp = {40.0F, 41.0F};
  chunk.kd = {1.0F, 1.5F};
  return chunk;
}

class ChunkSource final : public holoscan::Operator<> {
 public:
  explicit ChunkSource(std::vector<std::optional<isaac::JointCommandChunkCpuT>> chunks)
      : chunks_(std::move(chunks)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 10ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    if (index_ >= chunks_.size()) {
      return {};
    }
    auto& chunk = chunks_[index_++];
    if (!chunk) {
      return {};
    }
    auto capture_time = context.clock().now();
    if (!capture_time) {
      return holoscan::make_unexpected(std::move(capture_time).error());
    }
    return output.emit(*chunk, holoscan::EmitOptions{.capture_time = *capture_time});
  }

  holoscan::Output<isaac::JointCommandChunkCpuT> output;

 private:
  std::vector<std::optional<isaac::JointCommandChunkCpuT>> chunks_;
  std::size_t index_{};
};

struct CommandCapture {
  static void reset() {
    const std::lock_guard lock(mutex);
    commands.clear();
  }

  static void record(isaac::JointCommandCpuT command) {
    {
      const std::lock_guard lock(mutex);
      commands.push_back(std::move(command));
    }
    ready.notify_all();
  }

  static std::vector<isaac::JointCommandCpuT> wait_for(std::size_t count) {
    std::unique_lock lock(mutex);
    if (!ready.wait_for(lock, 5s, [count] { return commands.size() >= count; })) {
      return {};
    }
    return {commands.begin(), commands.begin() + static_cast<std::ptrdiff_t>(count)};
  }

  static std::size_t size() {
    const std::lock_guard lock(mutex);
    return commands.size();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<isaac::JointCommandCpuT> commands;
};

class CommandSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{input});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = input.receive_data();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    CommandCapture::record(std::move(*received));
    return {};
  }

  holoscan::Input<isaac::JointCommandCpuT> input;
};

struct RunningGraph {
  holoscan::Graph graph{"sequential-action-handler-test"};
  std::unique_ptr<holoscan::RunSession> session;
};

std::unique_ptr<RunningGraph> run_graph(
    std::vector<std::optional<isaac::JointCommandChunkCpuT>> chunks) {
  CommandCapture::reset();
  auto running = std::make_unique<RunningGraph>();
  const auto source = running->graph.op<ChunkSource>("source", std::move(chunks));
  auto handler = running->graph.op<SequentialActionHandlerOp>("handler");
  const auto sink = running->graph.op<CommandSink>("sink");
  handler.set_params(SequentialActionHandlerParams{.dt = 0.01});
  running->graph.add_flow(source->output, handler->action_chunk);
  running->graph.add_flow(handler->command, sink->input);
  running->graph.partition("source").add(source);
  running->graph.partition("handler").add(handler).add(sink);
  running->graph.set_default_clock(running->graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(running->graph);
  EXPECT_TRUE(plan.ok()) << plan.json();
  if (plan.ok()) {
    running->session = std::make_unique<holoscan::RunSession>(holoscan::run_async(plan));
  }
  return running;
}

void stop(RunningGraph& running) {
  ASSERT_NE(running.session, nullptr);
  running.session->request_stop();
  running.session->wait();
  EXPECT_EQ(running.session->termination(), holoscan::RunTermination::kStopped);
}

TEST(SequentialActionHandler, EmitsNothingBeforeAValidChunk) {
  auto invalid = make_chunk(0.0F);
  invalid.kp.pop_back();
  auto running = run_graph({std::nullopt, std::move(invalid)});
  std::this_thread::sleep_for(10ms);
  stop(*running);
  EXPECT_EQ(CommandCapture::size(), 0U);
}

TEST(SequentialActionHandler, PlaysActionsAndRepeatsTheLastAction) {
  auto running = run_graph({make_chunk(100.0F)});
  const auto commands = CommandCapture::wait_for(5U);
  stop(*running);

  ASSERT_EQ(commands.size(), 5U);
  EXPECT_FLOAT_EQ(commands.front().position[0], 100.0F);
  EXPECT_FLOAT_EQ(commands.back().position[0], 102.0F);
  for (std::size_t index = 0; index < commands.size(); ++index) {
    ASSERT_EQ(commands[index].position.size(), 2U);
    if (index > 0U) {
      EXPECT_GE(commands[index].position[0], commands[index - 1U].position[0]);
    }
    EXPECT_FLOAT_EQ(commands[index].position[1], commands[index].position[0] + 10.0F);
    EXPECT_FLOAT_EQ(commands[index].effort[0], commands[index].position[0] + 20.0F);
    EXPECT_EQ(commands[index].kp, (std::vector<float>{40.0F, 41.0F}));
    EXPECT_EQ(commands[index].kd, (std::vector<float>{1.0F, 1.5F}));
    EXPECT_EQ(commands[index].velocity, (std::vector<float>{0.0F, 0.0F}));
  }
}

TEST(SequentialActionHandler, StartsAtTheFirstActionOfAReplacementChunk) {
  auto running = run_graph({make_chunk(100.0F), std::nullopt, make_chunk(200.0F)});
  const auto commands = CommandCapture::wait_for(5U);
  stop(*running);

  const auto replacement = std::find_if(commands.begin(), commands.end(), [](const auto& command) {
    return !command.position.empty() && command.position[0] >= 200.0F;
  });
  ASSERT_NE(replacement, commands.end());
  EXPECT_FLOAT_EQ(replacement->position[0], 200.0F);
}

}  // namespace
}  // namespace isaac::os::ops
