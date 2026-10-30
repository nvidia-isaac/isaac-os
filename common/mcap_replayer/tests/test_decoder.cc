/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/run.hpp"
#include "holoscan/core/temporal_contract.hpp"
#include "holoscan/logger/logger.hpp"
#include "holoscan/schema/schema_package.hpp"
#include "holoscan/time/realtime_clock.hpp"

#include "common/mcap_replayer/decoder.h"
#include "common/mcap_replayer/message.h"
#include "common/mcap_replayer/typed_port.h"
#include "pose_cpu_schema_traits.h"

namespace isaac::os::mcap_replayer {
namespace {

using namespace std::chrono_literals;

isaac::PoseCpuT make_pose(double x) {
  isaac::PoseCpuT pose{};
  pose.position = holoscan::schema::Vector3d(x, 0.0, 0.0);
  pose.orientation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);
  return pose;
}

// Encodes value the same way a recording would have -- through the type's own generated codec,
// not through anything this library owns -- so the decoder under test is exercised against
// realistic bytes.
std::vector<std::byte> encode(const isaac::PoseCpuT& value) {
  const auto& descriptor = holoscan::schema_package<isaac::PoseCpuT>();
  std::vector<std::byte> buffer(descriptor.max_serialized_size());
  auto encoded = descriptor.encode(value, buffer);
  EXPECT_TRUE(encoded.has_value());
  buffer.resize(*encoded);
  return buffer;
}

TEST(Decoder, BuildsAMessageCarryingTheValueAndTheLogTime) {
  const MessageBuilder build = make_message_builder<isaac::PoseCpuT>();

  const std::vector<std::byte> payload = encode(make_pose(4.0));
  constexpr std::uint64_t kLogTimeNs = 123456789ULL;

  auto message = build(payload, {}, kLogTimeNs);

  ASSERT_TRUE(message.has_value()) << message.error();
  ASSERT_NE(*message, nullptr);
  EXPECT_EQ((*message)->log_time_ns, kLogTimeNs);
  EXPECT_NE(dynamic_cast<FlatbufferMessage<isaac::PoseCpuT>*>(message->get()), nullptr);
}

TEST(Decoder, EmittingBeforeThePublisherIsAttachedReportsInsteadOfCrashing) {
  TypedPort<isaac::PoseCpuT> port;
  const MessageBuilder build = make_message_builder<isaac::PoseCpuT>();

  auto message = build(encode(make_pose(1.0)), {}, 0);
  ASSERT_TRUE(message.has_value()) << message.error();

  // Building can run before start() attaches the publisher; publishing then must fail cleanly.
  const auto emitted = port.emit(**message);
  ASSERT_FALSE(emitted.has_value());
  EXPECT_EQ(emitted.error().code, holoscan::ErrorCode::kNotReady);
}

TEST(Decoder, GarbageBytesYieldNoEntryAndSayWhy) {
  const MessageBuilder build = make_message_builder<isaac::PoseCpuT>();

  const std::vector<std::byte> garbage(16, std::byte{0xFF});

  const auto message = build(garbage, {}, 0);

  ASSERT_FALSE(message.has_value());
  // The reason is returned rather than logged, so the caller can name the topic it belongs to
  // and limit how often a stream of failures says the same thing.
  EXPECT_FALSE(message.error().empty());
}

// Static, not a member of any operator: an OperatorHandle only ever exposes wiring access to
// its operator's ports (see OperatorPortProxy above), not the running instance itself, so the
// test thread has no other way to read back what a sink received.
struct Observation {
  static void reset() {
    const std::lock_guard lock(mutex);
    value.reset();
  }

  static void record(const isaac::PoseCpuT& message) {
    {
      const std::lock_guard lock(mutex);
      value = message;
    }
    ready.notify_all();
  }

  static std::optional<isaac::PoseCpuT> wait() {
    std::unique_lock lock(mutex);
    if (!ready.wait_for(lock, 5s, [] { return value.has_value(); })) {
      return std::nullopt;
    }
    return value;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::optional<isaac::PoseCpuT> value;
};

// Records every isaac::PoseCpuT that reaches its input into Observation, for the roundtrip test
// below to inspect off the graph's own thread.
class RecordingSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in_pose, "in_pose"); }

  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{in_pose});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in_pose.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    Observation::record(sample->data);
    return {};
  }

  holoscan::Input<isaac::PoseCpuT> in_pose;
};

class SinkOnlyGraph final : public holoscan::Graph {
 public:
  SinkOnlyGraph() : Graph("decoder_roundtrip") {}

  TypedPort<isaac::PoseCpuT> topic;

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    topic.import_into(*this, "out_pose");
    auto sink = op<RecordingSink>("sink");
    add_flow(topic.output(), sink->in_pose);
  }
};

TEST(Decoder, EmittingADecodedEntryReachesThePort) {
  const isaac::PoseCpuT sent = make_pose(7.0);
  const std::vector<std::byte> payload = encode(sent);
  Observation::reset();

  SinkOnlyGraph graph;
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime.has_value());
  graph.topic.attach(*runtime, plan.endpoint_catalog(), "out_pose");

  holoscan::RunSession session = holoscan::run_async(plan);

  // What the real pacing thread does when a message becomes due: build, then emit immediately --
  // no operator involved, since publishing no longer happens from inside compute(). run_async()
  // only starts the graph; it does not wait for the subscriber side of this topic to be ready to
  // receive, so the first publish attempt can outrace that startup and come back kNotReady. A
  // real pacing run never observes this because buffering and pacing delay its own first emit far
  // past graph startup -- this builds and retries fresh (an already-emitted message's value has
  // been moved from and cannot be resent) until the same race clears.
  const MessageBuilder build = make_message_builder<isaac::PoseCpuT>();
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  bool emitted = false;
  holoscan::Error last_error;
  while (!emitted && std::chrono::steady_clock::now() < deadline) {
    auto message = build(payload, {}, /*log_time_ns=*/42);
    ASSERT_TRUE(message.has_value()) << message.error();
    auto emit_result = graph.topic.emit(**message);
    emitted = emit_result.has_value();
    if (!emitted) {
      last_error = emit_result.error();
      std::this_thread::sleep_for(1ms);
    }
  }
  ASSERT_TRUE(emitted) << "publishing never became ready: "
                       << holoscan::runtime_error_message(last_error);

  std::optional<isaac::PoseCpuT> received = Observation::wait();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(received.has_value());
  EXPECT_DOUBLE_EQ(received->position.x(), sent.position.x());
}

}  // namespace
}  // namespace isaac::os::mcap_replayer
