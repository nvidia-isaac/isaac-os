/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/lifecycle.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/readiness_source.hpp"
#include "holoscan/core/run.hpp"
#include "holoscan/time/realtime_clock.hpp"

#include "common/mcap_logger/mcap_logger.h"
#include "common/mcap_logger/tests/chunk_compressions.h"
#include "common/mcap_logger/tests/post_with_retry.h"
#include "common/mcap_logger/tests/read_in_log_time_order.h"
#include "common/mcap_logger/tests/scoped_temp_directory.h"
#include "joint_command_cpu_schema_traits.h"
#include "mcap/reader.hpp"

namespace isaac::os::mcap_logger {
namespace {

using isaac::os::mcap_logger::testing::post_with_retry;

// Fixed, known-in-advance number of samples the generator below will ever be triggered to
// emit -- the test posts exactly this many notifications, so the expected count is guaranteed
// by construction rather than inferred from how long the graph happened to run.
constexpr int kMessageCount = 20;
// Posts the generator can have pending; the burst test posts hundreds without spacing, and a post
// past this is refused as backpressured.
constexpr int kTriggerCapacity = 512;

// Interval the test sleeps between posted notifications, so consecutive MCAP log_time values
// end up spaced roughly this far apart -- letting the read-back check that log_time reflects
// real publish timing, not some unrelated (or constant/broken) value.
constexpr auto kGeneratorPeriod = std::chrono::milliseconds(20);

// Publishes the running generator operator's NotificationSender out to the test, and lets the
// test block until it's available. holoscan::compile() reconstructs operators into its own
// immutable canonical Graph rather than reusing the exact objects compose() built (per
// execution_plan.hpp's ExecutionGraphView doc comment), so this is the one channel that
// reliably crosses that boundary: the same pattern the SDK's own test suite uses for exactly
// this purpose (see RuntimeNotificationProbeOperator in
// holoscan-sdk/public/tests/runtime/runtime_test_graph.cpp).
class GeneratorTriggerProbe {
 public:
  static void publish(holoscan::NotificationSender sender) {
    const std::lock_guard<std::mutex> lock(mutex_);
    sender_ = std::move(sender);
    cv_.notify_all();
  }

  // Blocks until the running graph's generator operator has armed and published its sender.
  [[nodiscard]] static holoscan::NotificationSender wait_for_sender() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [] { return sender_.has_value(); });
    return *sender_;
  }

  static void reset() {
    const std::lock_guard<std::mutex> lock(mutex_);
    sender_.reset();
    emitted_ = 0;
  }

  // Called by the generator after each emit.
  static void note_emitted() {
    const std::scoped_lock lock(mutex_);
    ++emitted_;
    cv_.notify_all();
  }

  // Blocks until the generator has emitted `count` samples, or `timeout` passes; true if it did.
  [[nodiscard]] static bool wait_for_emitted(int count, std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, timeout, [count] { return emitted_ >= count; });
  }

 private:
  static std::mutex mutex_;
  static std::condition_variable cv_;
  static std::optional<holoscan::NotificationSender> sender_;
  static int emitted_;
};

std::mutex GeneratorTriggerProbe::mutex_;
std::condition_variable GeneratorTriggerProbe::cv_;
std::optional<holoscan::NotificationSender> GeneratorTriggerProbe::sender_;
int GeneratorTriggerProbe::emitted_ = 0;

// Emits one JointCommandCpuT with a monotonically increasing first position value each time
// its NotificationSource is posted to, so the integration tests below can drive the graph an
// exact, known number of times without any external process and without guessing how long a
// wall-clock sleep needs to be to cover N iterations of a free-running clock.
class JointCommandGeneratorOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(command_out, "command_out");
    spec.notification_source(trigger_, "trigger").capacity(kTriggerCapacity);
    spec.lifecycle().stage(holoscan::LifecycleStage::kArm, &JointCommandGeneratorOp::arm);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnNotified{.event = trigger_});
    return contract;
  }

  // Acquires this operator's own NotificationSender during the Arm lifecycle stage and hands
  // it out to the test via GeneratorTriggerProbe -- the test then calls sender.post() to
  // trigger compute() exactly once per call.
  holoscan::LifecycleStatus arm(holoscan::LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(trigger_);
    if (!sender) {
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    GeneratorTriggerProbe::publish(std::move(*sender));
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    isaac::JointCommandCpuT value;
    value.position = {static_cast<float>(sequence_)};
    value.velocity = {0.0F};
    value.effort = {0.0F};
    value.kp = {0.0F};
    value.kd = {0.0F};
    ++sequence_;
    auto emitted = command_out.emit(value);
    GeneratorTriggerProbe::note_emitted();
    return emitted;
  }

  holoscan::Output<isaac::JointCommandCpuT> command_out;

 private:
  holoscan::NotificationSource trigger_;
  int sequence_ = 0;
};

class IntegrationTestGraph final : public holoscan::Graph {
 public:
  explicit IntegrationTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_integration_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto generator = op<JointCommandGeneratorOp>("generator");
    logger_.attach<isaac::JointCommandCpuT>(*this, "joint_command", generator->command_out);
  }

 private:
  HsdkMcapLogger& logger_;
};

// Like IntegrationTestGraph, but with a queue deep enough that a burst cannot drop anything.
class DeepQueueIntegrationTestGraph final : public holoscan::Graph {
 public:
  explicit DeepQueueIntegrationTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_deep_queue_integration_test"), logger_(logger) {}

 private:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto generator = op<JointCommandGeneratorOp>("generator");
    logger_.attach<isaac::JointCommandCpuT>(*this, "joint_command", generator->command_out,
                                            TopicLogOptions{.queue_depth = 1024});
  }

  HsdkMcapLogger& logger_;
};

// Every test in this fixture drives IntegrationTestGraph's generator via GeneratorTriggerProbe,
// whose sender_ slot is process-lifetime static state shared across all TEST_Fs below (see the
// class comment above). SetUp() resets it before each test so a forgotten reset can never leave
// a later test blocking on -- or silently reusing -- a previous test's stale sender; this is
// enforced by the fixture rather than left to each TEST_F to remember on its own.
class HsdkMcapLoggerIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override { GeneratorTriggerProbe::reset(); }

  // Records a few messages from the test graph with `options`, then reads back what each chunk of
  // the finished file says about its compression. Uses ASSERT_*, so a caller checks
  // HasFatalFailure() before using `chunks`.
  void RecordAndReadChunkCompressions(const HsdkMcapLoggerOptions& options,
                                      std::vector<ChunkCompression>& chunks) {
    {
      HsdkMcapLogger logger(options);
      IntegrationTestGraph graph(logger);
      const holoscan::ExecutionPlan plan = holoscan::compile(graph);
      ASSERT_TRUE(plan.ok());

      logger.start(plan.endpoint_catalog());
      auto run_session = holoscan::run_async(plan);
      holoscan::NotificationSender sender = GeneratorTriggerProbe::wait_for_sender();
      constexpr int kFewMessages = 3;
      for (int i = 0; i < kFewMessages; ++i) {
        auto posted = post_with_retry(sender);
        ASSERT_TRUE(posted) << holoscan::runtime_error_message(posted.error());
        std::this_thread::sleep_for(kGeneratorPeriod);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      run_session.request_stop();
      run_session.wait();
      logger.shutdown();
    }
    chunks = ReadChunkCompressions(options.output_path);
  }
};

TEST_F(HsdkMcapLoggerIntegrationTest, LogsSamplesEmittedByARunningGraph) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_integration_test.mcap");

  {
    HsdkMcapLogger logger(
        HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
    IntegrationTestGraph graph(logger);
    const holoscan::ExecutionPlan plan = holoscan::compile(graph);
    ASSERT_TRUE(plan.ok());

    logger.start(plan.endpoint_catalog());
    auto run_session = holoscan::run_async(plan);

    holoscan::NotificationSender sender = GeneratorTriggerProbe::wait_for_sender();
    // Post exactly kMessageCount notifications, spaced kGeneratorPeriod apart -- deterministic
    // in count (each post() is individually confirmed, not inferred from elapsed time) while
    // still producing real, known inter-publish timing for the log_time check below.
    for (int i = 0; i < kMessageCount; ++i) {
      auto posted = post_with_retry(sender);
      ASSERT_TRUE(posted) << holoscan::runtime_error_message(posted.error());
      std::this_thread::sleep_for(kGeneratorPeriod);
    }

    // TopicPoller waits for samples on its own thread, independent of the graph's own run
    // session -- so the very last posted sample may still be sitting unread in the channel right
    // after the loop above returns. Give the poller a brief moment to catch up before stopping it.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    run_session.request_stop();
    run_session.wait();
    logger.shutdown();  // Finalizes the output file directly; the scope exit below (running
                        // HsdkMcapLogger's destructor) is then just a no-op re-call of shutdown().
  }

  mcap::McapReader reader;
  const auto status = reader.open(output_path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  // log_time is expected to track kGeneratorPeriod (the real inter-publish spacing) rather
  // than, say, recorder-pipeline timing or a constant/broken value -- checked as
  // kGeneratorPeriod +/- kGeneratorPeriodTolerance to allow for scheduling jitter without
  // being so loose the check is meaningless.
  constexpr auto kGeneratorPeriodTolerance = std::chrono::milliseconds(5);
  const std::uint64_t kMinDeltaNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        kGeneratorPeriod - kGeneratorPeriodTolerance)
                                        .count();
  const std::uint64_t kMaxDeltaNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        kGeneratorPeriod + kGeneratorPeriodTolerance)
                                        .count();

  int message_count = 0;
  std::uint64_t previous_log_time = 0;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    EXPECT_EQ(view.channel->topic, "joint_command");
    EXPECT_EQ(view.schema->name, "isaac.JointCommandCpu");
    // No monotonicity assertion here: ReadInLogTimeOrder() yields messages sorted by log_time,
    // so one would be tautological. The spacing check below is the real assertion -- that
    // log_time tracks the generator's actual publish period rather than a constant, a
    // recorder-side timestamp, or a broken value.
    if (message_count > 0) {
      const std::uint64_t delta_ns = view.message.logTime - previous_log_time;
      EXPECT_GE(delta_ns, kMinDeltaNs) << "at message index " << message_count;
      EXPECT_LE(delta_ns, kMaxDeltaNs) << "at message index " << message_count;
    }
    previous_log_time = view.message.logTime;

    // The generator is triggered exactly kMessageCount times and writes its sequence number
    // into position[0], so decoding it back out lets us check message_count and ordering
    // exactly -- against a bound the test itself fixed in advance, not anything read back off
    // a live operator instance.
    const auto* message = isaac::GetJointCommandCpu(view.message.data);
    ASSERT_NE(message, nullptr);
    ASSERT_NE(message->position(), nullptr);
    ASSERT_EQ(message->position()->size(), 1U);
    EXPECT_EQ(message->position()->Get(0), static_cast<float>(message_count));

    ++message_count;
  }
  EXPECT_EQ(message_count, kMessageCount);
}

TEST_F(HsdkMcapLoggerIntegrationTest, CompressesChunksWithLz4UnlessToldOtherwise) {
  ScopedTempDirectory temp_directory;
  std::vector<ChunkCompression> chunks;
  RecordAndReadChunkCompressions(
      HsdkMcapLoggerOptions{.output_path = temp_directory.path("compression_default.mcap"),
                            .timestamp_output_path = false},
      chunks);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_FALSE(chunks.empty());
  for (const ChunkCompression& chunk : chunks) {
    EXPECT_EQ(chunk.algorithm, "lz4");
  }
}

TEST_F(HsdkMcapLoggerIntegrationTest, WritesTheCompressionItIsConfiguredWith) {
  ScopedTempDirectory temp_directory;
  std::vector<ChunkCompression> chunks;
  RecordAndReadChunkCompressions(
      HsdkMcapLoggerOptions{.output_path = temp_directory.path("compression_none.mcap"),
                            .timestamp_output_path = false,
                            .compression = foxglove::McapCompression::None},
      chunks);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_FALSE(chunks.empty());
  for (const ChunkCompression& chunk : chunks) {
    EXPECT_EQ(chunk.algorithm, "");
  }
}

TEST_F(HsdkMcapLoggerIntegrationTest, ShutdownFinalizesTheFileBeforeTheLoggerIsDestroyed) {
  // Regression test: shutdown() must itself finalize (close) the MCAP file, not merely rely on
  // HsdkMcapLogger's destructor running afterward. Reads the file back while `logger` is still
  // a live object (its destructor has not run yet), immediately after shutdown() returns.
  ScopedTempDirectory temp_directory;
  const auto output_path =
      temp_directory.path("mcap_logger_integration_test_shutdown_finalizes.mcap");

  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
  IntegrationTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  logger.start(plan.endpoint_catalog());
  auto run_session = holoscan::run_async(plan);

  holoscan::NotificationSender sender = GeneratorTriggerProbe::wait_for_sender();
  constexpr int kFewMessages = 3;
  // Paced kGeneratorPeriod apart, same as LogsSamplesEmittedByARunningGraph above -- posting
  // all kFewMessages back-to-back left the poller (1ms interval) needing to drain all of them
  // inside one fixed tail sleep, which flaked under host load. Spacing the posts gives it a
  // full period to catch up after each one, so the tail sleep only ever has to cover the last
  // sample, not all of them at once.
  for (int i = 0; i < kFewMessages; ++i) {
    auto posted = post_with_retry(sender);
    ASSERT_TRUE(posted) << holoscan::runtime_error_message(posted.error());
    std::this_thread::sleep_for(kGeneratorPeriod);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  run_session.request_stop();
  run_session.wait();
  logger.shutdown();

  // `logger` is still alive here -- its destructor has not run. If shutdown() didn't finalize
  // the file itself, this read-back would see a truncated/unfinalized MCAP file.
  mcap::McapReader reader;
  const auto status = reader.open(output_path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  int message_count = 0;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    (void)view;
    ++message_count;
  }
  EXPECT_EQ(message_count, kFewMessages);
}

TEST_F(HsdkMcapLoggerIntegrationTest, DestructorFinalizesTheFileWithoutExplicitShutdown) {
  // Regression test for the other half of finalization: HsdkMcapLogger's destructor must itself
  // finalize the file when the caller never calls shutdown() at all (e.g. the owning app exits
  // via RAII, not an explicit graceful-shutdown call). ShutdownFinalizesTheFileBeforeTheLoggerIs-
  // Destroyed above covers the explicit-call path; this test never calls shutdown().
  ScopedTempDirectory temp_directory;
  const auto output_path =
      temp_directory.path("mcap_logger_integration_test_destructor_finalizes.mcap");

  constexpr int kFewMessages = 3;
  {
    HsdkMcapLogger logger(
        HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
    IntegrationTestGraph graph(logger);
    const holoscan::ExecutionPlan plan = holoscan::compile(graph);
    ASSERT_TRUE(plan.ok());

    logger.start(plan.endpoint_catalog());
    auto run_session = holoscan::run_async(plan);

    holoscan::NotificationSender sender = GeneratorTriggerProbe::wait_for_sender();
    // Paced kGeneratorPeriod apart (see ShutdownFinalizesTheFileBeforeTheLoggerIsDestroyed's
    // matching comment above) so the poller drains each sample as it arrives instead of
    // needing to catch up on all of them inside one fixed tail sleep -- this test's only
    // intended variable is finalization, not also racing the poller for message count, which
    // the next test below covers instead.
    for (int i = 0; i < kFewMessages; ++i) {
      auto posted = post_with_retry(sender);
      ASSERT_TRUE(posted) << holoscan::runtime_error_message(posted.error());
      std::this_thread::sleep_for(kGeneratorPeriod);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    run_session.request_stop();
    run_session.wait();
    // No call to logger.shutdown() here -- `logger` goes out of scope at the end of this block,
    // and its destructor is the only thing that finalizes the file.
  }

  mcap::McapReader reader;
  const auto status = reader.open(output_path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  int message_count = 0;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    (void)view;
    ++message_count;
  }
  EXPECT_EQ(message_count, kFewMessages);
}

TEST_F(HsdkMcapLoggerIntegrationTest, ShutdownImmediatelyAfterStopProducesAReadableFile) {
  // Regression test for an abrupt/early shutdown: unlike every other test in this file, this one
  // does not give TopicPoller's poll thread a moment to drain already-published-but-not-yet-read
  // samples before shutdown() runs. shutdown() destroys pollers (and, transitively, channel
  // subscribers) and writes what each poller's queue holds while a sample may still be in flight
  // -- this must not corrupt the file or crash, even though it may mean some trailing samples are
  // never recorded. Message count is therefore checked as a bound, not an exact value; every
  // message that IS present is still decoded and checked for valid content, so this also catches
  // corruption (e.g. a torn write, or a use-after-free clobbering a queued sample) that would
  // otherwise show up only as garbage bytes rather than a missing message.
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_integration_test_abrupt_shutdown.mcap");

  constexpr int kFewMessages = 5;
  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
  IntegrationTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  logger.start(plan.endpoint_catalog());
  auto run_session = holoscan::run_async(plan);

  holoscan::NotificationSender sender = GeneratorTriggerProbe::wait_for_sender();
  for (int i = 0; i < kFewMessages; ++i) {
    auto posted = post_with_retry(sender);
    ASSERT_TRUE(posted) << holoscan::runtime_error_message(posted.error());
  }

  // No drain sleep: stop the graph and shut the logger down back-to-back, racing the poller and its
  // poll thread on purpose.
  run_session.request_stop();
  run_session.wait();
  logger.shutdown();

  mcap::McapReader reader;
  const auto status = reader.open(output_path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  int message_count = 0;
  int previous_position = -1;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    EXPECT_EQ(view.channel->topic, "joint_command");
    ASSERT_NE(view.schema, nullptr);
    EXPECT_EQ(view.schema->name, "isaac.JointCommandCpu");

    const auto* message = isaac::GetJointCommandCpu(view.message.data);
    ASSERT_NE(message, nullptr);
    ASSERT_NE(message->position(), nullptr);
    ASSERT_EQ(message->position()->size(), 1U);
    // Sequence numbers are assigned in generator compute() order, so any messages that did make
    // it into the file must still appear in that same increasing order -- catching corruption
    // (a message with garbage or out-of-range content) even when fewer than kFewMessages made it
    // in before shutdown() won the race.
    const auto position = static_cast<int>(message->position()->Get(0));
    EXPECT_GT(position, previous_position);
    EXPECT_GE(position, 0);
    EXPECT_LT(position, kFewMessages);
    previous_position = position;

    ++message_count;
  }
  EXPECT_GE(message_count, 0);
  EXPECT_LE(message_count, kFewMessages);
}

// The poll thread writes what the queue holds when it is told to stop. A burst with no spacing
// leaves the poller behind the producer, so there is something to write at that moment; the
// graph is stopped as soon as the last sample has been emitted, and every sample must be in the
// file, in order.
TEST_F(HsdkMcapLoggerIntegrationTest, WritesEverySampleOfABurstPublishedBeforeStop) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_integration_test_burst.mcap");

  constexpr int kBurst = 300;
  {
    HsdkMcapLogger logger(
        HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
    DeepQueueIntegrationTestGraph graph(logger);
    const holoscan::ExecutionPlan plan = holoscan::compile(graph);
    ASSERT_TRUE(plan.ok());

    logger.start(plan.endpoint_catalog());
    auto run_session = holoscan::run_async(plan);

    holoscan::NotificationSender sender = GeneratorTriggerProbe::wait_for_sender();
    for (int i = 0; i < kBurst; ++i) {
      auto posted = post_with_retry(sender);
      ASSERT_TRUE(posted) << holoscan::runtime_error_message(posted.error());
    }
    // Stop as soon as the generator has emitted the last sample, with no wait for the poller: it
    // is still working through the burst, and the stop has to leave it the queue to write.
    ASSERT_TRUE(GeneratorTriggerProbe::wait_for_emitted(kBurst, std::chrono::seconds(10)));

    run_session.request_stop();
    run_session.wait();
    logger.shutdown();
  }

  mcap::McapReader reader;
  const auto status = reader.open(output_path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  int message_count = 0;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    const auto* message = isaac::GetJointCommandCpu(view.message.data);
    ASSERT_NE(message, nullptr);
    ASSERT_NE(message->position(), nullptr);
    ASSERT_EQ(message->position()->size(), 1U);
    EXPECT_EQ(message->position()->Get(0), static_cast<float>(message_count));
    ++message_count;
  }
  EXPECT_EQ(message_count, kBurst);
}

}  // namespace
}  // namespace isaac::os::mcap_logger
