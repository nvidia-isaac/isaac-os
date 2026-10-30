/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/run.hpp"
#include "holoscan/core/temporal_contract.hpp"
#include "holoscan/schema/binary_schema.hpp"
#include "holoscan/schema/schema_package.hpp"
#include "holoscan/time/realtime_clock.hpp"

#include "common/mcap_logger/mcap_logger.h"
#include "common/mcap_logger/tests/scoped_temp_directory.h"
#include "common/mcap_replayer/mcap_file.h"
#include "common/mcap_replayer/mcap_replayer.h"
#include "common/mcap_replayer/tests/mcap_test_support.h"
#include "mcap/writer.hpp"
#include "pose_cpu_schema_traits.h"
#include "scoped_log_level.h"
#include "transform_cpu_schema_traits.h"
#include "twist_cpu_schema_traits.h"

namespace isaac::os::mcap_replayer {
namespace {

using namespace std::chrono_literals;
using isaac::os::mcap_replayer::test_support::read_message;

constexpr const char* kPoseTopic = "pose";
constexpr const char* kTwistTopic = "twist";
constexpr const char* kTransformTopic = "transform";

// Long enough that the pose fixture holds many times over the small buffer the loss test bounds
// itself against (ReplaysARecordingManyTimesTheBufferBoundWithoutLoss's kBufferMessages), so that
// test does not have to record its own file.
constexpr auto kFixtureDuration = 1600ms;
constexpr auto kPosePeriod = 5ms;

// Each fixture operator writes its sequence number into the x coordinate of its first vector
// field, so a replayed message identifies itself without depending on arrival timing.
class FixturePoseOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out_pose, "out_pose"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kPosePeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    isaac::PoseCpuT value;
    value.position = holoscan::schema::Vector3d(static_cast<double>(sequence_), 0.0, 0.0);
    value.orientation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);
    ++sequence_;
    return out_pose.emit(value);
  }

  holoscan::Output<isaac::PoseCpuT> out_pose;

 private:
  int sequence_ = 0;
};

class FixtureTwistOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out_twist, "out_twist"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 7ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    isaac::TwistCpuT value;
    value.linear = holoscan::schema::Vector3d(static_cast<double>(sequence_), 0.0, 0.0);
    value.angular = holoscan::schema::Vector3d(0.0, 0.0, 0.0);
    ++sequence_;
    return out_twist.emit(value);
  }

  holoscan::Output<isaac::TwistCpuT> out_twist;

 private:
  int sequence_ = 0;
};

class FixtureTransformOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out_transform, "out_transform"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = 11ms});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    isaac::TransformCpuT value;
    value.translation = holoscan::schema::Vector3d(static_cast<double>(sequence_), 0.0, 0.0);
    value.rotation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);
    ++sequence_;
    return out_transform.emit(value);
  }

  holoscan::Output<isaac::TransformCpuT> out_transform;

 private:
  int sequence_ = 0;
};

// Records pose alone, or pose alongside two other flat types at rates that share no common
// period with it -- interleaving of unequal rates is what a single shared buffer risks getting
// wrong, and a single-rate recording cannot expose it.
class FixtureGraph final : public holoscan::Graph {
 public:
  FixtureGraph(isaac::os::mcap_logger::HsdkMcapLogger& logger, bool multi_topic)
      : Graph("mcap_replayer_fixture"), logger_(logger), multi_topic_(multi_topic) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));

    auto pose_source = op<FixturePoseOp>("pose_source");
    logger_.attach<isaac::PoseCpuT>(*this, kPoseTopic, pose_source->out_pose);
    if (!multi_topic_) {
      return;
    }

    auto twist_source = op<FixtureTwistOp>("twist_source");
    logger_.attach<isaac::TwistCpuT>(*this, kTwistTopic, twist_source->out_twist);

    auto transform_source = op<FixtureTransformOp>("transform_source");
    logger_.attach<isaac::TransformCpuT>(*this, kTransformTopic, transform_source->out_transform);
  }

 private:
  isaac::os::mcap_logger::HsdkMcapLogger& logger_;
  bool multi_topic_;
};

// Where a fixture named name is written: under a private directory made once and removed when the
// test program exits, since the fixtures are shared by every test that asks for them.
std::filesystem::path fixture_directory(const std::string& name) {
  static const isaac::os::mcap_logger::ScopedTempDirectory root;
  const std::filesystem::path directory = root.path(name);
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  return directory;
}

std::filesystem::path write_fixture(const std::string& name, bool multi_topic) {
  using isaac::os::mcap_logger::HsdkMcapLogger;
  using isaac::os::mcap_logger::HsdkMcapLoggerOptions;

  const auto directory = fixture_directory(name);

  HsdkMcapLogger logger(HsdkMcapLoggerOptions{.output_path = directory / "fixture.mcap",
                                              .timestamp_output_path = false});
  FixtureGraph graph(logger, multi_topic);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  EXPECT_TRUE(plan.ok());
  logger.start(plan.endpoint_catalog());
  holoscan::RunSession session = holoscan::run_async(plan);
  std::this_thread::sleep_for(kFixtureDuration);
  session.request_stop();
  session.wait();
  logger.shutdown();
  // The logger writes a file only when it is configured with one; every fixture here is, so an
  // empty result means the fixture itself was mis-set up rather than the replayer.
  const std::optional<std::filesystem::path>& written = logger.output_path();
  EXPECT_TRUE(written.has_value()) << "the fixture logger produced no file to read back";
  return written.value_or(std::filesystem::path{});
}

// Recording a fixture costs real wall-clock time, so the two shapes every test below needs are
// each recorded once for the whole executable. Nothing here modifies a recording.
const std::filesystem::path& pose_fixture() {
  static const std::filesystem::path path = write_fixture("pose", /*multi_topic=*/false);
  return path;
}

const std::filesystem::path& multi_topic_fixture() {
  static const std::filesystem::path path = write_fixture("multi_topic", /*multi_topic=*/true);
  return path;
}

// Writes a recording by hand, one channel of pose messages under the schema name and schema
// bytes given, at the recorded times given. The logger always writes the schema it was really
// handed, at the times messages really happened, so a recording whose schema bytes have moved on
// or whose messages all share one instant can only be made this way.
std::filesystem::path write_recording_by_hand(const std::string& name,
                                              const std::vector<std::byte>& binary_schema,
                                              const std::vector<std::uint64_t>& log_times,
                                              bool unreadable_payloads = false) {
  const auto directory = fixture_directory(name);
  const auto path = directory / "by_hand.mcap";

  mcap::McapWriter writer;
  const mcap::McapWriterOptions options{""};
  const mcap::Status opened = writer.open(path.string(), options);
  EXPECT_TRUE(opened.ok()) << opened.message;

  mcap::Schema schema(
      std::string(holoscan::schema_package<isaac::PoseCpuT>().identity().canonical_name),
      "flatbuffer", mcap::ByteArray(binary_schema.begin(), binary_schema.end()));
  writer.addSchema(schema);
  mcap::Channel channel(kPoseTopic, "flatbuffer", schema.id);
  writer.addChannel(channel);

  for (std::size_t index = 0; index < log_times.size(); ++index) {
    if (unreadable_payloads) {
      const std::vector<std::byte> garbage(16, std::byte{0xFF});
      mcap::Message damaged{};
      damaged.channelId = channel.id;
      damaged.logTime = log_times[index];
      damaged.publishTime = log_times[index];
      damaged.data = garbage.data();
      damaged.dataSize = garbage.size();
      const mcap::Status written = writer.write(damaged);
      EXPECT_TRUE(written.ok()) << written.message;
      continue;
    }
    isaac::PoseCpuT pose;
    pose.position = holoscan::schema::Vector3d(static_cast<double>(index), 0.0, 0.0);
    pose.orientation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);
    const auto& descriptor = holoscan::schema_package<isaac::PoseCpuT>();
    std::vector<std::byte> payload(descriptor.max_serialized_size());
    const auto encoded = descriptor.encode(pose, payload);
    EXPECT_TRUE(encoded.has_value());
    payload.resize(*encoded);

    mcap::Message message{};
    message.channelId = channel.id;
    message.logTime = log_times[index];
    message.publishTime = log_times[index];
    message.data = payload.data();
    message.dataSize = payload.size();
    const mcap::Status written = writer.write(message);
    EXPECT_TRUE(written.ok()) << written.message;
  }

  writer.close();
  return path;
}

std::vector<std::byte> recorded_pose_schema() {
  const std::span<const std::byte> schema = holoscan::binary_schema<isaac::PoseCpuT>();
  return std::vector<std::byte>(schema.begin(), schema.end());
}

// Counts how often marker appears in text, for the tests that read what a run logged.
std::size_t occurrences_of(const std::string& text, const std::string& marker) {
  std::size_t count = 0;
  for (std::size_t at = text.find(marker); at != std::string::npos;
       at = text.find(marker, at + marker.size())) {
    ++count;
  }
  return count;
}

// What one pass over a recording holds. The reader loops, so a pass is bounded by watching for
// the recorded time to move backwards rather than by the reader running out.
struct RecordingFacts {
  std::unordered_map<std::string, std::size_t> counts;
  std::chrono::nanoseconds span{0};
};

RecordingFacts read_facts(const std::filesystem::path& path) {
  McapFile opened = McapFile::open(path, /*should_loop=*/true);
  std::unordered_map<mcap::ChannelId, std::string> topics;
  for (const mcap::ChannelId id : opened.channel_ids()) {
    topics[id] = opened.topic_of(id).value_or(std::string{});
  }

  RecordingFacts facts;
  RecordedMessage message;
  std::uint64_t first = 0;
  std::uint64_t previous = 0;
  bool started = false;
  while (read_message(opened, message)) {
    if (started && message.log_time_ns < previous) {
      break;
    }
    if (!started) {
      first = message.log_time_ns;
      started = true;
    }
    previous = message.log_time_ns;
    ++facts.counts[topics[message.channel_id]];
  }
  facts.span = std::chrono::nanoseconds(previous - first);
  return facts;
}

// One arrival at a wired consumer: which message it was, and when it turned up.
struct Arrival {
  int sequence = 0;
  std::chrono::steady_clock::time_point at;
};

int sequence_of(const isaac::PoseCpuT& value) { return static_cast<int>(value.position.x()); }
int sequence_of(const isaac::TwistCpuT& value) { return static_cast<int>(value.linear.x()); }
int sequence_of(const isaac::TransformCpuT& value) {
  return static_cast<int>(value.translation.x());
}

// Records every message a replayed topic delivers. The log is static because an operator built
// through Graph::op<>() only ever receives copy-constructed arguments, so it cannot be handed a
// reference to live test state; one instantiation per T keeps topics of different types apart.
template <typename T>
class CountingSinkOp final : public holoscan::Operator<> {
 public:
  static void reset() {
    const std::lock_guard<std::mutex> lock(mutex_);
    arrivals_.clear();
  }

  static std::size_t count() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return arrivals_.size();
  }

  static std::vector<Arrival> arrivals() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return arrivals_;
  }

  // Polls rather than waits on a condition: the point of every caller is an upper bound on how
  // long delivery may take, and the poll interval is far below every bound asserted against.
  static std::size_t wait_for(std::size_t wanted, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (count() < wanted && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
    return count();
  }

  // Messages are published individually, but at the accelerated playback speeds some of these
  // tests use, the pacing thread can release many in the time it takes this sink's own activation
  // to drain even one, so the queue has to be deep enough to absorb that gap rather than lose
  // messages to it.
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(256); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{in});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = in.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    const Arrival arrival{.sequence = sequence_of(received->data),
                          .at = std::chrono::steady_clock::now()};
    const std::lock_guard<std::mutex> lock(mutex_);
    arrivals_.push_back(arrival);
    return {};
  }

  holoscan::Input<T> in;

 private:
  static inline std::mutex mutex_;
  static inline std::vector<Arrival> arrivals_;
};

// Builds its replayer from inside compose(), where the replayer's attach() imports each topic and
// wires it with add_flow(), the only place a graph may be authored. The replayer itself needs no
// graph to be built.
class ReplayerTestGraph final : public holoscan::Graph {
 public:
  explicit ReplayerTestGraph(
      std::function<std::unique_ptr<McapReplayer>(holoscan::Graph&)> build_replayer)
      : Graph("mcap_replayer_test"), build_replayer_(std::move(build_replayer)) {}

  [[nodiscard]] McapReplayer& replayer() { return *replayer_; }

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    replayer_ = build_replayer_(*this);
  }

 private:
  std::function<std::unique_ptr<McapReplayer>(holoscan::Graph&)> build_replayer_;
  std::unique_ptr<McapReplayer> replayer_;
};

// The sequence numbers of a run's first wanted arrivals -- bounding a loss check to a prefix
// rather than the whole of arrivals, which a looping replay would otherwise keep growing forever.
std::vector<int> first_sequences(const std::vector<Arrival>& arrivals, std::size_t wanted) {
  std::vector<int> sequences;
  for (std::size_t index = 0; index < std::min(wanted, arrivals.size()); ++index) {
    sequences.push_back(arrivals[index].sequence);
  }
  return sequences;
}

// The run of sequence numbers a lossless pass must produce: consecutive from wherever the
// recording's own first message started, so a test detects a gap anywhere in the pass without
// also depending on the fixture having captured its very first emission.
std::vector<int> consecutive_from(int first, std::size_t count) {
  std::vector<int> expected(count);
  for (std::size_t index = 0; index < count; ++index) {
    expected[index] = first + static_cast<int>(index);
  }
  return expected;
}

// Receives replayed poses and sleeps for per_message_delay before accepting the next one, with a
// one-deep queue that is wired to drop its oldest unread sample (the default kDropOldest queue
// policy) rather than block the publisher, so a delay longer than the recorded interval makes the
// replayer's next emit silently overwrite the one still waiting here instead of finding nowhere
// to go.
class SlowSinkOp final : public holoscan::Operator<> {
 public:
  static void reset() {
    // Nothing to reset: this sink keeps no counts.
  }

  explicit SlowSinkOp(std::chrono::milliseconds per_message_delay) : delay_(per_message_delay) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(1); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{in});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = in.receive();
    if (!received) {
      return holoscan::make_unexpected(received.error());
    }
    std::this_thread::sleep_for(delay_);
    return {};
  }

  holoscan::Input<isaac::PoseCpuT> in;

 private:
  std::chrono::milliseconds delay_;
};

// True once every topic has accounted for a full pass of the recording: each of its recorded
// messages has either reached the consumer or been counted as refused.
bool pass_complete(const ReplayStats& stats, const std::map<std::string, std::size_t>& expected) {
  for (const auto& [topic, count] : expected) {
    if (stats.emitted(topic) + stats.refused(topic) < count) {
      return false;
    }
  }
  return true;
}

// Waits for a whole pass to be accounted for, and reports whether it was.
//
// The tests below wait on the run's own account rather than on a count of arrivals, because a
// message counted as refused is one the consumer never receives. Replay holds its first message
// back until every wired consumer has subscribed, so none is expected to be refused here, but the
// subscription it waits on is the publisher's own view of the connection rather than a guarantee
// of delivery, and these tests are about something else. What each one checks is the part that
// is replay's own: every message it published arrived, in order and without a gap.
bool wait_for_pass(const ReplayStats& stats, const std::map<std::string, std::size_t>& expected,
                   std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!pass_complete(stats, expected) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return pass_complete(stats, expected);
}

}  // namespace

TEST(McapReplayer, ReplaysEveryDeclaredTopicOfADifferentTypeToItsOwnConsumer) {
  const std::filesystem::path path = multi_topic_fixture();
  const RecordingFacts facts = read_facts(path);
  ASSERT_GT(facts.counts.at(kPoseTopic), 0U);
  ASSERT_GT(facts.counts.at(kTwistTopic), 0U);
  ASSERT_GT(facts.counts.at(kTransformTopic), 0U);

  CountingSinkOp<isaac::PoseCpuT>::reset();
  CountingSinkOp<isaac::TwistCpuT>::reset();
  CountingSinkOp<isaac::TransformCpuT>::reset();

  // Five times speed keeps the run short: the recording's 1.6 s span compresses to 320 ms here.
  constexpr double kSpeed = 5.0;
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(
        path, ReplayOptions{.playback_speed = kSpeed}, Topic<isaac::PoseCpuT>{kPoseTopic},
        Topic<isaac::TwistCpuT>{kTwistTopic}, Topic<isaac::TransformCpuT>{kTransformTopic});

    auto pose_sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("pose_sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, pose_sink->in);
    auto twist_sink = owner.op<CountingSinkOp<isaac::TwistCpuT>>("twist_sink");
    replayer->attach<isaac::TwistCpuT>(owner, kTwistTopic, twist_sink->in);
    auto transform_sink = owner.op<CountingSinkOp<isaac::TransformCpuT>>("transform_sink");
    replayer->attach<isaac::TransformCpuT>(owner, kTransformTopic, transform_sink->in);
    return replayer;
  });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);
  graph.replayer().start(plan.endpoint_catalog());

  const std::map<std::string, std::size_t> expected{
      {kPoseTopic, facts.counts.at(kPoseTopic)},
      {kTwistTopic, facts.counts.at(kTwistTopic)},
      {kTransformTopic, facts.counts.at(kTransformTopic)}};
  const ReplayStats& stats = graph.replayer().stats();
  ASSERT_TRUE(wait_for_pass(stats, expected, 20s)) << "the recording was not replayed through";

  // Every published message arrives on its own topic's port, and nowhere else: a run that
  // delivered one topic's messages to another's consumer would leave one of these short.
  const std::size_t poses = stats.emitted(kPoseTopic);
  const std::size_t twists = stats.emitted(kTwistTopic);
  const std::size_t transforms = stats.emitted(kTransformTopic);
  EXPECT_EQ(CountingSinkOp<isaac::PoseCpuT>::wait_for(poses, 10s), poses);
  EXPECT_EQ(CountingSinkOp<isaac::TwistCpuT>::wait_for(twists, 10s), twists);
  EXPECT_EQ(CountingSinkOp<isaac::TransformCpuT>::wait_for(transforms, 10s), transforms);
  // Nearly the whole recording, so a run that published almost nothing cannot satisfy the
  // equalities above by matching a handful of messages against a handful of arrivals.
  EXPECT_GT(poses * 10, facts.counts.at(kPoseTopic) * 9);
  EXPECT_GT(twists * 10, facts.counts.at(kTwistTopic) * 9);
  EXPECT_GT(transforms * 10, facts.counts.at(kTransformTopic) * 9);

  const std::vector<Arrival> pose_arrivals = CountingSinkOp<isaac::PoseCpuT>::arrivals();
  ASSERT_FALSE(pose_arrivals.empty());
  EXPECT_EQ(first_sequences(pose_arrivals, poses),
            consecutive_from(pose_arrivals.front().sequence, poses));

  graph.replayer().stop();
  session.request_stop();
  session.wait();

  // Replay holds back until every consumer here has subscribed, so a refusal would mean it sent
  // into a topic that had nobody to receive it after all.
  for (const auto& [topic, count] : expected) {
    EXPECT_EQ(stats.refused(topic), 0U) << "topic '" << topic << "' could not publish everything";
  }
}

TEST(McapReplayer, NeitherImportsNorReplaysATopicNobodyAttached) {
  const std::filesystem::path path = multi_topic_fixture();
  const RecordingFacts facts = read_facts(path);
  ASSERT_GT(facts.counts.at(kPoseTopic), 0U);
  ASSERT_GT(facts.counts.at(kTwistTopic), 0U);
  CountingSinkOp<isaac::PoseCpuT>::reset();

  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(path, ReplayOptions{.playback_speed = 5.0},
                                                   Topic<isaac::PoseCpuT>{kPoseTopic},
                                                   Topic<isaac::TwistCpuT>{kTwistTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("pose_sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);
  graph.replayer().start(plan.endpoint_catalog());

  const ReplayStats& stats = graph.replayer().stats();
  ASSERT_TRUE(wait_for_pass(stats, {{kPoseTopic, facts.counts.at(kPoseTopic)}}, 20s));
  graph.replayer().stop();
  session.request_stop();
  session.wait();

  // The topic has no endpoint to publish on, so its messages are left out of the run entirely
  // rather than counted as refused.
  EXPECT_EQ(stats.emitted(kTwistTopic), 0U);
  EXPECT_EQ(stats.refused(kTwistTopic), 0U);
  EXPECT_EQ(stats.emitted(kPoseTopic), facts.counts.at(kPoseTopic));
}

TEST(McapReplayer, RejectsAttachingOneTopicToTwoConsumersAtCompile) {
  const std::filesystem::path path = multi_topic_fixture();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(path, Topic<isaac::PoseCpuT>{kPoseTopic});
    auto first = owner.op<CountingSinkOp<isaac::PoseCpuT>>("first_sink");
    auto second = owner.op<CountingSinkOp<isaac::PoseCpuT>>("second_sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, first->in);
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, second->in);
    return replayer;
  });
  // An imported topic cannot fan out, and a second attach() must not fail earlier on a duplicate
  // import.
  EXPECT_FALSE(holoscan::compile(graph).ok());
}

TEST(McapReplayer, ReportsHowManyMessagesTheRecordingHoldsOnATopic) {
  const std::filesystem::path path = multi_topic_fixture();
  const RecordingFacts facts = read_facts(path);
  McapReplayer replayer(path, Topic<isaac::PoseCpuT>{kPoseTopic});

  EXPECT_EQ(replayer.message_count(kPoseTopic), facts.counts.at(kPoseTopic));
  EXPECT_FALSE(replayer.message_count("not_recorded").has_value());
}

TEST(McapReplayer, SaysWhetherARecordingHoldsATopicBeforeAReplayerExists) {
  const std::filesystem::path path = multi_topic_fixture();

  EXPECT_TRUE(McapReplayer::recording_has_topic(path, kPoseTopic));
  EXPECT_FALSE(McapReplayer::recording_has_topic(path, "not_recorded"));
  EXPECT_THROW((void)McapReplayer::recording_has_topic(path / "missing.mcap", kPoseTopic),
               std::runtime_error);
}

TEST(McapReplayer, RejectsATopicThatIsNotInTheRecording) {
  // Construction throws before it reaches the one graph-authoring call it makes, so this does
  // not have to run from inside compose() -- and running it outside means the failure surfaces
  // as a construction error rather than being swallowed and re-reported as a composition
  // failure. verify_topic() logs the specific reason via HOLOSCAN_LOG_WARN rather than building
  // it into the exception, so this only checks that construction is rejected at all.
  holoscan::Graph graph("mcap_replayer_rejection_test");
  EXPECT_THROW(McapReplayer(multi_topic_fixture(), Topic<isaac::PoseCpuT>{"does_not_exist"}),
               std::invalid_argument);
}

TEST(McapReplayer, RejectsATopicWhoseRecordedSchemaDoesNotMatchTheRequestedType) {
  holoscan::Graph graph("mcap_replayer_rejection_test");
  EXPECT_THROW(McapReplayer(pose_fixture(), Topic<isaac::TwistCpuT>{kPoseTopic}),
               std::invalid_argument);
}

TEST(McapReplayer, RejectsWiringATopicAsADifferentTypeThanItWasDeclared) {
  const std::filesystem::path path = pose_fixture();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(path, Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  // Both calls throw on the topic lookup, before they reach the graph at all, so they can be made
  // after composition has already finished -- where the message survives to be checked -- against
  // targets no operator owns.
  holoscan::Input<isaac::TwistCpuT> wrong_type;
  holoscan::Input<isaac::PoseCpuT> unknown_topic;
  EXPECT_THROW(graph.replayer().attach<isaac::TwistCpuT>(graph, kPoseTopic, wrong_type),
               std::invalid_argument);
  EXPECT_THROW(graph.replayer().attach<isaac::PoseCpuT>(graph, "never_declared", unknown_topic),
               std::invalid_argument);
}

TEST(McapReplayer, ReplaysInAboutATenthOfTheRecordedSpanAtTenTimesSpeed) {
  // Ten seconds at 100 Hz, written by hand: at ten times speed the run lasts a second, long enough
  // that a stall of the process at the start is a small part of it, which a recording a few
  // hundred milliseconds long could not give.
  constexpr std::size_t kMessages = 1001;
  std::vector<std::uint64_t> log_times;
  for (std::size_t index = 0; index < kMessages; ++index) {
    log_times.push_back(index * 10'000'000U);
  }
  const std::filesystem::path path =
      write_recording_by_hand("ten_times_speed", recorded_pose_schema(), log_times);
  const RecordingFacts facts = read_facts(path);
  const std::size_t recorded = facts.counts.at(kPoseTopic);
  ASSERT_EQ(recorded, kMessages);
  ASSERT_GT(facts.span, 0ns);

  CountingSinkOp<isaac::PoseCpuT>::reset();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(path, ReplayOptions{.playback_speed = 10.0},
                                                   Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);

  const auto began = std::chrono::steady_clock::now();
  graph.replayer().start(plan.endpoint_catalog());
  const ReplayStats& stats = graph.replayer().stats();
  ASSERT_TRUE(wait_for_pass(stats, {{kPoseTopic, recorded}}, 30s))
      << "the recording was not replayed through";
  const std::size_t delivered = stats.emitted(kPoseTopic);
  ASSERT_EQ(CountingSinkOp<isaac::PoseCpuT>::wait_for(delivered, 10s), delivered);
  // Indexed rather than taken from the back of the vector: replay does not loop by default, so
  // the two are the same message here, but indexing is what would still isolate the first pass
  // if a later test reused this shape with looping enabled.
  const auto elapsed = CountingSinkOp<isaac::PoseCpuT>::arrivals()[delivered - 1].at - began;

  graph.replayer().stop();
  session.request_stop();
  session.wait();

  // Held to between nine and eleven times speed, which is what keeps ten times speed from passing
  // as five or twenty. Targets are absolute, so no accumulated overshoot can bring a run under its
  // own compressed span; the lower bound is an eleventh under it. The upper bound is a ninth over,
  // about a tenth of a second, which start-up and a stall of the process have to fit inside.
  const auto compressed_span = facts.span / 10;
  EXPECT_GT(elapsed, compressed_span * 10 / 11)
      << "a replay faster than eleven times speed is not pacing to the recorded timeline";
  EXPECT_LT(elapsed, compressed_span * 10 / 9)
      << "a replay slower than nine times speed is not pacing to the recorded timeline";
}

// A recording whose messages come in pairs -- two milliseconds apart, then ninety-eight -- rather
// than evenly. A replay that kept the average rate but spaced its messages evenly, or rounded them
// onto any grid coarse enough to blur that pattern, would match the recorded rate exactly and
// still be off by tens of milliseconds on every other message. Only a replay that reproduces each
// recorded interval keeps the typical schedule error small.
TEST(McapReplayer, ReproducesAnUnevenRecordedCadenceRatherThanItsAverageRate) {
  constexpr std::size_t kMessages = 41;
  std::vector<std::uint64_t> log_times;
  std::uint64_t at = 0;
  for (std::size_t index = 0; index < kMessages; ++index) {
    log_times.push_back(at);
    at += index % 2 == 0 ? 2'000'000U : 98'000'000U;
  }
  const std::filesystem::path path =
      write_recording_by_hand("uneven_cadence", recorded_pose_schema(), log_times);

  CountingSinkOp<isaac::PoseCpuT>::reset();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(path, Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);
  graph.replayer().start(plan.endpoint_catalog());
  const ReplayStats& stats = graph.replayer().stats();
  ASSERT_TRUE(wait_for_pass(stats, {{kPoseTopic, kMessages}}, 30s))
      << "the recording was not replayed through";
  graph.replayer().stop();
  session.request_stop();
  session.wait();

  // Evenly spaced at the same average rate, every other message would be 48 ms out of line with
  // its neighbours, for a typical error of about 24 ms. A faithful replay measures a fraction of a
  // millisecond. A single scheduler stall delays only the messages due while it lasts -- the
  // schedule is absolute, so the run catches up after it. Here that is a pair, because the two
  // halves of a pair are only 2 ms apart: two messages of forty-one delayed by about d move the
  // typical error to roughly 0.22 d, so this bound tolerates a stall of about 45 ms. Measured
  // typical error stays under 0.2 ms even on a heavily loaded host.
  const std::optional<std::chrono::nanoseconds> typical = stats.typical_schedule_error(kPoseTopic);
  ASSERT_TRUE(typical.has_value());
  EXPECT_LT(*typical, 10ms) << "the replay kept the recorded rate but not the recorded intervals";
}

// The reader starting the recording over is the one event that crosses every part of the system
// at once: the reader wraps, and the release step, on reaching the backwards step in recorded
// time, re-bases the timing onto a new origin. Each of those is covered on its own elsewhere; only
// a live graph shows that they hand off to each other. If re-basing were lost, every message of
// the second pass would be due at an instant already long past and the whole pass would go out at
// once -- which is what the per-pass cadence below would catch and nothing else would.
TEST(McapReplayer, KeepsTheRecordedCadenceAcrossTheStartOfEachNewPass) {
  constexpr std::size_t kMessages = 10;
  constexpr auto kSpacing = 25ms;
  constexpr std::size_t kPasses = 3;

  std::vector<std::uint64_t> log_times;
  for (std::size_t index = 0; index < kMessages; ++index) {
    log_times.push_back(
        static_cast<std::uint64_t>(index) *
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(kSpacing).count()));
  }
  const std::filesystem::path path =
      write_recording_by_hand("wrap_pacing", recorded_pose_schema(), log_times);

  CountingSinkOp<isaac::PoseCpuT>::reset();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(path, ReplayOptions{.loop = true},
                                                   Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);
  graph.replayer().start(plan.endpoint_catalog());

  // One pass more than is asserted on below, as a margin: the grouping below only begins a pass at
  // a message numbered zero, so a first pass that somehow lost its opening message would not count.
  const std::size_t wanted = kMessages * (kPasses + 1);
  ASSERT_GE(CountingSinkOp<isaac::PoseCpuT>::wait_for(wanted, 30s), wanted);
  const std::vector<Arrival> arrivals = CountingSinkOp<isaac::PoseCpuT>::arrivals();

  graph.replayer().stop();
  session.request_stop();
  session.wait();

  // Each message carries its own position in the recording, so a pass begins wherever that
  // starts over rather than being counted off by arrival number.
  std::vector<std::vector<Arrival>> passes;
  for (const Arrival& arrival : arrivals) {
    if (arrival.sequence == 0) {
      passes.emplace_back();
    }
    if (!passes.empty()) {
      passes.back().push_back(arrival);
    }
  }
  ASSERT_GE(passes.size(), kPasses);

  for (std::size_t pass = 0; pass < kPasses; ++pass) {
    ASSERT_EQ(passes[pass].size(), kMessages) << "pass " << pass << " was not replayed whole";
    const auto cadence = (passes[pass].back().at - passes[pass].front().at) / (kMessages - 1);
    // Bounded loosely on both sides: the lower bound fails only if the pass was flushed rather
    // than paced, and the upper leaves room for a loaded machine.
    EXPECT_GT(cadence, kSpacing * 3 / 5)
        << "pass " << pass << " went out faster than it was recorded";
    EXPECT_LT(cadence, kSpacing * 2) << "pass " << pass << " went out slower than it was recorded";
  }

  for (std::size_t pass = 0; pass + 1 < kPasses; ++pass) {
    // The recording holds no gap between its last message and its first, so the new pass starts
    // as soon as the previous one ends rather than waiting out a recorded interval that was
    // never recorded.
    const auto across = passes[pass + 1].front().at - passes[pass].back().at;
    EXPECT_LT(across, kSpacing / 2)
        << "the first message of a new pass was paced against the pass before it";
  }

  // The replay's own account of the same passes, each measured within itself. A pass flushed at
  // once because re-basing was lost would put its last message 225 ms out of line with its first,
  // and a first message held back a whole spacing while the rest of its pass kept their times
  // would be 25 ms out of line. A loaded machine's jitter spreads a pass by a few milliseconds,
  // and 10 ms, under half the spacing, lies between.
  const std::optional<std::chrono::nanoseconds> worst =
      graph.replayer().stats().worst_schedule_error(kPoseTopic);
  ASSERT_TRUE(worst.has_value());
  std::printf("topic %s: worst error %.3f ms over the passes across the wrap\n", kPoseTopic,
              std::chrono::duration<double, std::milli>(*worst).count());
  EXPECT_LT(*worst, 10ms) << "a pass did not keep the recorded spacing across the new pass";
}

TEST(McapReplayer, ReplaysARecordingManyTimesTheBufferBoundWithoutLoss) {
  const std::filesystem::path path = pose_fixture();
  const RecordingFacts facts = read_facts(path);
  const std::size_t recorded = facts.counts.at(kPoseTopic);
  // The bound below is what makes this test mean anything: the recording has to be many times
  // what the buffer can hold, so the reader blocks and resumes repeatedly during the run.
  constexpr std::size_t kBufferMessages = 8;
  ASSERT_GT(recorded, kBufferMessages * 10);

  CountingSinkOp<isaac::PoseCpuT>::reset();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(
        path, ReplayOptions{.playback_speed = 50.0, .buffer_messages = kBufferMessages},
        Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);
  graph.replayer().start(plan.endpoint_catalog());
  const ReplayStats& stats = graph.replayer().stats();
  ASSERT_TRUE(wait_for_pass(stats, {{kPoseTopic, recorded}}, 30s))
      << "the recording was not replayed through";
  const std::size_t delivered = stats.emitted(kPoseTopic);
  ASSERT_EQ(CountingSinkOp<isaac::PoseCpuT>::wait_for(delivered, 10s), delivered);
  // The whole recording: the reader blocking on a full buffer must cost nothing, and replay does
  // not send before the consumer has subscribed, so nothing is lost to start-up either.
  EXPECT_EQ(delivered, recorded);

  const std::vector<Arrival> arrivals = CountingSinkOp<isaac::PoseCpuT>::arrivals();
  ASSERT_FALSE(arrivals.empty());
  EXPECT_EQ(first_sequences(arrivals, delivered),
            consecutive_from(arrivals.front().sequence, delivered))
      << "a buffer smaller than the recording must block the reader, never drop a message";

  graph.replayer().stop();
  session.request_stop();
  session.wait();
}

// A consumer slower than the recording must not slow replay down. Replay publishes every message
// it decodes on the recorded schedule and moves on; whether the consumer ever read a given one is
// not something replay can see, because the default kDropOldest wiring lets a publish overwrite
// an unread sample instead of waiting for room. So this checks the two things that are still
// replay's own: the run kept the recorded span, and every message of the recording was accounted
// for exactly once.
TEST(McapReplayer, StaysOnTheRecordedScheduleWhenTheConsumerIsTooSlow) {
  const std::filesystem::path path = pose_fixture();
  const RecordingFacts facts = read_facts(path);
  const std::size_t recorded_poses = facts.counts.at(kPoseTopic);
  ASSERT_GT(recorded_poses, 10U);
  const double recorded_span_s = std::chrono::duration<double>(facts.span).count();

  SlowSinkOp::reset();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(path, Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<SlowSinkOp>("slow_sink", 100ms);
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    // The consumer needs its own execution partition, or its sleep also holds up the replay side
    // and the whole premise -- replay carrying on at the recorded timing while the consumer
    // falls behind -- never arises in the first place.
    owner.partition("consumer").add(sink);
    owner.auto_partition_remaining();
    return replayer;
  });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);

  const auto started = std::chrono::steady_clock::now();
  graph.replayer().start(plan.endpoint_catalog());

  const std::map<std::string, std::size_t> expected{{kPoseTopic, recorded_poses}};
  const auto deadline = started + 30s;
  while (std::chrono::steady_clock::now() < deadline &&
         !pass_complete(graph.replayer().stats(), expected)) {
    std::this_thread::sleep_for(5ms);
  }
  const double replay_span_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  // Ending the run first is what makes the counters below a settled account of it rather than a
  // sample of one still in motion.
  graph.replayer().stop();
  const std::size_t emitted = graph.replayer().stats().emitted(kPoseTopic);
  const std::size_t refused = graph.replayer().stats().refused(kPoseTopic);
  session.request_stop();
  session.wait();

  EXPECT_LT(replay_span_s, recorded_span_s * 1.5)
      << "replay waited for the consumer instead of staying on the recorded schedule";
  // Exact, not just a lower bound: replay does not loop by default, so the recording's
  // recorded_poses messages are the only ones that will ever exist to be accounted for, and every
  // one of them ends as either published or refused -- never both, never neither.
  EXPECT_EQ(emitted + refused, recorded_poses)
      << "replay must account for every message of the recording exactly once";
  EXPECT_GT(emitted, 0U) << "a wired topic must have published something";
}

TEST(McapReplayer, StoppingTwiceEmitsTheSummaryOnlyOnce) {
  const std::filesystem::path path = pose_fixture();
  CountingSinkOp<isaac::PoseCpuT>::reset();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(path, ReplayOptions{.playback_speed = 20.0},
                                                   Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);
  graph.replayer().start(plan.endpoint_catalog());
  ASSERT_GT(CountingSinkOp<isaac::PoseCpuT>::wait_for(1, 30s), 0U)
      << "at least one message must arrive before the summary has anything to report";

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  graph.replayer().stop();
  graph.replayer().stop();
  const std::string output =
      testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();

  EXPECT_EQ(occurrences_of(output, "Topic 'pose' delivered"), 1U)
      << "a second stop must not repeat the run's summary; output was:\n"
      << output;

  session.request_stop();
  session.wait();
}

TEST(McapReplayer, ReportsWhenTheRunHasEndedAfterSendingEverythingItRead) {
  const std::filesystem::path path = pose_fixture();
  const RecordingFacts facts = read_facts(path);
  CountingSinkOp<isaac::PoseCpuT>::reset();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(path, ReplayOptions{.playback_speed = 20.0},
                                                   Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  EXPECT_FALSE(graph.replayer().finished()) << "nothing has run before start()";
  holoscan::RunSession session = holoscan::run_async(plan);
  graph.replayer().start(plan.endpoint_catalog());

  const auto deadline = std::chrono::steady_clock::now() + 30s;
  while (!graph.replayer().finished() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  ASSERT_TRUE(graph.replayer().finished());
  const ReplayStats& stats = graph.replayer().stats();
  EXPECT_EQ(
      stats.emitted(kPoseTopic) + stats.refused(kPoseTopic) + stats.decode_failures(kPoseTopic),
      facts.counts.at(kPoseTopic))
      << "the run ended before everything it read had gone out";
  EXPECT_FALSE(stats.read_failed());

  graph.replayer().stop();
  session.request_stop();
  session.wait();
}

TEST(McapReplayer, DoesNotEndWhileALoopingReplayKeepsGoing) {
  const std::filesystem::path path = pose_fixture();
  const RecordingFacts facts = read_facts(path);
  CountingSinkOp<isaac::PoseCpuT>::reset();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer =
        std::make_unique<McapReplayer>(path, ReplayOptions{.playback_speed = 20.0, .loop = true},
                                       Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);
  graph.replayer().start(plan.endpoint_catalog());

  // A second pass has begun once more messages have gone out than the recording holds.
  const std::size_t one_pass = facts.counts.at(kPoseTopic);
  const auto deadline = std::chrono::steady_clock::now() + 30s;
  while (graph.replayer().stats().emitted(kPoseTopic) <= one_pass &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  ASSERT_GT(graph.replayer().stats().emitted(kPoseTopic), one_pass);
  EXPECT_FALSE(graph.replayer().finished());

  graph.replayer().stop();
  EXPECT_TRUE(graph.replayer().finished()) << "a stopped run has ended";
  session.request_stop();
  session.wait();
}

// The schema bytes are the check that matters, and the name is not a substitute for them: a
// schema can keep its name across a version that moved its fields, which decodes into
// plausible-looking wrong values rather than failing. The recording below carries exactly the
// name the requested type has, so only the bytes can reject it.
TEST(McapReplayer, RejectsATopicRecordedUnderADifferentVersionOfTheSameSchema) {
  std::vector<std::byte> altered = recorded_pose_schema();
  ASSERT_FALSE(altered.empty());
  altered.pop_back();
  const std::filesystem::path path =
      write_recording_by_hand("altered_schema", altered, /*log_times=*/{1000, 2000});

  holoscan::Graph graph("mcap_replayer_rejection_test");
  EXPECT_THROW(McapReplayer(path, Topic<isaac::PoseCpuT>{kPoseTopic}), std::invalid_argument);
}

// Replay started well before its graph: the start-up gate holds the first message back until the
// consumer has subscribed, so nothing is lost to the delay, and the timeline starts only once the
// gate opens. What this checks is that the delay is not also charged to the recording -- if the
// graph's start-up had been counted as recorded time, everything recorded during it would already
// be overdue once the graph caught up and would flush as a burst instead of arriving at the
// recorded cadence.
TEST(McapReplayer, DoesNotChargeTheGraphsOwnStartUpDelayToTheRecordedTimeline) {
  const std::filesystem::path path = pose_fixture();
  const RecordingFacts facts = read_facts(path);
  const std::size_t recorded_poses = facts.counts.at(kPoseTopic);
  constexpr std::size_t kMessagesToTime = 40;
  // Far past what the graph needs to come up, so the gate is really what holds replay back, and
  // far inside the time the gate gives a consumer before stopping the run.
  constexpr auto kGraphStartDelay = 400ms;
  ASSERT_GT(recorded_poses, kMessagesToTime);

  CountingSinkOp<isaac::PoseCpuT>::reset();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(path, Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  graph.replayer().start(plan.endpoint_catalog());
  std::this_thread::sleep_for(kGraphStartDelay);

  const auto graph_began = std::chrono::steady_clock::now();
  holoscan::RunSession session = holoscan::run_async(plan);
  ASSERT_GE(CountingSinkOp<isaac::PoseCpuT>::wait_for(kMessagesToTime, 30s), kMessagesToTime);
  const auto elapsed =
      CountingSinkOp<isaac::PoseCpuT>::arrivals()[kMessagesToTime - 1].at - graph_began;

  graph.replayer().stop();
  const std::size_t refused = graph.replayer().stats().refused(kPoseTopic);
  session.request_stop();
  session.wait();

  EXPECT_EQ(refused, 0U) << "replay sent messages before the graph's consumer had subscribed";
  EXPECT_FALSE(graph.replayer().stats().never_subscribed(kPoseTopic));
  const std::vector<Arrival> arrivals = CountingSinkOp<isaac::PoseCpuT>::arrivals();
  EXPECT_EQ(first_sequences(arrivals, kMessagesToTime),
            consecutive_from(arrivals.front().sequence, kMessagesToTime));

  // Half of what those messages take in the recording, which is a bound a paced replay clears
  // easily and a replay that flushed its buffer cannot: the messages waited out in the 400 ms
  // above are more than these.
  EXPECT_GT(elapsed, kPosePeriod * kMessagesToTime / 2)
      << "the delay before the graph started must not have been replayed away";
  EXPECT_LT(elapsed, 10s);
}

// A topic wired to a consumer whose graph never starts is a wiring mistake, and a run that
// silently replayed nothing would look exactly like a recording with nothing on that topic. So
// the run has to stop on its own without sending anything, say which topic it was waiting on,
// and still shut down cleanly afterwards. The recording also carries a topic that is replayed
// but never attach()ed, which nothing is ever expected to subscribe to and which must therefore
// never be named.
TEST(McapReplayer, StopsWithoutSendingAnythingWhenAnAttachedTopicsConsumerNeverSubscribes) {
  const std::filesystem::path path = multi_topic_fixture();
  CountingSinkOp<isaac::PoseCpuT>::reset();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(path, Topic<isaac::PoseCpuT>{kPoseTopic},
                                                   Topic<isaac::TwistCpuT>{kTwistTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  // The graph is never run, so the consumer never subscribes.
  graph.replayer().start(plan.endpoint_catalog());
  const ReplayStats& stats = graph.replayer().stats();
  const auto deadline = std::chrono::steady_clock::now() + 30s;
  while (!stats.never_subscribed(kPoseTopic) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  const bool gave_up = stats.never_subscribed(kPoseTopic);

  // The run has already stopped its own threads, so all that is left for stop() is to join them
  // and log the summary -- which must not wait on a reading thread still filling a buffer.
  const auto stop_began = std::chrono::steady_clock::now();
  graph.replayer().stop();
  const auto stop_took = std::chrono::steady_clock::now() - stop_began;
  graph.replayer().stop();
  const std::string output =
      testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();

  ASSERT_TRUE(gave_up) << "a consumer that never subscribed must end the run; output was:\n"
                       << output;
  EXPECT_FALSE(stats.never_subscribed(kTwistTopic))
      << "a topic nobody attached is not waiting on a consumer";
  EXPECT_LT(stop_took, 1s);

  // Nothing was sent, not even into nothing: the run never began, so no message counts as refused
  // either.
  EXPECT_EQ(stats.emitted(kPoseTopic), 0U);
  EXPECT_EQ(stats.refused(kPoseTopic), 0U);
  EXPECT_EQ(stats.emitted(kTwistTopic), 0U);
  EXPECT_EQ(stats.refused(kTwistTopic), 0U);
  EXPECT_EQ(CountingSinkOp<isaac::PoseCpuT>::count(), 0U);

  const std::string error_marker = "Replay stopped without sending anything";
  ASSERT_EQ(occurrences_of(output, error_marker), 1U) << output;
  const std::string error_line = output.substr(output.find(error_marker));
  const std::string error = error_line.substr(0, error_line.find('\n'));
  EXPECT_NE(error.find("'pose'"), std::string::npos) << error;
  EXPECT_EQ(error.find("'twist'"), std::string::npos) << error;
  EXPECT_EQ(occurrences_of(output, "could not publish a replayed message"), 0U)
      << "a run that never began must not also report ordinary refusals; output was:\n"
      << output;

  // Logged once despite the second stop(), and saying the run never began rather than that the
  // recording held nothing.
  EXPECT_EQ(occurrences_of(output, "never had that consumer subscribe"), 1U) << output;
  EXPECT_EQ(occurrences_of(output, "The recording held none"), 0U) << output;
}

// A recording whose messages all carry one recorded time has no cadence to reproduce, so it
// replays as fast as its consumer accepts. Saying so once is the whole of the handling -- and
// saying it about an ordinary recording would be worse than not saying it at all, so an ordinary
// replay past the end of its recording is checked to stay silent.
TEST(McapReplayer, SaysOnceThatARecordingCarriesNoTimelineAndSaysItOfNothingElse) {
  const std::string marker = "carries the same recorded time";
  constexpr std::size_t kBufferMessages = 4;

  const std::filesystem::path flat = write_recording_by_hand(
      "flat_timeline", recorded_pose_schema(), std::vector<std::uint64_t>(20, 1000));
  CountingSinkOp<isaac::PoseCpuT>::reset();
  ReplayerTestGraph flat_graph([&flat](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(
        flat, ReplayOptions{.loop = true, .buffer_messages = kBufferMessages},
        Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });
  const holoscan::ExecutionPlan flat_plan = holoscan::compile(flat_graph);
  ASSERT_TRUE(flat_plan.ok());

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  {
    holoscan::RunSession session = holoscan::run_async(flat_plan);
    flat_graph.replayer().start(flat_plan.endpoint_catalog());
    CountingSinkOp<isaac::PoseCpuT>::wait_for(kBufferMessages * 20, 30s);
    flat_graph.replayer().stop();
    session.request_stop();
    session.wait();
  }
  const std::string flat_output =
      testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();
  EXPECT_EQ(occurrences_of(flat_output, marker), 1U)
      << "a recording without a timeline must be reported exactly once; output was:\n"
      << flat_output;

  const std::filesystem::path paced = pose_fixture();
  const RecordingFacts facts = read_facts(paced);
  const std::size_t recorded = facts.counts.at(kPoseTopic);
  CountingSinkOp<isaac::PoseCpuT>::reset();
  ReplayerTestGraph paced_graph([&paced](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(
        paced,
        ReplayOptions{.playback_speed = 100.0, .loop = true, .buffer_messages = kBufferMessages},
        Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });
  const holoscan::ExecutionPlan paced_plan = holoscan::compile(paced_graph);
  ASSERT_TRUE(paced_plan.ok());

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  {
    holoscan::RunSession session = holoscan::run_async(paced_plan);
    paced_graph.replayer().start(paced_plan.endpoint_catalog());
    // Well past one pass, so the recording has started over at least once: the moment a new pass
    // begins is where a careless check would mistake a rewind for a missing timeline.
    CountingSinkOp<isaac::PoseCpuT>::wait_for(recorded * 2, 30s);
    paced_graph.replayer().stop();
    session.request_stop();
    session.wait();
  }
  const std::string paced_output =
      testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();
  EXPECT_EQ(occurrences_of(paced_output, marker), 0U)
      << "an ordinary recording starting over is not a recording without a timeline; output "
         "was:\n"
      << paced_output;
}

// A recording whose messages cannot be read back as the type they are replayed as is the one
// way a run loses messages that nothing on the delivery side ever sees. Left uncounted, the run
// would report itself as a healthy one; left unbounded, the reading side would read a recording
// it can do nothing with at full speed for as long as the run lasted, since a buffer that never
// fills is the only thing that ever slows it down.
TEST(McapReplayer, CountsMessagesItCouldNotReadBackAndStopsRatherThanRereadingThem) {
  constexpr std::size_t kMessages = 5;
  std::vector<std::uint64_t> log_times;
  for (std::size_t index = 0; index < kMessages; ++index) {
    log_times.push_back(static_cast<std::uint64_t>(index) * 1'000'000ULL);
  }
  const std::filesystem::path path = write_recording_by_hand(
      "unreadable_payloads", recorded_pose_schema(), log_times, /*unreadable_payloads=*/true);

  CountingSinkOp<isaac::PoseCpuT>::reset();
  ReplayerTestGraph graph([&path](holoscan::Graph& owner) {
    auto replayer = std::make_unique<McapReplayer>(path, Topic<isaac::PoseCpuT>{kPoseTopic});
    auto sink = owner.op<CountingSinkOp<isaac::PoseCpuT>>("sink");
    replayer->attach<isaac::PoseCpuT>(owner, kPoseTopic, sink->in);
    return replayer;
  });
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  {
    const ScopedLogLevel suppress_expected_warnings(holoscan::LogLevel::ERROR);
    graph.replayer().start(plan.endpoint_catalog());

    // The reading side gives up once a whole pass has produced nothing to send, so this settles
    // rather than climbing; the bound is on how long that takes, not on how far it gets.
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (graph.replayer().stats().decode_failures(kPoseTopic) < kMessages &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
    graph.replayer().stop();
  }

  const ReplayStats& stats = graph.replayer().stats();
  EXPECT_GE(stats.decode_failures(kPoseTopic), kMessages);
  EXPECT_EQ(stats.emitted(kPoseTopic), 0U);
  EXPECT_EQ(CountingSinkOp<isaac::PoseCpuT>::count(), 0U);

  bool says_so = false;
  bool claims_health = false;
  for (const std::string& line : stats.summary_lines()) {
    says_so = says_so || line.find("could not be read back") != std::string::npos;
    claims_health = claims_health || line.find("on schedule") != std::string::npos;
  }
  EXPECT_TRUE(says_so) << "a run that lost every message must say so in its summary";
  EXPECT_FALSE(claims_health) << "a run that delivered nothing must not report itself as healthy";
}

TEST(McapReplayer, RejectsOptionsThatCannotDescribeAReplay) {
  holoscan::Graph graph("mcap_replayer_options_test");
  const std::filesystem::path path = pose_fixture();

  EXPECT_THROW(
      McapReplayer(path, ReplayOptions{.playback_speed = 0.0}, Topic<isaac::PoseCpuT>{kPoseTopic}),
      std::invalid_argument);
  EXPECT_THROW(
      McapReplayer(path, ReplayOptions{.playback_speed = -1.0}, Topic<isaac::PoseCpuT>{kPoseTopic}),
      std::invalid_argument);
  EXPECT_THROW(
      McapReplayer(path, ReplayOptions{.buffer_messages = 0}, Topic<isaac::PoseCpuT>{kPoseTopic}),
      std::invalid_argument);
}

TEST(McapReplayer, ReportsAnErrorNamingTheRecordingItCannotOpen) {
  holoscan::Graph graph("mcap_replayer_missing_file_test");
  try {
    McapReplayer replayer("/nonexistent/none.mcap", Topic<isaac::PoseCpuT>{kPoseTopic});
    FAIL() << "expected a recording that cannot be opened to be reported";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("none.mcap"), std::string::npos) << error.what();
  }
}

}  // namespace isaac::os::mcap_replayer
