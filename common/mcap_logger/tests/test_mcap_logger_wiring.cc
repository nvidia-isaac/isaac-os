/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/schemas/tensor_packet_schema_traits.hpp"
#include "holoscan/sensor_io/sensor_schema_package.hpp"
#include "holoscan/time/realtime_clock.hpp"

#include "common/image_schema/image_schema.h"
#include "common/mcap_logger/mcap_logger.h"
#include "common/mcap_logger/tests/scoped_temp_directory.h"
#include "joint_command_cpu_schema_traits.h"

namespace isaac::os::mcap_logger {
namespace {

// Starts a logger whose graph has a video-encoded topic on CUDA device 0, and checks the outcome
// this host should produce. start() is where the encoder is built, and it throws where the host
// has no usable hardware encoder rather than recording raw tensors in place of the video that was
// asked for. The tests that use this are about wiring, not about encoding, so they have to pass on
// a host without an encoder too -- a Jetson Orin, say -- but not by ignoring what start() does
// there: each host must produce the outcome its hardware dictates.
void ExpectStartMatchesEncoderAvailability(HsdkMcapLogger& logger,
                                           const holoscan::ChannelEndpointCatalog& catalog) {
  if (FrameEncoder::hardware_available(0)) {
    EXPECT_NO_THROW(logger.start(catalog));
  } else {
    EXPECT_THROW(logger.start(catalog), std::runtime_error);
  }
  // A start() that threw has still left a logger that can be shut down, so that a caller that
  // catches the failure does not leak its poll threads.
  EXPECT_NO_THROW(logger.shutdown());
}

// Asserts that `action` throws std::invalid_argument whose message contains
// `expected_substring`. Which message a rejection carries is the point of these tests and not a
// detail: colliding with an ordinary topic and colliding with a video-encoded topic's metadata
// sidecar have different fixes, so merging the two messages -- or routing one collision to the
// other's text -- has to fail here rather than pass as "some invalid_argument".
template <typename Action>
void ExpectInvalidArgumentContaining(Action&& action, std::string_view expected_substring) {
  try {
    std::forward<Action>(action)();
    ADD_FAILURE() << "Expected std::invalid_argument mentioning \"" << expected_substring << "\".";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find(expected_substring), std::string::npos)
        << "Thrown message: " << error.what();
  }
}

// Exists only to give a graph output port of a real, schema_package<T>()-backed type for
// attach<T>() to wire onto -- compile() and start() are what this test exercises, not
// runtime data flow, but every operator still needs a valid contract() to be schedulable at
// all (an output-only operator with no contract() fails to compile as a graph). Matches the
// SourceOp pattern in common/ros2_bridge/tests/test_ros2_bridge_wiring.cc.
class StubSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(command_out, "command_out"); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(20)});
    return contract;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return command_out.emit(isaac::JointCommandCpuT{});
  }
  holoscan::Output<isaac::JointCommandCpuT> command_out;
};

class WiringTestGraph final : public holoscan::Graph {
 public:
  explicit WiringTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_wiring_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto source = op<StubSourceOp>("source");
    logger_.attach<isaac::JointCommandCpuT>(*this, "joint_command", source->command_out);
  }

 private:
  HsdkMcapLogger& logger_;
};

TEST(HsdkMcapLoggerWiringTest, AttachedTopicCompilesAndStartsCleanly) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_wiring_test.mcap");

  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
  WiringTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  EXPECT_NO_THROW(logger.start(plan.endpoint_catalog()));
  EXPECT_NO_THROW(logger.shutdown());
}

TEST(HsdkMcapLoggerWiringTest, StartTwiceThrows) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_wiring_test2.mcap");

  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
  WiringTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  logger.start(plan.endpoint_catalog());
  EXPECT_THROW(logger.start(plan.endpoint_catalog()), std::runtime_error);
  logger.shutdown();
}

// Two independent source ports attached under the same port_name -- attach() must reject the
// second one rather than silently overwriting the first's starter in starters_, which would
// leave the first topic's poller never created and the first attach() call raising no error at
// all.
class DuplicatePortNameTestGraph final : public holoscan::Graph {
 public:
  explicit DuplicatePortNameTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_duplicate_port_name_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto source1 = op<StubSourceOp>("source1");
    auto source2 = op<StubSourceOp>("source2");
    logger_.attach<isaac::JointCommandCpuT>(*this, "joint_command", source1->command_out);
    ExpectInvalidArgumentContaining(
        [&] {
          logger_.attach<isaac::JointCommandCpuT>(*this, "joint_command", source2->command_out);
        },
        "already has a topic attached under port_name");
  }

 private:
  HsdkMcapLogger& logger_;
};

TEST(HsdkMcapLoggerWiringTest, AttachingDuplicatePortNameThrows) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_wiring_test_duplicate.mcap");

  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
  DuplicatePortNameTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  EXPECT_NO_THROW(logger.start(plan.endpoint_catalog()));
  EXPECT_NO_THROW(logger.shutdown());
}

TEST(HsdkMcapLoggerWiringTest, StartAfterShutdownThrows) {
  // shutdown() is terminal: it must not be possible to start() again afterward. Before this
  // fix, start() only checked whether the runtime existed, and shutdown() reset the runtime --
  // so calling start() after shutdown() would silently succeed and spawn pollers that submit
  // to writers that have already stopped.
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_wiring_test3.mcap");

  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
  WiringTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  logger.start(plan.endpoint_catalog());
  logger.shutdown();
  EXPECT_THROW(logger.start(plan.endpoint_catalog()), std::runtime_error);
}

TEST(HsdkMcapLoggerWiringTest, ShutdownWithoutStartIsANoOp) {
  ScopedTempDirectory temp_directory;
  HsdkMcapLogger logger(HsdkMcapLoggerOptions{
      .output_path = temp_directory.path("mcap_logger_wiring_test_unused.mcap"),
      .timestamp_output_path = false});
  EXPECT_NO_THROW(logger.shutdown());
}

TEST(HsdkMcapLoggerWiringTest, TimestampOutputPathDefaultsToAGeneratedFileInsideTheGivenDirectory) {
  // timestamp_output_path defaults to true (left unset here), so output_path below is a
  // directory, not the literal file to write -- unlike every other test in this file, which
  // opts out via .timestamp_output_path = false to get an exact, predictable path.
  ScopedTempDirectory temp_directory;
  const auto& output_dir = temp_directory.path();
  HsdkMcapLogger logger(HsdkMcapLoggerOptions{.output_path = output_dir});

  ASSERT_TRUE(logger.output_path().has_value());
  const std::filesystem::path& resolved = *logger.output_path();
  EXPECT_EQ(resolved.parent_path(), output_dir);
  EXPECT_EQ(resolved.extension(), ".mcap");
  EXPECT_TRUE(resolved.filename().string().starts_with("mcap_log_"));

  logger.shutdown();
  EXPECT_TRUE(std::filesystem::exists(resolved));
}

// Never actually runs compute(); only exists to give a graph output port of a real
// tensor-schema type for attach<T>(..., contract) to wire onto.
class StubImageSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(image_out, "image_out"); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(20)});
    return contract;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
  holoscan::Output<holoscan::schema::ImageT> image_out;
};

class TensorSchemaWiringTestGraph final : public holoscan::Graph {
 public:
  explicit TensorSchemaWiringTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_tensor_schema_wiring_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto source = op<StubImageSourceOp>("source");
    logger_.attach<holoscan::schema::ImageT>(
        *this, "camera", source->image_out,
        image_schema::ImageTensorFieldContract(1920, 1080, holoscan::schema::ImageEncoding_RGB8));
  }

 private:
  HsdkMcapLogger& logger_;
};

TEST(HsdkMcapLoggerWiringTest, AttachedTensorSchemaTopicCompilesAndStartsCleanly) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_tensor_schema_wiring_test.mcap");

  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
  TensorSchemaWiringTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  EXPECT_NO_THROW(logger.start(plan.endpoint_catalog()));
  EXPECT_NO_THROW(logger.shutdown());
}

// Same wiring as TensorSchemaWiringTestGraph, but opting in to hardware video encoding. The
// encoder itself is only created at start(), so this compiles and starts identically on a host
// with no video-encoding device -- there it just records raw tensors and warns once.
class EncodedTensorSchemaWiringTestGraph final : public holoscan::Graph {
 public:
  explicit EncodedTensorSchemaWiringTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_encoded_tensor_schema_wiring_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto source = op<StubImageSourceOp>("source");
    logger_.attach<holoscan::schema::ImageT>(
        *this, "camera", source->image_out,
        image_schema::ImageTensorFieldContract(1920, 1080, holoscan::schema::ImageEncoding_RGB8),
        TopicLogOptions{.encode = VideoEncodeOptions{}});
  }

 private:
  HsdkMcapLogger& logger_;
};

TEST(HsdkMcapLoggerWiringTest, AttachedEncodedTensorSchemaTopicCompilesAndStartsCleanly) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_encoded_wiring_test.mcap");

  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
  EncodedTensorSchemaWiringTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  ExpectStartMatchesEncoderAvailability(logger, plan.endpoint_catalog());
}

// An encoded topic records its image metadata on a "<topic>/metadata" channel, so that name is
// no longer free for an ordinary topic. Attaching one there anyway would put two schemas on one
// channel -- a channel takes its schema from whichever write lands first and keeps it for life --
// so the later writer's messages would be recorded under the other's schema and read back as
// garbage, with nothing logged. Both orders have to be rejected, since either attach() can come
// first.
class EncodedThenMetadataNameTestGraph final : public holoscan::Graph {
 public:
  explicit EncodedThenMetadataNameTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_encoded_then_metadata_name_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto image_source = op<StubImageSourceOp>("image_source");
    auto source = op<StubSourceOp>("source");
    logger_.attach<holoscan::schema::ImageT>(
        *this, "camera", image_source->image_out,
        image_schema::ImageTensorFieldContract(1920, 1080, holoscan::schema::ImageEncoding_RGB8),
        TopicLogOptions{.encode = VideoEncodeOptions{}});
    ExpectInvalidArgumentContaining(
        [&] {
          logger_.attach<isaac::JointCommandCpuT>(*this, "camera/metadata", source->command_out);
        },
        "already records the image metadata of a video-encoded");
  }

 private:
  HsdkMcapLogger& logger_;
};

TEST(HsdkMcapLoggerWiringTest, AttachingOverAnEncodedTopicsMetadataChannelThrows) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_encoded_then_metadata_name.mcap");

  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
  EncodedThenMetadataNameTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  ExpectStartMatchesEncoderAvailability(logger, plan.endpoint_catalog());
}

// Runs on every host, including one with no encoder: an ordinal far past any GPU this host could
// have is absent everywhere, so start() has to refuse it everywhere. The integration test that
// needs a real encoder can tell a refusal caused by the device from one caused by the host; this
// one pins only that the refusal happens, names the topic and device, and leaves a logger that
// can still be shut down -- which is also what the no-encoder branch of the helper above relies on.
class EncodedOnAbsentDeviceTestGraph final : public holoscan::Graph {
 public:
  explicit EncodedOnAbsentDeviceTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_encoded_absent_device_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto source = op<StubImageSourceOp>("source");
    logger_.attach<holoscan::schema::ImageT>(
        *this, "camera", source->image_out,
        image_schema::ImageTensorFieldContract(1920, 1080, holoscan::schema::ImageEncoding_RGB8),
        TopicLogOptions{.encode = VideoEncodeOptions{.cuda_device = kAbsentDevice}});
  }

 private:
  static constexpr std::uint32_t kAbsentDevice = 1024;
  HsdkMcapLogger& logger_;
};

TEST(HsdkMcapLoggerWiringTest, StartThrowsForAnEncodedTopicOnADeviceNoHostHas) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_encoded_absent_device.mcap");

  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
  EncodedOnAbsentDeviceTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  try {
    logger.start(plan.endpoint_catalog());
    ADD_FAILURE() << "start() accepted a topic on a GPU that does not exist.";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("topic 'camera'"), std::string::npos) << message;
    EXPECT_NE(message.find("GPU 1024"), std::string::npos) << message;
  }
  EXPECT_NO_THROW(logger.shutdown());
}

class MetadataNameThenEncodedTestGraph final : public holoscan::Graph {
 public:
  explicit MetadataNameThenEncodedTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_metadata_name_then_encoded_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto source = op<StubSourceOp>("source");
    auto image_source = op<StubImageSourceOp>("image_source");
    logger_.attach<isaac::JointCommandCpuT>(*this, "camera/metadata", source->command_out);
    ExpectInvalidArgumentContaining(
        [&] {
          logger_.attach<holoscan::schema::ImageT>(
              *this, "camera", image_source->image_out,
              image_schema::ImageTensorFieldContract(1920, 1080,
                                                     holoscan::schema::ImageEncoding_RGB8),
              TopicLogOptions{.encode = VideoEncodeOptions{}});
        },
        "it records its image metadata under");
  }

 private:
  HsdkMcapLogger& logger_;
};

TEST(HsdkMcapLoggerWiringTest, EncodingATopicWhoseMetadataChannelIsAlreadyAttachedThrows) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_metadata_name_then_encoded.mcap");

  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
  MetadataNameThenEncodedTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  // The rejected attach() leaves nothing behind: only the first topic starts.
  EXPECT_NO_THROW(logger.start(plan.endpoint_catalog()));
  EXPECT_NO_THROW(logger.shutdown());
}

// Video encoding needs a frame's width, height and encoding, and only holoscan.schema.Image
// carries them -- the transported tensor is a flat byte buffer with no shape of its own. Every
// other tensor-schema type has to be rejected rather than encoded against guessed dimensions.
// holoscan.schema.TensorPacket is such a type: it transports a descriptor Tensor exactly like
// ImageT does, and describes it with a sequence number, a source name and a scale.
class StubTensorPacketSourceOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(packet_out, "packet_out"); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = std::chrono::milliseconds(20)});
    return contract;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
  holoscan::Output<holoscan::schema::TensorPacketT> packet_out;
};

// The ready-made contract helper is holoscan.schema.Image-specific, so this names the
// TensorPacket `data` field (tensor_packet.fbs's `data: Tensor (id: 3)`) itself: one contiguous
// uint8 buffer, the same shape of contract ImageTensorFieldContract() builds for an image.
[[nodiscard]] holoscan::DescriptorTensorTopicContract<holoscan::schema::TensorPacketT>
TensorPacketTensorFieldContract(std::uint64_t max_tensor_bytes) {
  holoscan::DescriptorTensorTopicContract<holoscan::schema::TensorPacketT> result;
  result.field(
      holoscan::TensorFieldSelector<holoscan::schema::TensorPacketT>{.field_id = 3U,
                                                                     .tensor_index = 0U},
      holoscan::TensorBoundaryContract{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = holoscan::MemoryKind::kHost,
                  .dtype = DLDataType{kDLUInt, 8U, 1U},
                  .rank = 1U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(max_tensor_bytes),
          .transfer =
              holoscan::TensorTransferPolicy{
                  .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
              },
      });
  return result;
}

class EncodedNonImageTensorSchemaTestGraph final : public holoscan::Graph {
 public:
  explicit EncodedNonImageTensorSchemaTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_encoded_non_image_tensor_schema_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto source = op<StubTensorPacketSourceOp>("source");
    ExpectInvalidArgumentContaining(
        [&] {
          logger_.attach<holoscan::schema::TensorPacketT>(
              *this, "packets", source->packet_out,
              TensorPacketTensorFieldContract(1920U * 1080U * 3U),
              TopicLogOptions{.encode = VideoEncodeOptions{}});
        },
        "supported for camera images only");
  }

 private:
  HsdkMcapLogger& logger_;
};

TEST(HsdkMcapLoggerWiringTest, EncodingANonImageTensorSchemaTopicThrows) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_encoded_non_image.mcap");

  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
  EncodedNonImageTensorSchemaTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  // The rejected attach() leaves nothing behind: no starter to replay, no name claimed.
  EXPECT_NO_THROW(logger.start(plan.endpoint_catalog()));
  EXPECT_NO_THROW(logger.shutdown());
}

// The logger owns the edge into each exported topic: the oldest sample is dropped past the topic's
// depth. Everything else the caller put on the edge is kept.
TEST(TopicLogOptionsTest, EdgeOptionsDropTheOldestAtTheTopicsDepth) {
  holoscan::ConnectionOptions requested;
  requested.queue_policy = holoscan::QueuePolicy::kBlocking;
  requested.queue_depth = 3U;
  requested.topic_name = "alias";

  const auto edge = EdgeOptionsFor(TopicLogOptions{.queue_depth = 24}, requested);

  EXPECT_EQ(edge.queue_policy, holoscan::QueuePolicy::kDropOldest);
  EXPECT_EQ(edge.queue_depth, 24U);
  EXPECT_EQ(edge.topic_name, "alias");
}

TEST(TopicLogOptionsTest, TheDefaultDepthIs16) { EXPECT_EQ(TopicLogOptions{}.queue_depth, 16U); }

class EdgePolicyTestGraph final : public holoscan::Graph {
 public:
  explicit EdgePolicyTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_edge_policy_test"), logger_(logger) {}

 private:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto source = op<StubSourceOp>("source");
    logger_.attach<isaac::JointCommandCpuT>(*this, "joint_command", source->command_out,
                                            TopicLogOptions{.queue_depth = 7});
  }

  HsdkMcapLogger& logger_;
};

// The compiled plan, not just the helper, carries a drop-oldest edge of the stated depth: a
// caller-supplied blocking edge cannot leave the logger able to back-pressure the producer.
TEST(HsdkMcapLoggerWiringTest, AttachWiresADropOldestEdgeOfTheTopicsDepth) {
  ScopedTempDirectory temp_directory;
  HsdkMcapLogger logger(HsdkMcapLoggerOptions{
      .output_path = temp_directory.path("edge_policy.mcap"), .timestamp_output_path = false});
  EdgePolicyTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  const auto edges = plan.payload_edges();
  ASSERT_FALSE(edges.empty());
  for (const auto& edge : edges) {
    EXPECT_EQ(edge.queue_policy, holoscan::QueuePolicy::kDropOldest);
    EXPECT_EQ(edge.queue_depth, 7U);
  }
  EXPECT_NO_THROW(logger.shutdown());
}

class TensorEdgePolicyTestGraph final : public holoscan::Graph {
 public:
  explicit TensorEdgePolicyTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_tensor_edge_policy_test"), logger_(logger) {}

 private:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto source = op<StubImageSourceOp>("source");
    logger_.attach<holoscan::schema::ImageT>(
        *this, "camera", source->image_out,
        image_schema::ImageTensorFieldContract(1920, 1080, holoscan::schema::ImageEncoding_RGB8),
        TopicLogOptions{.queue_depth = 7});
  }

  HsdkMcapLogger& logger_;
};

// The tensor-schema attach overload builds its edge separately from the plain one, so it is
// checked on its own.
TEST(HsdkMcapLoggerWiringTest, TensorAttachWiresADropOldestEdgeOfTheTopicsDepth) {
  ScopedTempDirectory temp_directory;
  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = temp_directory.path("tensor_edge_policy.mcap"),
                            .timestamp_output_path = false});
  TensorEdgePolicyTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());

  const auto edges = plan.payload_edges();
  ASSERT_FALSE(edges.empty());
  for (const auto& edge : edges) {
    EXPECT_EQ(edge.queue_policy, holoscan::QueuePolicy::kDropOldest);
    EXPECT_EQ(edge.queue_depth, 7U);
  }
  EXPECT_NO_THROW(logger.shutdown());
}

// Every way attach() refuses a topic's options, in one compose so each is checked against a logger
// that has accepted none of the others: a rejected attach() must leave nothing behind.
class RejectedTopicOptionsTestGraph final : public holoscan::Graph {
 public:
  explicit RejectedTopicOptionsTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_rejected_topic_options_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto source = op<StubSourceOp>("source");
    auto image_source = op<StubImageSourceOp>("image_source");
    const auto image_contract =
        image_schema::ImageTensorFieldContract(640, 480, holoscan::schema::ImageEncoding_RGB8);
    // A depth of zero would hold no sample at all.
    ExpectInvalidArgumentContaining(
        [&] {
          logger_.attach<isaac::JointCommandCpuT>(*this, "zero", source->command_out,
                                                  TopicLogOptions{.queue_depth = 0});
        },
        "queue depth is 0");
    ExpectInvalidArgumentContaining(
        [&] {
          logger_.attach<holoscan::schema::ImageT>(*this, "zero_tensor", image_source->image_out,
                                                   image_contract,
                                                   TopicLogOptions{.queue_depth = 0});
        },
        "queue depth is 0");
    // Video encoding is for tensor-schema image topics; a plain topic has no frame to encode.
    ExpectInvalidArgumentContaining(
        [&] {
          logger_.attach<isaac::JointCommandCpuT>(*this, "plain_video", source->command_out,
                                                  TopicLogOptions{.encode = VideoEncodeOptions{}});
        },
        "supported for camera images only");
    // The rejected attaches above left nothing on the port, so a valid one still succeeds.
    EXPECT_NO_THROW(logger_.attach<isaac::JointCommandCpuT>(
        *this, "deep_plain", source->command_out, TopicLogOptions{.queue_depth = 1024}));
  }

 private:
  HsdkMcapLogger& logger_;
};

TEST(HsdkMcapLoggerWiringTest, AttachRejectsTopicOptionsNoTopicCouldRunWith) {
  ScopedTempDirectory temp_directory;
  HsdkMcapLogger logger(
      HsdkMcapLoggerOptions{.output_path = temp_directory.path("rejected_topic_options.mcap"),
                            .timestamp_output_path = false});
  RejectedTopicOptionsTestGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  EXPECT_NO_THROW(logger.start(plan.endpoint_catalog()));
  EXPECT_NO_THROW(logger.shutdown());
}

TEST(HsdkMcapLoggerWiringTest, OutputPathIsNulloptWhenFileWritingIsDisabled) {
  HsdkMcapLogger logger(HsdkMcapLoggerOptions{
      .enable_write_file = false, .enable_stream = true, .websocket_port = 0});
  EXPECT_FALSE(logger.output_path().has_value());
  logger.shutdown();
}

TEST(HsdkMcapLoggerWiringTest, ConstructorThrowsWhenNeitherFileNorStreamIsEnabled) {
  EXPECT_THROW(
      HsdkMcapLogger(HsdkMcapLoggerOptions{.enable_write_file = false, .enable_stream = false}),
      std::runtime_error);
}

TEST(HsdkMcapLoggerWiringTest, StreamingCanBeEnabledAlongsideFileWriting) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_wiring_test_streaming.mcap");

  HsdkMcapLogger logger(HsdkMcapLoggerOptions{.output_path = output_path,
                                              .timestamp_output_path = false,
                                              .enable_stream = true,
                                              .websocket_port = 0});
  ASSERT_TRUE(logger.output_path().has_value());
  EXPECT_EQ(*logger.output_path(), std::filesystem::absolute(output_path).lexically_normal());
  logger.shutdown();
}

}  // namespace
}  // namespace isaac::os::mcap_logger
