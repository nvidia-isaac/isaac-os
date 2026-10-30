/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/diagnostic.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/lifecycle.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/readiness_source.hpp"
#include "holoscan/core/run.hpp"
#include "holoscan/memory/tensor_storage.hpp"
#include "holoscan/sensor_io/image_metadata.hpp"
#include "holoscan/sensor_io/sensor_schema_package.hpp"
#include "holoscan/time/realtime_clock.hpp"
#include <cuda_runtime.h>
#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/reflection.h>
#include <flatbuffers/reflection_generated.h>

#include "common/image_schema/image_schema.h"
#include "common/mcap_logger/mcap_logger.h"
#include "common/mcap_logger/tests/post_with_retry.h"
#include "common/mcap_logger/tests/read_in_log_time_order.h"
#include "common/mcap_logger/tests/scoped_temp_directory.h"
#include "mcap/reader.hpp"
#include "tensor_frame_generated.h"

namespace isaac::os::mcap_logger {
namespace {

namespace wire = ::isaac::mcap_logger;

using isaac::os::mcap_logger::testing::post_with_retry;

constexpr int kMessageCount = 5;
constexpr std::uint32_t kWidth = 4;
constexpr std::uint32_t kHeight = 2;
constexpr std::size_t kFrameBytes = kWidth * kHeight * 3U;  // RGB8: 3 bytes/pixel.
// Interval between posted notifications -- same reasoning as
// test_mcap_logger_integration.cc's kGeneratorPeriod: TensorTopicPoller waits for samples on
// its own thread and wakes as soon as one is posted, far faster than this period, which is what
// makes posting-then-immediately-checking-content deterministic rather than racing an overwrite of
// an unread sample.
constexpr auto kNotificationPeriod = std::chrono::milliseconds(20);

// Same GeneratorTriggerProbe pattern as test_mcap_logger_integration.cc: publishes the running
// generator operator's NotificationSender out to the test, since holoscan::compile()
// reconstructs operators into its own immutable canonical Graph rather than reusing compose()'s
// objects -- this is the one channel that reliably crosses that boundary.
class GeneratorTriggerProbe {
 public:
  static void publish(holoscan::NotificationSender sender) {
    const std::lock_guard<std::mutex> lock(mutex_);
    sender_ = std::move(sender);
    cv_.notify_all();
  }

  [[nodiscard]] static holoscan::NotificationSender wait_for_sender() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [] { return sender_.has_value(); });
    return *sender_;
  }

  static void reset() {
    const std::lock_guard<std::mutex> lock(mutex_);
    sender_.reset();
  }

 private:
  static std::mutex mutex_;
  static std::condition_variable cv_;
  static std::optional<holoscan::NotificationSender> sender_;
};

std::mutex GeneratorTriggerProbe::mutex_;
std::condition_variable GeneratorTriggerProbe::cv_;
std::optional<holoscan::NotificationSender> GeneratorTriggerProbe::sender_;

// Emits one kWidth x kHeight RGB8 ImageT frame (each byte = sequence_, so the test can verify
// exact content) each time its NotificationSource is posted to.
class ImageGeneratorOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(image_out, "image_out");
    spec.notification_source(trigger_, "trigger").capacity(kMessageCount);
    spec.lifecycle().stage(holoscan::LifecycleStage::kArm, &ImageGeneratorOp::arm);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnNotified{.event = trigger_});
    return contract;
  }

  holoscan::LifecycleStatus arm(holoscan::LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(trigger_);
    if (!sender) {
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    GeneratorTriggerProbe::publish(std::move(*sender));
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    // ImageT::data is a std::unique_ptr<holoscan::Tensor> (confirmed against
    // DescriptorTensorSchemaNativeTraits<schema::ImageT>::tensor(), which returns
    // value.data.get() -- holoscan/sensor_io/sensor_schema_package.hpp). Build the Tensor by
    // wrapping a heap buffer this Tensor exclusively owns via a release function -- the
    // simplest of make_tensor()'s overloads (holoscan/memory/tensor_storage.hpp) for a value
    // with no external retention requirement (no ROS2/rclcpp message to keep alive, unlike
    // common/ros2_bridge/tensor_converter.h's TensorMemoryReference-based approach, which
    // exists specifically to keep a *borrowed* rclcpp buffer alive -- not needed here since
    // this buffer is allocated for, and owned exclusively by, this one Tensor).
    auto* pixels = new std::uint8_t[kFrameBytes];
    std::memset(pixels, static_cast<int>(sequence_), kFrameBytes);

    const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(kFrameBytes)};
    auto tensor = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                                        holoscan::MemoryStorageType::kSystem, pixels,
                                        [](void* ptr) -> holoscan::expected<void, holoscan::Error> {
                                          delete[] static_cast<std::uint8_t*>(ptr);
                                          return {};
                                        });
    if (!tensor) {
      delete[] pixels;
      return holoscan::make_unexpected(tensor.error());
    }

    holoscan::schema::ImageT value;
    value.height = kHeight;
    value.width = kWidth;
    value.encoding = holoscan::schema::ImageEncoding_RGB8;
    value.data = std::make_unique<holoscan::Tensor>(std::move(*tensor));

    ++sequence_;
    return image_out.emit(value);
  }

  holoscan::Output<holoscan::schema::ImageT> image_out;

 private:
  holoscan::NotificationSource trigger_;
  int sequence_ = 0;
};

// This is a macro because GTEST_SKIP() returns from the enclosing function. A helper would only
// return from itself, leaving the calling test to run and mask the skip with failures.
#define SKIP_IF_NO_CUDA_DEVICE()                                                 \
  do {                                                                           \
    int device_count = 0;                                                        \
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) { \
      GTEST_SKIP() << "No CUDA device available";                                \
    }                                                                            \
  } while (0)

// holoscan::make_tensor()'s plain pointer+release_function overload (what ImageGeneratorOp above
// uses for host memory) wraps the pointer in a TensorMemoryReference that does not track a
// readiness stream -- fine for host memory, but the tensor-schema transport refuses to stage any
// CUDA-resident Tensor whose memory reference can't report one (see
// TensorMemoryReference::set_readiness_stream()'s doc comment on producer-to-consumer CUDA
// synchronization). This minimal override records whichever stream the producer sets and answers
// has_readiness_stream_contract() with true so the transport accepts it.
class CudaBufferMemoryReference final : public holoscan::TensorMemoryReference {
 public:
  explicit CudaBufferMemoryReference(void* pointer) : pointer_(pointer) {}
  ~CudaBufferMemoryReference() override { cudaFree(pointer_); }

  bool set_readiness_stream(cudaStream_t stream) override {
    readiness_stream_ = stream;
    return true;
  }
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }
  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override {
    return readiness_stream_;
  }

  // Also required: the transport calls this on the retained source Tensor after enqueuing a
  // device-to-device (or device-to-host) copy on its own stream, so this buffer isn't reused or
  // freed until that copy actually completes. This reference has no pool to defer reuse in, so
  // it just remembers the stream; cudaFree() below still only runs once, from the destructor,
  // after every retaining Tensor handle (including the transport's copy) has released.
  bool set_deallocation_stream(cudaStream_t stream) override {
    deallocation_stream_ = stream;
    return true;
  }

 private:
  void* pointer_;
  cudaStream_t readiness_stream_ = nullptr;
  cudaStream_t deallocation_stream_ = nullptr;
};

// Device-memory counterpart to ImageGeneratorOp above: same frame content and pacing, but the
// emitted Tensor is real CUDA device memory (cudaMalloc + cudaMemcpy host-to-device), so a graph
// using this operator exercises TensorTopicPoller's kDLCUDA materialization branch (cudaMemcpy
// device-to-host) instead of the kDLCPU/kDLCUDAHost std::memcpy branch every other test in this
// file exercises.
class CudaImageGeneratorOp final : public holoscan::Operator<> {
 public:
  ~CudaImageGeneratorOp() override {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(image_out, "image_out");
    spec.notification_source(trigger_, "trigger").capacity(kMessageCount);
    spec.lifecycle().stage(holoscan::LifecycleStage::kArm, &CudaImageGeneratorOp::arm);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnNotified{.event = trigger_});
    return contract;
  }

  holoscan::LifecycleStatus arm(holoscan::LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(trigger_);
    if (!sender) {
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    GeneratorTriggerProbe::publish(std::move(*sender));
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    // Lazily created, then reused for the operator's lifetime. The transport requires a
    // non-null "readiness stream" on every CUDA-resident Tensor it stages (see
    // Tensor::set_readiness_stream()'s doc comment: it's what producer-to-consumer
    // synchronization is keyed on), even though the cudaMemcpy below is synchronous and the
    // data is already ready by the time set_readiness_stream() is called.
    if (stream_ == nullptr) {
      if (cudaStreamCreate(&stream_) != cudaSuccess) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kResourceExhausted, "cudaStreamCreate failed."});
      }
    }

    std::array<std::uint8_t, kFrameBytes> host_pixels;
    host_pixels.fill(static_cast<std::uint8_t>(sequence_));

    void* device_pixels = nullptr;
    if (cudaMalloc(&device_pixels, kFrameBytes) != cudaSuccess) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kResourceExhausted, "cudaMalloc failed."});
    }
    if (cudaMemcpy(device_pixels, host_pixels.data(), kFrameBytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      cudaFree(device_pixels);
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "cudaMemcpy (H2D) failed."});
    }

    const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(kFrameBytes)};
    auto memory_reference = std::make_shared<CudaBufferMemoryReference>(device_pixels);
    auto tensor = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                                        holoscan::MemoryStorageType::kDevice, device_pixels,
                                        memory_reference);
    if (!tensor) {
      return holoscan::make_unexpected(tensor.error());
    }
    tensor->set_readiness_stream(stream_);

    holoscan::schema::ImageT value;
    value.height = kHeight;
    value.width = kWidth;
    value.encoding = holoscan::schema::ImageEncoding_RGB8;
    value.data = std::make_unique<holoscan::Tensor>(std::move(*tensor));

    ++sequence_;
    return image_out.emit(value);
  }

  holoscan::Output<holoscan::schema::ImageT> image_out;

 private:
  holoscan::NotificationSource trigger_;
  int sequence_ = 0;
  cudaStream_t stream_ = nullptr;
};

class TensorSchemaIntegrationTestGraph final : public holoscan::Graph {
 public:
  explicit TensorSchemaIntegrationTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_tensor_schema_integration_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto generator = op<ImageGeneratorOp>("generator");
    logger_.attach<holoscan::schema::ImageT>(
        *this, "camera", generator->image_out,
        image_schema::ImageTensorFieldContract(kWidth, kHeight,
                                               holoscan::schema::ImageEncoding_RGB8));
  }

 private:
  HsdkMcapLogger& logger_;
};

// CUDA-device-memory counterpart to TensorSchemaIntegrationTestGraph. memory_kind must be
// kCudaDevice, not the default kHost, or the transport would copy CudaImageGeneratorOp's
// device-resident Tensor to host before TensorTopicPoller ever sees it, defeating the point.
class CudaTensorSchemaIntegrationTestGraph final : public holoscan::Graph {
 public:
  explicit CudaTensorSchemaIntegrationTestGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_tensor_schema_cuda_integration_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    auto generator = op<CudaImageGeneratorOp>("generator");
    logger_.attach<holoscan::schema::ImageT>(
        *this, "camera", generator->image_out,
        image_schema::ImageTensorFieldContract(kWidth, kHeight,
                                               holoscan::schema::ImageEncoding_RGB8,
                                               holoscan::MemoryKind::kCudaDevice));
  }

 private:
  HsdkMcapLogger& logger_;
};

// Mirrors HsdkMcapLoggerIntegrationTest's fixture in test_mcap_logger_integration.cc: resets
// GeneratorTriggerProbe's shared static sender_ slot before each test, so a future TEST_F added
// to this fixture can't forget to reset it and end up blocking on -- or silently reusing -- a
// previous test's stale sender.
class HsdkMcapLoggerTensorSchemaIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override { GeneratorTriggerProbe::reset(); }
};

TEST_F(HsdkMcapLoggerTensorSchemaIntegrationTest, LogsImageFramesEmittedByARunningGraph) {
  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_tensor_schema_integration_test.mcap");

  {
    HsdkMcapLogger logger(
        HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
    TensorSchemaIntegrationTestGraph graph(logger);
    const holoscan::ExecutionPlan plan = holoscan::compile(graph);
    ASSERT_TRUE(plan.ok());

    logger.start(plan.endpoint_catalog());
    auto run_session = holoscan::run_async(plan);

    holoscan::NotificationSender sender = GeneratorTriggerProbe::wait_for_sender();
    for (int i = 0; i < kMessageCount; ++i) {
      auto posted = post_with_retry(sender);
      ASSERT_TRUE(posted) << holoscan::runtime_error_message(posted.error());
      std::this_thread::sleep_for(kNotificationPeriod);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    run_session.request_stop();
    run_session.wait();
    logger.shutdown();
  }

  mcap::McapReader reader;
  const auto status = reader.open(output_path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  int message_count = 0;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    EXPECT_EQ(view.channel->topic, "camera");
    ASSERT_NE(view.schema, nullptr);
    EXPECT_EQ(view.schema->name, "isaac.mcap_logger.TensorFrame");
    EXPECT_EQ(view.schema->encoding, "flatbuffer");
    EXPECT_EQ(view.channel->messageEncoding, "flatbuffer");

    const auto* data = reinterpret_cast<const std::uint8_t*>(view.message.data);
    flatbuffers::Verifier verifier(data, view.message.dataSize);
    ASSERT_TRUE(wire::VerifyTensorFrameBuffer(verifier));
    const auto* frame = flatbuffers::GetRoot<wire::TensorFrame>(data);

    ASSERT_NE(frame->metadata_schema_name(), nullptr);
    EXPECT_EQ(frame->metadata_schema_name()->str(), "holoscan.schema.Image");

    ASSERT_NE(frame->metadata(), nullptr);
    flatbuffers::Verifier metadata_verifier(frame->metadata()->data(), frame->metadata()->size());
    ASSERT_TRUE(metadata_verifier.VerifyBuffer<holoscan::schema::Image>(nullptr));
    const auto* image = flatbuffers::GetRoot<holoscan::schema::Image>(frame->metadata()->data());
    EXPECT_EQ(image->height(), static_cast<std::int32_t>(kHeight));
    EXPECT_EQ(image->width(), static_cast<std::int32_t>(kWidth));
    EXPECT_EQ(image->encoding(), holoscan::schema::ImageEncoding_RGB8);

    // Prove metadata_binary_schema genuinely, generically describes metadata's bytes -- via
    // FlatBuffers' own runtime reflection, independent of this test's compile-time knowledge of
    // holoscan::schema::Image above -- exactly what a future generic decoder (a C++ replayer, or
    // a rewritten Foxglove converter) would rely on instead of a type hardcoded at compile time.
    ASSERT_NE(frame->metadata_binary_schema(), nullptr);
    EXPECT_GT(frame->metadata_binary_schema()->size(), 0U);
    const auto* reflected_schema = reflection::GetSchema(frame->metadata_binary_schema()->data());
    ASSERT_NE(reflected_schema, nullptr);
    ASSERT_NE(reflected_schema->root_table(), nullptr);
    EXPECT_TRUE(flatbuffers::Verify(*reflected_schema, *reflected_schema->root_table(),
                                    frame->metadata()->data(), frame->metadata()->size()));

    ASSERT_NE(frame->tensors(), nullptr);
    ASSERT_EQ(frame->tensors()->size(), 1U);
    const auto* entry = frame->tensors()->Get(0);
    EXPECT_EQ(entry->dtype(), wire::TensorDType_UInt8);
    ASSERT_NE(entry->shape(), nullptr);
    ASSERT_EQ(entry->shape()->size(), 1U);
    EXPECT_EQ(entry->shape()->Get(0), static_cast<std::uint32_t>(kFrameBytes));

    ASSERT_NE(entry->data(), nullptr);
    ASSERT_EQ(entry->data()->size(), kFrameBytes);
    const auto expected_byte = static_cast<std::uint8_t>(message_count);
    for (std::size_t i = 0; i < kFrameBytes; ++i) {
      EXPECT_EQ(entry->data()->Get(static_cast<flatbuffers::uoffset_t>(i)), expected_byte)
          << "at frame " << message_count << ", byte " << i;
    }

    ++message_count;
  }
  EXPECT_EQ(message_count, kMessageCount);
}

// Optional: only runs where a CUDA device is actually present (GTEST_SKIP() otherwise, reported
// as SKIP rather than FAIL or a silently absent test). Otherwise identical to
// LogsImageFramesEmittedByARunningGraph above, byte-for-byte -- the only difference is that
// CudaImageGeneratorOp's Tensor is real device memory, so TensorTopicPoller's materialization
// task must take its cudaMemcpy (kDLCUDA) branch rather than the std::memcpy one to produce the
// same correct output.
TEST_F(HsdkMcapLoggerTensorSchemaIntegrationTest, LogsImageFramesFromACudaDeviceTensor) {
  SKIP_IF_NO_CUDA_DEVICE();

  ScopedTempDirectory temp_directory;
  const auto output_path =
      temp_directory.path("mcap_logger_tensor_schema_cuda_integration_test.mcap");

  {
    HsdkMcapLogger logger(
        HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
    CudaTensorSchemaIntegrationTestGraph graph(logger);
    // The compiler refuses to assume device 0 for a Tensor-bearing output port -- it rejects an
    // implicit choice (IMPLICIT_DEVICE_ZERO_REJECTED) rather than guessing wrong. Every CUDA
    // Tensor output must be explicitly bound to a device via CompileOptions.
    holoscan::CompileOptions compile_options;
    compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "generator", .output_port = "image_out", .device = holoscan::DeviceId{0}});
    const holoscan::ExecutionPlan plan = holoscan::compile(graph, compile_options);
    if (!plan.ok()) {
      // A nontrivial CompileOptions is easy to get subtly wrong (e.g. the wrong operator_path/
      // output_port), and the resulting diagnostic tokens are the only way to tell why -- worth
      // printing them directly rather than just asserting plan.ok().
      for (const auto& diagnostic : plan.diagnostics()) {
        ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
      }
    }
    ASSERT_TRUE(plan.ok());

    logger.start(plan.endpoint_catalog());
    auto run_session = holoscan::run_async(plan);

    holoscan::NotificationSender sender = GeneratorTriggerProbe::wait_for_sender();
    for (int i = 0; i < kMessageCount; ++i) {
      auto posted = post_with_retry(sender);
      ASSERT_TRUE(posted) << holoscan::runtime_error_message(posted.error());
      std::this_thread::sleep_for(kNotificationPeriod);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    run_session.request_stop();
    run_session.wait();
    logger.shutdown();
  }

  mcap::McapReader reader;
  const auto status = reader.open(output_path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  int message_count = 0;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    EXPECT_EQ(view.channel->topic, "camera");
    ASSERT_NE(view.schema, nullptr);
    EXPECT_EQ(view.schema->name, "isaac.mcap_logger.TensorFrame");
    EXPECT_EQ(view.schema->encoding, "flatbuffer");
    EXPECT_EQ(view.channel->messageEncoding, "flatbuffer");

    const auto* data = reinterpret_cast<const std::uint8_t*>(view.message.data);
    flatbuffers::Verifier verifier(data, view.message.dataSize);
    ASSERT_TRUE(wire::VerifyTensorFrameBuffer(verifier));
    const auto* frame = flatbuffers::GetRoot<wire::TensorFrame>(data);

    ASSERT_NE(frame->metadata_schema_name(), nullptr);
    EXPECT_EQ(frame->metadata_schema_name()->str(), "holoscan.schema.Image");

    ASSERT_NE(frame->metadata(), nullptr);
    flatbuffers::Verifier metadata_verifier(frame->metadata()->data(), frame->metadata()->size());
    ASSERT_TRUE(metadata_verifier.VerifyBuffer<holoscan::schema::Image>(nullptr));
    const auto* image = flatbuffers::GetRoot<holoscan::schema::Image>(frame->metadata()->data());
    EXPECT_EQ(image->height(), static_cast<std::int32_t>(kHeight));
    EXPECT_EQ(image->width(), static_cast<std::int32_t>(kWidth));
    EXPECT_EQ(image->encoding(), holoscan::schema::ImageEncoding_RGB8);

    // Prove metadata_binary_schema genuinely, generically describes metadata's bytes -- via
    // FlatBuffers' own runtime reflection, independent of this test's compile-time knowledge of
    // holoscan::schema::Image above -- exactly what a future generic decoder (a C++ replayer, or
    // a rewritten Foxglove converter) would rely on instead of a type hardcoded at compile time.
    ASSERT_NE(frame->metadata_binary_schema(), nullptr);
    EXPECT_GT(frame->metadata_binary_schema()->size(), 0U);
    const auto* reflected_schema = reflection::GetSchema(frame->metadata_binary_schema()->data());
    ASSERT_NE(reflected_schema, nullptr);
    ASSERT_NE(reflected_schema->root_table(), nullptr);
    EXPECT_TRUE(flatbuffers::Verify(*reflected_schema, *reflected_schema->root_table(),
                                    frame->metadata()->data(), frame->metadata()->size()));

    ASSERT_NE(frame->tensors(), nullptr);
    ASSERT_EQ(frame->tensors()->size(), 1U);
    const auto* entry = frame->tensors()->Get(0);
    EXPECT_EQ(entry->dtype(), wire::TensorDType_UInt8);
    ASSERT_NE(entry->shape(), nullptr);
    ASSERT_EQ(entry->shape()->size(), 1U);
    EXPECT_EQ(entry->shape()->Get(0), static_cast<std::uint32_t>(kFrameBytes));

    ASSERT_NE(entry->data(), nullptr);
    ASSERT_EQ(entry->data()->size(), kFrameBytes);
    const auto expected_byte = static_cast<std::uint8_t>(message_count);
    for (std::size_t i = 0; i < kFrameBytes; ++i) {
      EXPECT_EQ(entry->data()->Get(static_cast<flatbuffers::uoffset_t>(i)), expected_byte)
          << "at frame " << message_count << ", byte " << i;
    }

    ++message_count;
  }
  EXPECT_EQ(message_count, kMessageCount);
}

}  // namespace
}  // namespace isaac::os::mcap_logger
