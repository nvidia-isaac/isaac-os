/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "operators/unitree_sim_ucx/unitree_sim_ucx_ops_adapter.h"

namespace isaac::os::ops {
namespace {
struct ImageTestState {
  std::atomic<const void*> source_pixels{nullptr};
  std::atomic<std::int64_t> source_capture_ns{0};
};

struct ImageTestStateBinding {
  std::shared_ptr<ImageTestState> state;
};

void check_image_pixels(const holoscan::Tensor& image, const ImageTestState& state) {
  std::array<unsigned char, 12> pixels{};
  ASSERT_EQ(image.nbytes(), pixels.size());
  EXPECT_EQ(image.data(), state.source_pixels.load());
  ASSERT_EQ(cudaMemcpy(pixels.data(), image.data(), pixels.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (auto pixel : pixels) EXPECT_EQ(pixel, 17);
}

class ImageSourceOp final : public holoscan::Operator<> {
 public:
  explicit ImageSourceOp(ImageTestStateBinding binding) : state_(std::move(binding.state)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "out_message")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 3U},
            .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = 12U},
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{10}});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    if (emitted_) return {};
    auto now = context.clock().now();
    if (!now) return holoscan::make_unexpected(now.error());
    state_->source_capture_ns.store(now->timestamp_ns);
    const std::array<std::int64_t, 3> shape{2, 2, 3};
    auto loan = output.allocate_tensor({.shape = shape, .dtype = {kDLUInt, 8U, 1U}});
    if (!loan) return holoscan::make_unexpected(loan.error());
    auto writer = loan->write(context.cuda_stream());
    if (!writer) return holoscan::make_unexpected(writer.error());
    state_->source_pixels.store(writer->data());
    if (cudaMemsetAsync(writer->data(), 17, 12, context.cuda_stream().get()) != cudaSuccess ||
        cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess)
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "Unable to initialize the test camera frame."});
    auto committed = std::move(*writer).commit();
    if (!committed) return holoscan::make_unexpected(committed.error());
    holoscan::schema::ImageT image;
    image.width = 2;
    image.height = 2;
    image.encoding = holoscan::schema::ImageEncoding_RGB8;
    emitted_ = true;
    return output.emit_tensor(std::move(*loan), image, {.capture_time = *now});
  }
  holoscan::Output<holoscan::schema::ImageT> output;

 private:
  std::shared_ptr<ImageTestState> state_;
  bool emitted_{false};
};
class ImageSinkOp final : public holoscan::Operator<> {
 public:
  explicit ImageSinkOp(ImageTestStateBinding binding) : state_(std::move(binding.state)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice}});
  }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto received = input.receive();
    if (!received) return holoscan::make_unexpected(received.error());
    check_image_pixels(received->data, *state_);
    EXPECT_EQ(received->metadata.capture_time().timestamp_ns, state_->source_capture_ns.load());
    context.request_completion();
    return {};
  }
  holoscan::Input<holoscan::Tensor> input;

 private:
  std::shared_ptr<ImageTestState> state_;
};
TEST(UnitreeSimUcxAdapter, PreservesPayloadAfterNativeMessageIsReleased) {
  auto shutdown = holoscan::ManagedShutdownSignals::create();
  ASSERT_TRUE(shutdown);
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP();
  holoscan::Graph graph{"image-publication-test"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const ImageTestStateBinding state{std::make_shared<ImageTestState>()};
  auto source = graph.op<ImageSourceOp>("source", state);
  auto sink = graph.op<ImageSinkOp>("sink", state);
  auto adapter = graph.op<UnitreeSimImageToTensorOp>("adapter");
  graph.add_flow(source->output, adapter->input);
  graph.add_flow(adapter->output, sink->input);
  graph.partition("source").add(source);
  graph.partition("adapter").add(adapter);
  graph.partition("sink").add(sink);
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      {.operator_path = "source", .output_port = "out_message", .device = holoscan::DeviceId{0}});
  options.deployment.bind_tensor_output_device(
      {.operator_path = "adapter", .output_port = "output", .device = holoscan::DeviceId{0}});
  const auto plan = holoscan::compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = holoscan::run_async(plan, *shutdown);
  session.wait();
  std::string diagnostics;
  for (const auto& diagnostic : session.diagnostics()) {
    diagnostics += holoscan::format_diagnostic(diagnostic, plan.graph());
  }
  EXPECT_NE(session.termination(), holoscan::RunTermination::kFailed) << diagnostics;
}
}  // namespace
}  // namespace isaac::os::ops
