/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "camera_info_cpu_schema_traits.h"

namespace isaac::os {
namespace {

using namespace std::chrono_literals;

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

  holoscan::Graph graph{"camera-info-cpu-message-test"};
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

TEST(CameraInfoCpuMessages, TransportsCameraInfo) {
  isaac::CameraInfoCpuT message{};
  message.header = std::make_shared<holoscan::schema::HeaderT>();
  message.header->capture_timestamp_ns = 7;
  message.header->frame_id = "camera_optical_frame";
  message.height = 480U;
  message.width = 640U;
  message.distortion_model = "plumb_bob";
  message.d = {0.1, 0.2, 0.0, 0.0, 0.0};
  // k/r are holoscan.schema.Matrix3d -- a plain flatc struct (not native_type/Eigen-backed), so
  // constructed from a flat row-major std::array directly, same as the SDK's own Matrix3f
  // elsewhere. Non-symmetric throughout (the identity matrix real rectification data often
  // defaults to would be symmetric, and wouldn't distinguish correct row-major handling from an
  // accidentally-transposed one).
  message.k = holoscan::schema::Matrix3d(
      std::array<double, 9>{600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0});
  message.r = holoscan::schema::Matrix3d(
      std::array<double, 9>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
  // p is isaac.Matrix3x4d, Eigen::RowMajor -- Eigen's comma initializer fills row-major,
  // matching the wire layout regardless of Eigen's own (column-major, by default) in-memory
  // storage order.
  message.p << 600.0, 0.0, 320.0, 0.0, 0.0, 600.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  message.binning_x = 1U;
  message.binning_y = 1U;
  message.roi_x_offset = 0U;
  message.roi_y_offset = 0U;
  message.roi_height = 0U;
  message.roi_width = 0U;
  message.roi_do_rectify = false;

  const std::int64_t expected_timestamp_ns = message.header->capture_timestamp_ns;
  const std::uint32_t expected_height = message.height;
  const std::uint32_t expected_width = message.width;
  const std::string expected_distortion_model = message.distortion_model;
  const std::vector<double> expected_d = message.d;
  const holoscan::schema::Matrix3d expected_k = message.k;
  const holoscan::schema::Matrix3d expected_r = message.r;
  const Eigen::Matrix<double, 3, 4, Eigen::RowMajor> expected_p = message.p;
  const std::uint32_t expected_binning_x = message.binning_x;
  const bool expected_roi_do_rectify = message.roi_do_rectify;

  const auto value = transport_one(std::move(message));

  ASSERT_TRUE(value.has_value());
  ASSERT_NE(value->header, nullptr);
  EXPECT_EQ(value->header->capture_timestamp_ns, expected_timestamp_ns);
  EXPECT_EQ(value->height, expected_height);
  EXPECT_EQ(value->width, expected_width);
  EXPECT_EQ(value->distortion_model, expected_distortion_model);
  ASSERT_EQ(value->d.size(), expected_d.size());
  EXPECT_DOUBLE_EQ(value->d[0], expected_d[0]);
  for (flatbuffers::uoffset_t i = 0; i < 9; ++i) {
    EXPECT_DOUBLE_EQ(value->k.m()->Get(i), expected_k.m()->Get(i));
    EXPECT_DOUBLE_EQ(value->r.m()->Get(i), expected_r.m()->Get(i));
  }
  EXPECT_TRUE(value->p.isApprox(expected_p));
  EXPECT_EQ(value->binning_x, expected_binning_x);
  EXPECT_EQ(value->roi_do_rectify, expected_roi_do_rectify);
}

}  // namespace
}  // namespace isaac::os
