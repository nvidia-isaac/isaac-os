/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

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

#include "odometry_cpu_schema_traits.h"
#include "pose_cpu_schema_traits.h"
#include "transform_cpu_schema_traits.h"
#include "twist_cpu_schema_traits.h"

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

  holoscan::Graph graph{"geometry-cpu-message-test"};
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

TEST(GeometryCpuMessages, TransportsPose) {
  isaac::PoseCpuT message{};
  message.position = holoscan::schema::Vector3d(1.0, 2.0, 3.0);
  message.orientation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);
  message.header = std::make_shared<holoscan::schema::HeaderT>();
  message.header->capture_timestamp_ns = 42;
  message.header->frame_id = "map";

  const double expected_x = message.position.x();
  const double expected_z = message.position.z();
  const double expected_w = message.orientation.w();
  const std::int64_t expected_timestamp_ns = message.header->capture_timestamp_ns;
  const std::string expected_frame_id = message.header->frame_id;

  const auto value = transport_one(std::move(message));

  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(value->position.x(), expected_x);
  EXPECT_DOUBLE_EQ(value->position.z(), expected_z);
  EXPECT_DOUBLE_EQ(value->orientation.w(), expected_w);
  ASSERT_NE(value->header, nullptr);
  EXPECT_EQ(value->header->capture_timestamp_ns, expected_timestamp_ns);
  EXPECT_EQ(value->header->frame_id, expected_frame_id);
}

TEST(GeometryCpuMessages, TransportsPoseWithNoHeader) {
  isaac::PoseCpuT message{};
  message.position = holoscan::schema::Vector3d(0.0, 0.0, 0.0);
  message.orientation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);

  const auto value = transport_one(std::move(message));

  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(value->header, nullptr);
}

TEST(GeometryCpuMessages, TransportsTwist) {
  isaac::TwistCpuT message{};
  message.linear = holoscan::schema::Vector3d(1.0, 0.0, 0.0);
  message.angular = holoscan::schema::Vector3d(0.0, 0.0, 0.5);

  const double expected_linear_x = message.linear.x();
  const double expected_angular_z = message.angular.z();

  const auto value = transport_one(std::move(message));

  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(value->linear.x(), expected_linear_x);
  EXPECT_DOUBLE_EQ(value->angular.z(), expected_angular_z);
}

TEST(GeometryCpuMessages, TransportsTransform) {
  isaac::TransformCpuT message{};
  message.translation = holoscan::schema::Vector3d(0.1, 0.2, 0.3);
  message.rotation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);
  message.child_frame_id = "base_link";

  const double expected_translation_y = message.translation.y();
  const std::string expected_child_frame_id = message.child_frame_id;

  const auto value = transport_one(std::move(message));

  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(value->translation.y(), expected_translation_y);
  EXPECT_EQ(value->child_frame_id, expected_child_frame_id);
}

TEST(GeometryCpuMessages, TransportsOdometry) {
  isaac::OdometryCpuT message{};
  message.header = std::make_shared<holoscan::schema::HeaderT>();
  message.header->capture_timestamp_ns = 100;
  message.header->frame_id = "odom";
  message.child_frame_id = "base_link";
  message.pose = std::make_shared<isaac::PoseCpuT>();
  message.pose->position = holoscan::schema::Vector3d(1.0, 2.0, 0.0);
  message.pose->orientation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);
  message.pose_covariance = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();
  // (0, 1) and (1, 0) get deliberately different values: an all-diagonal (or otherwise
  // symmetric) covariance matrix would isApprox-match its own transpose, so it wouldn't
  // distinguish correct row-major handling from an accidentally-transposed one.
  message.pose_covariance(0, 1) = 0.01;
  message.pose_covariance(1, 0) = 0.02;
  message.twist = std::make_shared<isaac::TwistCpuT>();
  message.twist->linear = holoscan::schema::Vector3d(0.5, 0.0, 0.0);
  message.twist->angular = holoscan::schema::Vector3d(0.0, 0.0, 0.1);
  message.twist_covariance = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();
  message.twist_covariance(0, 1) = 0.03;
  message.twist_covariance(1, 0) = 0.04;

  const std::int64_t expected_timestamp_ns = message.header->capture_timestamp_ns;
  const std::string expected_child_frame_id = message.child_frame_id;
  const double expected_pose_x = message.pose->position.x();
  const Eigen::Matrix<double, 6, 6, Eigen::RowMajor> expected_pose_covariance =
      message.pose_covariance;
  const double expected_twist_angular_z = message.twist->angular.z();
  const Eigen::Matrix<double, 6, 6, Eigen::RowMajor> expected_twist_covariance =
      message.twist_covariance;

  const auto value = transport_one(std::move(message));

  ASSERT_TRUE(value.has_value());
  ASSERT_NE(value->header, nullptr);
  EXPECT_EQ(value->header->capture_timestamp_ns, expected_timestamp_ns);
  EXPECT_EQ(value->child_frame_id, expected_child_frame_id);
  ASSERT_NE(value->pose, nullptr);
  EXPECT_DOUBLE_EQ(value->pose->position.x(), expected_pose_x);
  EXPECT_TRUE(value->pose_covariance.isApprox(expected_pose_covariance));
  ASSERT_NE(value->twist, nullptr);
  EXPECT_DOUBLE_EQ(value->twist->angular.z(), expected_twist_angular_z);
  EXPECT_TRUE(value->twist_covariance.isApprox(expected_twist_covariance));
}

}  // namespace
}  // namespace isaac::os
