/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "operators/unitree_sim_ucx/unitree_sim_ucx_ops_adapter.h"

namespace isaac::os::ops {
namespace {
using namespace std::chrono_literals;

struct Observation {
  std::mutex mutex;
  std::condition_variable ready;
  int count{};
};

struct ObservationState {
  std::shared_ptr<Observation> observation;
};

class PairSourceOp final : public holoscan::Operator<> {
 public:
  PairSourceOp(bool imu_first, bool matching) : imu_first_(imu_first), matching_(matching) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(joints, "joints");
    spec.output(imu, "imu");
  }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 20ms});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& ctx) override {
    if (sent_) return {};
    sent_ = true;
    auto now = ctx.clock().now();
    if (!now) return holoscan::make_unexpected(now.error());
    isaac::JointStateCpuT state;
    state.position = {0.125F};
    holoscan::schema::ImuT inertial;
    inertial.orientation = holoscan::schema::Quaternionf{0.0F, 0.0F, 0.0F, 1.0F};
    const holoscan::EmitOptions state_options{.capture_time = *now};
    const holoscan::EmitOptions imu_options{
        .capture_time = ctx.clock().at_ns(now->timestamp_ns + (matching_ ? 0 : 20000000))};
    auto first = imu_first_ ? imu.emit(inertial, imu_options) : joints.emit(state, state_options);
    if (!first) return first;
    // Force the downstream partition to see only one sensor before its partner arrives.
    std::this_thread::sleep_for(100ms);
    return imu_first_ ? joints.emit(state, state_options) : imu.emit(inertial, imu_options);
  }
  holoscan::Output<isaac::JointStateCpuT> joints;
  holoscan::Output<holoscan::schema::ImuT> imu;

 private:
  bool imu_first_;
  bool matching_;
  bool sent_{};
};

// Exercise the unchanged real WBC selection contract without loading a GPU policy.
class WbcInputProbeOp final : public holoscan::Operator<> {
 public:
  explicit WbcInputProbeOp(ObservationState state) : observation_(std::move(state.observation)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(joints, "joints");
    spec.input(imu, "imu");
  }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAligned{.ports = holoscan::ports(joints, imu), .window = 0ns});
    result.input(joints, holoscan::Exact{});
    result.input(imu, holoscan::Exact{});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto state = joints.receive();
    if (!state) return holoscan::make_unexpected(state.error());
    auto inertial = imu.receive();
    if (!inertial) return holoscan::make_unexpected(inertial.error());
    EXPECT_EQ(state->metadata.capture_time().timestamp_ns,
              inertial->metadata.capture_time().timestamp_ns);
    EXPECT_EQ(state->data.position, std::vector<float>{0.125F});
    EXPECT_FLOAT_EQ(inertial->data.orientation.w(), 1.0F);
    std::lock_guard lock(observation_->mutex);
    ++observation_->count;
    observation_->ready.notify_all();
    return {};
  }
  holoscan::Input<isaac::JointStateCpuT> joints;
  holoscan::Input<holoscan::schema::ImuT> imu;

 private:
  std::shared_ptr<Observation> observation_;
};

TEST(SimStatePairing, DelayedInputsReachTheRealWbcContractOnlyAsMatchingPairs) {
  for (const bool imu_first : {false, true}) {
    for (const bool matching : {false, true}) {
      SCOPED_TRACE(imu_first);
      SCOPED_TRACE(matching);
      auto observation = std::make_shared<Observation>();
      holoscan::Graph graph{"sim-state-pairing"};
      graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
      const auto source = graph.op<PairSourceOp>("source", imu_first, matching);
      const auto adapter = graph.op<UnitreeSimStateOp>("adapter");
      const auto probe = graph.op<WbcInputProbeOp>("wbc_contract", ObservationState{observation});
      const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                               .queue_depth = 2U};
      graph.add_flow(source->joints, adapter->joint_state_input, latest);
      graph.add_flow(source->imu, adapter->imu_input, latest);
      graph.add_flow(adapter->joint_state, probe->joints);
      graph.add_flow(adapter->imu, probe->imu);
      graph.partition("transport").add(source);
      graph.partition("control").add(adapter).add(probe);
      const auto plan = holoscan::compile(graph);
      ASSERT_TRUE(plan.ok()) << plan.json();
      auto session = holoscan::run_async(plan);
      {
        std::unique_lock lock(observation->mutex);
        const bool received = observation->ready.wait_for(
            lock, matching ? 2s : 300ms, [observation] { return observation->count > 0; });
        EXPECT_EQ(received, matching);
      }
      session.request_stop();
      session.wait();
      EXPECT_EQ(session.termination(), holoscan::RunTermination::kStopped);
      EXPECT_EQ(observation->count, matching ? 1 : 0);
    }
  }
}
}  // namespace
}  // namespace isaac::os::ops
