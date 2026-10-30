/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_G1_LOW_LEVEL_G1_LOW_LEVEL_JOINT_STATE_SUBSCRIBER_H
#define ISAAC_OS_OPERATORS_G1_LOW_LEVEL_G1_LOW_LEVEL_JOINT_STATE_SUBSCRIBER_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <utility>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <unitree/idl/hg/HandState_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "joint_state_cpu_schema_traits.h"
#include "operators/g1_low_level/g1_params.h"
#include "operators/g1_low_level/mode_machine.h"

namespace isaac::os::ops {

/// Publishes the Unitree G1 low-level joint state and pelvis IMU onto the graph.
///
/// DDS delivery is asynchronous, so the handlers cache the newest sample and the periodic compute
/// forwards it. Only the fields the WBC policy consumes are carried: LowState also reports joint
/// acceleration and estimated torque, which JointStateCpu has no room for.
class G1JointStateSubscriberOp final : public holoscan::Operator<G1DdsParams> {
 public:
  explicit G1JointStateSubscriberOp(std::shared_ptr<ModeMachine> mode_machine,
                                    std::chrono::nanoseconds period = std::chrono::milliseconds{20})
      : mode_machine_(std::move(mode_machine)), period_(period) {
    if (!mode_machine_) {
      throw std::invalid_argument(
          "The G1 joint state subscriber requires a shared robot control mode.");
    }
  }

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Output<isaac::JointStateCpuT> joint_state;
  holoscan::Output<holoscan::schema::ImuT> imu;

 private:
  /// Copies one HandState sample into the joint slots starting at `offset`.
  void handle_hand_state(const void* message, std::size_t offset);
  void handle_low_state(const void* message);
  /// Blocks until a LowState newer than the last forwarded one arrives, or `period_` elapses.
  bool wait_for_sample();

  std::shared_ptr<ModeMachine> mode_machine_;
  std::chrono::nanoseconds period_;

  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber_;
  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::HandState_> left_hand_subscriber_;
  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::HandState_> right_hand_subscriber_;

  // Guards the cached sample against the DDS callback threads.
  std::shared_mutex state_mutex_;
  isaac::JointStateCpuT joint_state_message_;
  holoscan::schema::ImuT imu_message_;

  // True once stop() begins; signals the DDS callbacks to bail before the subscribers are reset.
  std::atomic<bool> stopping_{false};
  std::atomic<std::uint64_t> lowstate_message_count_{0};
  std::uint64_t last_lowstate_message_count_{0};

  std::size_t joint_count_{};
  std::uint8_t mode_machine_value_{0};
  bool with_hands_{false};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_G1_LOW_LEVEL_G1_LOW_LEVEL_JOINT_STATE_SUBSCRIBER_H
