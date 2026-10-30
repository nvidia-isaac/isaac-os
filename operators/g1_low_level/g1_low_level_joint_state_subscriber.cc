/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/g1_low_level/g1_low_level_joint_state_subscriber.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <holoscan/logger/logger.hpp>

#include "operators/g1_low_level/unitree_imu_conversion.h"

namespace isaac::os::ops {
namespace {

constexpr std::size_t kBodyMotors = 29U;
constexpr std::size_t kHandMotors = 7U;
constexpr std::size_t kBodyOnlyJoints = kBodyMotors;
constexpr std::size_t kBodyAndHandJoints = kBodyMotors + (2U * kHandMotors);

}  // namespace

void G1JointStateSubscriberOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(joint_state, "out_joint_state");
  spec.output(imu, "out_imu");
}

holoscan::Contract G1JointStateSubscriberOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnClock{.period = period_});
  return contract;
}

void G1JointStateSubscriberOp::start() {
  const G1DdsParams& config = params();
  joint_count_ = static_cast<std::size_t>(config.g1_num_motors);
  if (joint_count_ == kBodyOnlyJoints) {
    with_hands_ = false;
  } else if (joint_count_ == kBodyAndHandJoints) {
    with_hands_ = true;
  } else {
    throw std::runtime_error("Invalid number of motors: " + std::to_string(config.g1_num_motors) +
                             ". Expected " + std::to_string(kBodyOnlyJoints) +
                             " for the G1 body or " + std::to_string(kBodyAndHandJoints) +
                             " for the G1 with Dex 3-1 hands.");
  }

  joint_state_message_.position.assign(joint_count_, 0.0F);
  joint_state_message_.velocity.assign(joint_count_, 0.0F);
  imu_message_.orientation = identity_orientation();
  imu_message_.angular_velocity = to_schema_vector({.x = 0.0F, .y = 0.0F, .z = 0.0F});
  imu_message_.linear_acceleration = to_schema_vector({.x = 0.0F, .y = 0.0F, .z = 0.0F});

  mode_machine_->set(static_cast<std::uint8_t>(config.mode_machine));

  stopping_.store(false);
  unitree::robot::ChannelFactory::Instance()->Init(static_cast<int>(config.dds_domain_id),
                                                   config.network_interface);
  lowstate_subscriber_ =
      std::make_shared<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>>(
          "rt/lowstate");
  lowstate_subscriber_->InitChannel([this](const void* message) { handle_low_state(message); }, 1);
  if (with_hands_) {
    left_hand_subscriber_ =
        std::make_shared<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::HandState_>>(
            "rt/dex3/left/state");
    left_hand_subscriber_->InitChannel(
        [this](const void* message) { handle_hand_state(message, kBodyMotors); }, 1);
    right_hand_subscriber_ =
        std::make_shared<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::HandState_>>(
            "rt/dex3/right/state");
    right_hand_subscriber_->InitChannel(
        [this](const void* message) { handle_hand_state(message, kBodyMotors + kHandMotors); }, 1);
  }
  last_lowstate_message_count_ = lowstate_message_count_.load();
}

void G1JointStateSubscriberOp::stop() {
  // Shut the DDS subscribers down before the cached sample they write into goes away.
  stopping_.store(true);
  const auto close_subscriber = [](auto& subscriber) {
    if (subscriber) {
      subscriber->CloseChannel();
      subscriber.reset();
    }
  };
  close_subscriber(lowstate_subscriber_);
  close_subscriber(left_hand_subscriber_);
  close_subscriber(right_hand_subscriber_);
  // Wait out any callback already running inside a handler.
  const std::lock_guard<std::shared_mutex> drain(state_mutex_);
}

bool G1JointStateSubscriberOp::wait_for_sample() {
  const auto deadline = std::chrono::steady_clock::now() + period_;
  while (lowstate_message_count_.load() == last_lowstate_message_count_) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  last_lowstate_message_count_ = lowstate_message_count_.load();
  return true;
}

holoscan::expected<void, holoscan::Error> G1JointStateSubscriberOp::compute(
    holoscan::ExecutionContext& context) {
  if (!wait_for_sample()) {
    // Forwarding the previous sample would feed the policy a stale observation, so let the
    // downstream alignment simply not fire this tick.
    HOLOSCAN_LOG_WARN("No Unitree G1 LowState sample arrived within the control period.");
    return {};
  }

  auto capture_time = context.clock().now();
  if (!capture_time) {
    return holoscan::make_unexpected(std::move(capture_time).error());
  }

  const holoscan::EmitOptions options{.capture_time = *capture_time};
  {
    const std::shared_lock<std::shared_mutex> lock(state_mutex_);
    // Publish matching IMU data before joint state triggers WBC.
    auto emitted = imu.emit(imu_message_, options);
    if (!emitted) {
      return emitted;
    }
    emitted = joint_state.emit(joint_state_message_, options);
    if (!emitted) {
      return emitted;
    }
  }

  auto finished = context.clock().now();
  if (!finished) {
    return holoscan::make_unexpected(std::move(finished).error());
  }
  return {};
}

void G1JointStateSubscriberOp::handle_low_state(const void* message) {
  // Ignore deliveries that arrive during teardown.
  if (stopping_.load()) {
    return;
  }
  const auto& low_state = *static_cast<const unitree_hg::msg::dds_::LowState_*>(message);
  {
    const std::unique_lock<std::shared_mutex> lock(state_mutex_);
    for (std::size_t index = 0; index < kBodyMotors; ++index) {
      joint_state_message_.position[index] = low_state.motor_state()[index].q();
      joint_state_message_.velocity[index] = low_state.motor_state()[index].dq();
    }

    const auto& imu_state = low_state.imu_state();
    imu_message_.orientation = to_schema_orientation({.w = imu_state.quaternion()[0],
                                                      .x = imu_state.quaternion()[1],
                                                      .y = imu_state.quaternion()[2],
                                                      .z = imu_state.quaternion()[3]});
    imu_message_.angular_velocity = to_schema_vector({.x = imu_state.gyroscope()[0],
                                                      .y = imu_state.gyroscope()[1],
                                                      .z = imu_state.gyroscope()[2]});
    imu_message_.linear_acceleration = to_schema_vector({.x = imu_state.accelerometer()[0],
                                                         .y = imu_state.accelerometer()[1],
                                                         .z = imu_state.accelerometer()[2]});
  }

  if (low_state.mode_machine() != mode_machine_value_) {
    mode_machine_value_ = low_state.mode_machine();
    HOLOSCAN_LOG_INFO("Robot control mode changed to {}.", static_cast<int>(mode_machine_value_));
    mode_machine_->set(mode_machine_value_);
  }
  lowstate_message_count_.fetch_add(1);
}

void G1JointStateSubscriberOp::handle_hand_state(const void* message, std::size_t offset) {
  // Ignore deliveries that arrive during teardown.
  if (stopping_.load()) {
    return;
  }
  const auto& hand_state = *static_cast<const unitree_hg::msg::dds_::HandState_*>(message);
  const std::unique_lock<std::shared_mutex> lock(state_mutex_);
  for (std::size_t index = 0; index < kHandMotors; ++index) {
    joint_state_message_.position[offset + index] = hand_state.motor_state()[index].q();
    joint_state_message_.velocity[offset + index] = hand_state.motor_state()[index].dq();
  }
}

}  // namespace isaac::os::ops
