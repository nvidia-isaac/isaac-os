/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/g1_low_level/g1_low_level_joint_command_publisher.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

namespace isaac::os::ops {
namespace {

constexpr std::size_t kBodyMotors = 29U;
constexpr std::size_t kHandMotors = 7U;
constexpr std::size_t kBodyOnlyJoints = kBodyMotors;
constexpr std::size_t kBodyAndHandJoints = kBodyMotors + (2U * kHandMotors);

/// CRC used by LowCmd, taken from the unitree_sdk2 examples.
std::uint32_t crc32_core(const std::uint32_t* data, std::uint32_t length) {
  std::uint32_t crc = 0xFFFFFFFFU;
  constexpr std::uint32_t kPolynomial = 0x04c11db7U;
  for (std::uint32_t word = 0; word < length; ++word) {
    std::uint32_t bit_mask = 1U << 31U;
    for (std::uint32_t bit = 0; bit < 32U; ++bit) {
      if (crc & 0x80000000U) {
        crc = (crc << 1U) ^ kPolynomial;
      } else {
        crc <<= 1U;
      }
      if (data[word] & bit_mask) {
        crc ^= kPolynomial;
      }
      bit_mask >>= 1U;
    }
  }
  return crc;
}

/// Copies one motor's worth of the command into a LowCmd or HandCmd motor slot.
template <typename MotorCmd>
void fill_motor_cmd(MotorCmd& motor_cmd, const isaac::JointCommandCpuT& command,
                    std::size_t index) {
  motor_cmd.mode() = 1;  // 1 enables the motor, 0 disables it.
  motor_cmd.q() = command.position[index];
  motor_cmd.dq() = command.velocity[index];
  motor_cmd.tau() = command.effort[index];
  motor_cmd.kp() = command.kp[index];
  motor_cmd.kd() = command.kd[index];
}

}  // namespace

void G1JointCommandPublisherOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(command, "in_joint_command");
}

holoscan::Contract G1JointCommandPublisherOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{command});
  return contract;
}

void G1JointCommandPublisherOp::start() {
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

  unitree::robot::ChannelFactory::Instance()->Init(static_cast<int>(config.dds_domain_id),
                                                   config.network_interface);

  // The robot refuses low-level commands while a high-level mode owns the joints.
  unitree::robot::b2::MotionSwitcherClient motion_switcher;
  motion_switcher.SetTimeout(5.0F);
  motion_switcher.Init();
  std::string form;
  std::string name;
  while (motion_switcher.CheckMode(form, name), !name.empty()) {
    if (motion_switcher.ReleaseMode()) {
      HOLOSCAN_LOG_ERROR("Unable to switch the robot into release mode.");
    }
    std::this_thread::sleep_for(std::chrono::seconds{5});
  }

  lowcmd_publisher_ =
      std::make_shared<unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>>(
          "rt/lowcmd");
  lowcmd_publisher_->InitChannel();
  if (with_hands_) {
    left_hand_publisher_ =
        std::make_shared<unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_>>(
            "rt/dex3/left/cmd");
    left_hand_publisher_->InitChannel();
    right_hand_publisher_ =
        std::make_shared<unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_>>(
            "rt/dex3/right/cmd");
    right_hand_publisher_->InitChannel();
  }
}

holoscan::expected<void, holoscan::Error> G1JointCommandPublisherOp::compute(
    holoscan::ExecutionContext& context) {
  auto received = command.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }

  const isaac::JointCommandCpuT& joint_command = received->data;
  if (joint_command.position.size() != joint_count_ ||
      joint_command.velocity.size() != joint_count_ ||
      joint_command.effort.size() != joint_count_ || joint_command.kp.size() != joint_count_ ||
      joint_command.kd.size() != joint_count_) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "The joint command must carry one position, velocity, effort, kp, and kd "
                        "value per configured motor."});
  }

  unitree_hg::msg::dds_::LowCmd_ low_cmd;
  low_cmd.mode_pr() = 0;
  low_cmd.mode_machine() = mode_machine_->get();
  for (std::size_t index = 0; index < kBodyMotors; ++index) {
    fill_motor_cmd(low_cmd.motor_cmd().at(index), joint_command, index);
  }
  low_cmd.crc() =
      crc32_core(reinterpret_cast<const std::uint32_t*>(&low_cmd), (sizeof(low_cmd) >> 2U) - 1U);
  lowcmd_publisher_->Write(low_cmd);

  if (with_hands_) {
    unitree_hg::msg::dds_::HandCmd_ left_hand_cmd;
    unitree_hg::msg::dds_::HandCmd_ right_hand_cmd;
    left_hand_cmd.motor_cmd().resize(kHandMotors);
    right_hand_cmd.motor_cmd().resize(kHandMotors);
    for (std::size_t index = 0; index < kHandMotors; ++index) {
      fill_motor_cmd(left_hand_cmd.motor_cmd().at(index), joint_command, kBodyMotors + index);
      fill_motor_cmd(right_hand_cmd.motor_cmd().at(index), joint_command,
                     kBodyMotors + kHandMotors + index);
    }
    left_hand_publisher_->Write(left_hand_cmd);
    right_hand_publisher_->Write(right_hand_cmd);
  }

  ++published_count_;
  if (expected_count_ != 0U && published_count_ >= expected_count_) {
    context.request_completion();
  }
  return {};
}

}  // namespace isaac::os::ops
