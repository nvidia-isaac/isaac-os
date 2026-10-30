/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_G1_LOW_LEVEL_G1_LOW_LEVEL_JOINT_COMMAND_PUBLISHER_H
#define ISAAC_OS_OPERATORS_G1_LOW_LEVEL_G1_LOW_LEVEL_JOINT_COMMAND_PUBLISHER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <unitree/idl/hg/HandCmd_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>

#include "joint_command_cpu_schema_traits.h"
#include "operators/g1_low_level/g1_params.h"
#include "operators/g1_low_level/mode_machine.h"

namespace isaac::os::ops {

/// Writes joint commands to the Unitree G1 low-level DDS channels.
///
/// Each received command is published as one LowCmd, plus one HandCmd per Dex 3-1 hand when the
/// configured joint count covers them.
class G1JointCommandPublisherOp final : public holoscan::Operator<G1DdsParams> {
 public:
  /// @param mode_machine Shared control mode updated by the paired state subscriber.
  /// @param expected_count Requests run completion after this many commands; zero runs on.
  explicit G1JointCommandPublisherOp(std::shared_ptr<ModeMachine> mode_machine,
                                     std::size_t expected_count = 0U)
      : mode_machine_(std::move(mode_machine)), expected_count_(expected_count) {
    if (!mode_machine_) {
      throw std::invalid_argument(
          "The G1 joint command publisher requires a shared robot control mode.");
    }
  }

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::JointCommandCpuT> command;

 private:
  std::shared_ptr<ModeMachine> mode_machine_;
  std::size_t expected_count_{};
  std::size_t published_count_{};

  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;
  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> left_hand_publisher_;
  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> right_hand_publisher_;

  std::size_t joint_count_{};
  bool with_hands_{false};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_G1_LOW_LEVEL_G1_LOW_LEVEL_JOINT_COMMAND_PUBLISHER_H
