/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/mock/mock_sink_op.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {
namespace {

bool valid_command(const isaac::JointCommandCpuT& command, std::size_t joint_count) {
  const auto finite = [](float value) { return std::isfinite(value); };
  const auto nonzero = [](float value) { return value != 0.0F; };
  const bool sizes_match = command.position.size() == joint_count &&
                           command.velocity.size() == joint_count &&
                           command.effort.size() == joint_count &&
                           command.kp.size() == joint_count && command.kd.size() == joint_count;
  if (!sizes_match) {
    return false;
  }
  const bool finite_values = std::ranges::all_of(command.position, finite) &&
                             std::ranges::all_of(command.velocity, finite) &&
                             std::ranges::all_of(command.effort, finite) &&
                             std::ranges::all_of(command.kp, finite) &&
                             std::ranges::all_of(command.kd, finite);
  // The current WBC implementation deliberately emits zero velocity and effort commands.
  const bool has_nonzero_command = std::ranges::any_of(command.position, nonzero) ||
                                   std::ranges::any_of(command.kp, nonzero) ||
                                   std::ranges::any_of(command.kd, nonzero);
  return finite_values && has_nonzero_command;
}

}  // namespace

void MockSinkOp::setup(holoscan::OperatorSpec& spec) { spec.input(command, "in_joint_command"); }

holoscan::Contract MockSinkOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{command});
  return contract;
}

holoscan::expected<void, holoscan::Error> MockSinkOp::compute(holoscan::ExecutionContext&) {
  auto received = command.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  if (first_command_) {
    HOLOSCAN_LOG_INFO("WBC control loop is running.");
    first_command_ = false;
  }
  if (!valid_command(received->data, joint_count_)) {
    HOLOSCAN_LOG_ERROR("The WBC command is invalid for {} joints.", joint_count_);
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
  }
  return {};
}

}  // namespace isaac::os::ops
