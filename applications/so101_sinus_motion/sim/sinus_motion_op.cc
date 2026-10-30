/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "applications/so101_sinus_motion/sim/sinus_motion_op.h"

#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>

namespace isaac::os::apps::so101_sinus_motion_sim {
namespace {

float checked_command_value(double value, const char* name, std::size_t index) {
  // Check the range before casting. Decimal rounding is allowed, but overflow,
  // non-finite values, and underflow to zero cannot form a valid command.
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max() ||
      (value != 0.0 && static_cast<float>(value) == 0.0F)) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kInvalidArgument,
        std::format("Invalid value in {}[{}] = {:.{}g}: joint commands require finite float values "
                    "without overflow or underflow.",
                    name, index, value, std::numeric_limits<double>::max_digits10));
  }
  return static_cast<float>(value);
}

std::vector<double> expand(std::vector<double> values, std::size_t size, const char* name) {
  if (values.empty()) {
    values.push_back(0.0);
  }
  if (values.size() == 1U) {
    values.resize(size, values.front());
  }
  if (values.size() != size) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kInvalidArgument,
        std::string{"SinusMotionOp: "} + name + " must have length 1 or num_joints.");
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    checked_command_value(values[i], name, i);
  }
  return values;
}

}  // namespace

SinusMotionOp::SinusMotionOp(SinusMotionParams params) : params_(std::move(params)) {
  if (params_.num_joints <= 0) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "SinusMotionOp: num_joints must be positive.");
  }
  const auto n = static_cast<std::size_t>(params_.num_joints);
  params_.default_position = expand(std::move(params_.default_position), n, "default_position");
  params_.kp = expand(std::move(params_.kp), n, "kp");
  params_.kd = expand(std::move(params_.kd), n, "kd");
  if (params_.sin_frequency < 0.0 || params_.sin_amplitude < 0.0) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kInvalidArgument,
        "SinusMotionOp: sin_frequency and sin_amplitude must be non-negative.");
  }
}

void SinusMotionOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(joint_state_in, "in_joint_state");
  spec.output(joint_command_out, "out_joint_command");
}

holoscan::Contract SinusMotionOp::contract() const {
  holoscan::Contract contract;
  // Drive off arriving JointState so the command rate tracks the sim's tick.
  contract.trigger(holoscan::OnEach{joint_state_in});
  return contract;
}

holoscan::expected<void, holoscan::Error> SinusMotionOp::compute(
    holoscan::ExecutionContext& /*context*/) {
  auto received = joint_state_in.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }

  // The UCX decoder preserves the simulator's wire timestamp as capture time.
  // Keep that timestamp through command publication so sine phase follows Sim.
  const auto capture_time = received->metadata.capture_time();
  const std::int64_t sim_time_ns = capture_time.timestamp_ns;
  const double now_seconds = static_cast<double>(sim_time_ns) * 1e-9;
  if (start_time_seconds_ < 0.0) {
    start_time_seconds_ = now_seconds;
  }
  const double t = now_seconds - start_time_seconds_;

  const auto n = static_cast<std::size_t>(params_.num_joints);
  const double sin_value =
      params_.sin_amplitude * std::sin(2.0 * std::numbers::pi * params_.sin_frequency * t);

  std::vector<float> position(n);
  for (std::size_t i = 0; i < n; ++i) {
    position[i] = checked_command_value(params_.default_position[i] + sin_value, "position", i);
  }

  isaac::JointCommandCpuT command;
  command.position = std::move(position);
  command.velocity.assign(n, 0.0F);
  command.effort.assign(n, 0.0F);
  command.kp.assign(params_.kp.begin(), params_.kp.end());
  command.kd.assign(params_.kd.begin(), params_.kd.end());
  auto emitted = joint_command_out.emit(std::move(command),
                                        holoscan::EmitOptions{.capture_time = capture_time});
  if (!emitted) {
    return holoscan::make_unexpected(std::move(emitted).error());
  }
  return {};
}

}  // namespace isaac::os::apps::so101_sinus_motion_sim
