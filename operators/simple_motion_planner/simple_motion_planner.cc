/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/simple_motion_planner/simple_motion_planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {
namespace {

using namespace std::chrono_literals;

/// Reports whether the command sizes every field to `joint_count` and carries finite positions.
bool valid_command(const isaac::JointCommandCpuT& command, int joint_count) {
  const auto finite = [](float value) { return std::isfinite(value); };
  return static_cast<int>(command.position.size()) == joint_count &&
         static_cast<int>(command.velocity.size()) == joint_count &&
         static_cast<int>(command.effort.size()) == joint_count &&
         static_cast<int>(command.kp.size()) == joint_count &&
         static_cast<int>(command.kd.size()) == joint_count &&
         std::all_of(command.position.begin(), command.position.end(), finite);
}

}  // namespace

bool validate_simple_motion_planner_params(const SimpleMotionPlannerParams& params) {
  return params.num_motors > 0 && std::isfinite(params.max_step_size) && params.max_step_size > 0.0;
}

void SimpleMotionPlannerOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(target, "in_target").queue_depth(1U);
  spec.input(state, "in_state").queue_depth(1U);
  spec.output(command, "out_command");
}

holoscan::Contract SimpleMotionPlannerOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{state});
  contract.input(target, holoscan::LatestWithin{.max_age = 40ms});
  contract.on_missing(target, holoscan::SafeDefault{});
  return contract;
}

void SimpleMotionPlannerOp::start() {
  if (!validate_simple_motion_planner_params(params())) {
    HOLOSCAN_LOG_ERROR("The simple motion planner has invalid parameters.");
    throw std::runtime_error("The simple motion planner has invalid parameters.");
  }
  target_position_.assign(static_cast<std::size_t>(params().num_motors), 0.0F);
  has_target_ = false;
}

holoscan::expected<void, holoscan::Error> SimpleMotionPlannerOp::compute(
    holoscan::ExecutionContext&) {
  auto received_state = state.receive();
  if (!received_state) {
    return holoscan::make_unexpected(std::move(received_state).error());
  }
  const isaac::StateSampleCpuT& measured = received_state->data;
  const SimpleMotionPlannerParams& config = params();
  if (static_cast<int>(measured.position.size()) != config.num_motors ||
      !std::all_of(measured.position.begin(), measured.position.end(),
                   [](float value) { return std::isfinite(value); })) {
    return {};
  }
  if (!target.is_missing()) {
    auto received_target = target.receive();
    if (!received_target) {
      return holoscan::make_unexpected(std::move(received_target).error());
    }
    const isaac::JointCommandCpuT& desired = received_target->data;
    if (valid_command(desired, config.num_motors)) {
      target_position_.assign(desired.position.begin(), desired.position.end());
      has_target_ = true;
    }
  }
  if (!has_target_) {
    return {};
  }

  auto output = command.allocate();
  if (!output) {
    return holoscan::make_unexpected(std::move(output).error());
  }
  const std::size_t joint_count = static_cast<std::size_t>(config.num_motors);
  isaac::JointCommandCpuT& message = output->value();
  message.position.resize(joint_count);
  message.velocity.assign(joint_count, 0.0F);
  message.effort.assign(joint_count, 0.0F);
  message.kp.assign(joint_count, 0.0F);
  message.kd.assign(joint_count, 0.0F);
  const float step = static_cast<float>(config.max_step_size);
  for (std::size_t index = 0; index < joint_count; ++index) {
    message.position[index] = std::clamp(target_position_[index], measured.position[index] - step,
                                         measured.position[index] + step);
  }
  holoscan::EmitOptions options{.capture_time = received_state->metadata.capture_time()};
  if (received_state->metadata.trace_id != 0U) {
    options.trace = holoscan::TraceContext::root(received_state->metadata.trace_id);
  }
  return command.emit(std::move(*output), options);
}

}  // namespace isaac::os::ops
