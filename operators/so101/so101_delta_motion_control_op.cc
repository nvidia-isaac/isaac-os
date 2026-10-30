/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/so101/so101_delta_motion_control_op.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {
namespace {

constexpr std::size_t kArmJointCount = 4U;
constexpr std::size_t kJointCount = 6U;

bool valid_state(const isaac::StateSampleCpuT& sample) {
  return sample.position.size() == kJointCount &&
         std::all_of(sample.position.begin(), sample.position.end(),
                     [](float value) { return std::isfinite(value); });
}

}  // namespace

bool validate_so101_delta_motion_params(const So101DeltaMotionParams& params) {
  const auto finite = [](const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  if (!std::isfinite(params.action_scale) || !std::isfinite(params.action_max_step_size) ||
      !std::isfinite(params.motion_max_step_size) || !std::isfinite(params.target_expiry) ||
      params.action_scale <= 0.0 || params.action_max_step_size <= 0.0 ||
      params.motion_max_step_size <= 0.0 || params.target_expiry <= 0.0 ||
      params.fixed_joint_positions.size() != 2U ||
      params.joint_pos_limits_lower.size() != kArmJointCount ||
      params.joint_pos_limits_upper.size() != kArmJointCount ||
      !finite(params.fixed_joint_positions) || !finite(params.joint_pos_limits_lower) ||
      !finite(params.joint_pos_limits_upper)) {
    return false;
  }
  for (std::size_t index = 0; index < kArmJointCount; ++index) {
    if (params.joint_pos_limits_lower[index] > params.joint_pos_limits_upper[index]) {
      return false;
    }
  }
  return true;
}

void So101DeltaMotionControlOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(delta, "in_delta").queue_depth(1U);
  spec.input(state, "in_state").queue_depth(1U);
  spec.output(command, "out_command");
}

holoscan::Contract So101DeltaMotionControlOp::contract() const {
  const auto target_expiry = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(params().target_expiry));
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{state});
  contract.input(delta, holoscan::LatestWithin{.max_age = target_expiry});
  contract.on_missing(delta, holoscan::SafeDefault{});
  return contract;
}

void So101DeltaMotionControlOp::start() {
  if (!validate_so101_delta_motion_params(params())) {
    throw std::invalid_argument("The SO-101 motion controller has invalid parameters.");
  }
}

void So101DeltaMotionControlOp::stop() {
  HOLOSCAN_LOG_INFO(
      "Sent {} joint commands for {} joint states; {} policy {} expired before the next one "
      "arrived.",
      commands_, states_, expired_targets_, expired_targets_ == 1U ? "target" : "targets");
}

holoscan::expected<void, holoscan::Error> So101DeltaMotionControlOp::compute(
    holoscan::ExecutionContext& context) {
  const auto now = context.clock().now();
  if (!now) {
    return holoscan::make_unexpected(std::move(now).error());
  }
  auto received_state = state.receive();
  if (!received_state) {
    return holoscan::make_unexpected(std::move(received_state).error());
  }
  const isaac::StateSampleCpuT& measured = received_state->data;
  if (!valid_state(measured)) {
    has_target_ = false;
    return {};
  }

  ++states_;

  const So101DeltaMotionParams& config = params();
  if (has_target_) {
    const bool expired =
        static_cast<double>((*now).timestamp_ns - target_time_ns_) / 1'000'000'000.0 >
        config.target_expiry;
    if (expired) ++expired_targets_;
    if (expired || target_fault_epoch_ != measured.fault_epoch ||
        measured.source_sequence < target_sequence_) {
      has_target_ = false;
    }
  }
  if (!delta.is_missing()) {
    auto received_delta = delta.receive();
    if (!received_delta) {
      return holoscan::make_unexpected(std::move(received_delta).error());
    }
    const isaac::PolicyDeltaCpuT& action = received_delta->data;
    const bool accepts_target =
        action.action.size() == kArmJointCount && action.fault_epoch == measured.fault_epoch &&
        action.source_sequence <= measured.source_sequence &&
        (!has_accepted_delta_ || action.fault_epoch != target_fault_epoch_ ||
         action.source_sequence > target_sequence_) &&
        std::all_of(action.action.begin(), action.action.end(),
                    [](float value) { return std::isfinite(value); });
    if (accepts_target) {
      for (std::size_t index = 0; index < kArmJointCount; ++index) {
        const float desired = measured.position[index] +
                              action.action[index] * static_cast<float>(config.action_scale);
        const float limited =
            std::clamp(desired, static_cast<float>(config.joint_pos_limits_lower[index]),
                       static_cast<float>(config.joint_pos_limits_upper[index]));
        target_[index] = std::clamp(
            limited, measured.position[index] - static_cast<float>(config.action_max_step_size),
            measured.position[index] + static_cast<float>(config.action_max_step_size));
      }
      target_[4] = static_cast<float>(config.fixed_joint_positions[0]);
      target_[5] = static_cast<float>(config.fixed_joint_positions[1]);
      target_sequence_ = action.source_sequence;
      target_fault_epoch_ = action.fault_epoch;
      target_time_ns_ = (*now).timestamp_ns;
      has_accepted_delta_ = true;
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
  isaac::JointCommandCpuT& message = output->value();
  message.position.resize(kJointCount);
  // The SO-101 sink uses position control. Populate the unused channels to preserve the shared
  // JointCommandCpu invariant that every command array has one entry per joint.
  message.velocity.assign(kJointCount, 0.0F);
  message.effort.assign(kJointCount, 0.0F);
  message.kp.assign(kJointCount, 0.0F);
  message.kd.assign(kJointCount, 0.0F);
  for (std::size_t index = 0; index < kJointCount; ++index) {
    message.position[index] = std::clamp(
        target_[index], measured.position[index] - static_cast<float>(config.motion_max_step_size),
        measured.position[index] + static_cast<float>(config.motion_max_step_size));
  }
  holoscan::EmitOptions options{.capture_time = received_state->metadata.capture_time()};
  if (received_state->metadata.trace_id != 0U) {
    options.trace = holoscan::TraceContext::root(received_state->metadata.trace_id);
  }
  auto emitted = command.emit(std::move(*output), options);
  if (emitted) ++commands_;
  return emitted;
}

}  // namespace isaac::os::ops
