/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/sinus_motion/sinus_motion_op.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace isaac::os::ops {
namespace {

constexpr double kNanosecondsPerSecond = 1.0e9;

/// Reports whether every entry is finite.
bool finite_vector(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

}  // namespace

bool validate_sinus_motion_params(const SinusMotionParams& params) {
  return params.num_joints > 0 &&
         static_cast<int>(params.default_pos.size()) == params.num_joints &&
         finite_vector(params.default_pos) && std::isfinite(params.sin_amplitude) &&
         std::isfinite(params.sin_frequency) && params.sin_amplitude >= 0.0 &&
         params.sin_frequency >= 0.0;
}

void SinusMotionOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(state, "in_state").queue_depth(1U);
  spec.output(command, "out_command");
}

holoscan::Contract SinusMotionOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{state});
  return contract;
}

void SinusMotionOp::start() {
  if (!validate_sinus_motion_params(params())) {
    throw std::runtime_error("The sinus motion operator has invalid parameters.");
  }
}

holoscan::expected<void, holoscan::Error> SinusMotionOp::compute(
    holoscan::ExecutionContext& context) {
  const auto now = context.clock().now();
  if (!now) {
    return holoscan::make_unexpected(std::move(now).error());
  }
  auto received = state.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  const isaac::StateSampleCpuT& sample = received->data;
  const SinusMotionParams& config = params();
  if (static_cast<int>(sample.position.size()) != config.num_joints) {
    return {};
  }
  if (start_time_ns_ < 0) {
    start_time_ns_ = (*now).timestamp_ns;
  }
  const double elapsed_s =
      static_cast<double>((*now).timestamp_ns - start_time_ns_) / kNanosecondsPerSecond;
  const float offset = static_cast<float>(
      config.sin_amplitude * std::sin(2.0 * std::numbers::pi * config.sin_frequency * elapsed_s));

  auto output = command.allocate();
  if (!output) {
    return holoscan::make_unexpected(std::move(output).error());
  }
  const std::size_t joint_count = static_cast<std::size_t>(config.num_joints);
  isaac::JointCommandCpuT& message = output->value();
  message.position.resize(joint_count);
  message.velocity.assign(joint_count, 0.0F);
  message.effort.assign(joint_count, 0.0F);
  message.kp.assign(joint_count, 0.0F);
  message.kd.assign(joint_count, 0.0F);
  for (std::size_t index = 0; index < joint_count; ++index) {
    message.position[index] = static_cast<float>(config.default_pos[index]) + offset;
  }
  holoscan::EmitOptions options{.capture_time = received->metadata.capture_time()};
  if (received->metadata.trace_id != 0U) {
    options.trace = holoscan::TraceContext::root(received->metadata.trace_id);
  }
  return command.emit(std::move(*output), options);
}

}  // namespace isaac::os::ops
