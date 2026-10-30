/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/sequential_action_handler/sequential_action_handler_op.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {

void SequentialActionHandlerOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(action_chunk, "in_action_chunk").queue_depth(1U);
  spec.output(command, "out_joint_command");
}

holoscan::Contract SequentialActionHandlerOp::contract() const {
  const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(params().dt));
  holoscan::Contract contract;
  contract.trigger(holoscan::OnClock{.period = period});
  contract.input(action_chunk, holoscan::LatestWithin{.max_age = std::chrono::nanoseconds::max()});
  contract.on_missing(action_chunk, holoscan::SafeDefault{});
  return contract;
}

void SequentialActionHandlerOp::start() {
  if (!std::isfinite(params().dt) || params().dt <= 0.0) {
    throw std::runtime_error("The action interval must be a positive finite duration.");
  }
  period_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::duration<double>(params().dt))
                   .count();
  if (period_ns_ <= 0) {
    throw std::runtime_error("The action interval must be at least one nanosecond.");
  }
  latest_chunk_ = {};
  latest_chunk_time_ns_ = 0;
  started_ = false;
}

holoscan::expected<void, holoscan::Error> SequentialActionHandlerOp::compute(
    holoscan::ExecutionContext& context) {
  auto now = context.clock().now();
  if (!now) {
    return holoscan::make_unexpected(std::move(now).error());
  }

  if (!action_chunk.is_missing()) {
    auto received = action_chunk.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    if (!validate_joint_command_chunk(received->data)) {
      HOLOSCAN_LOG_ERROR(
          "The action chunk dimensions do not match its joint and action counts; retaining the "
          "previous valid chunk.");
    } else {
      latest_chunk_ = std::move(received->data);
      latest_chunk_time_ns_ = now->timestamp_ns;
      started_ = true;
    }
  }

  if (!started_) {
    return {};
  }

  const std::int64_t elapsed_ns =
      std::max<std::int64_t>(0, now->timestamp_ns - latest_chunk_time_ns_);
  const std::size_t action_index = std::min<std::size_t>(
      static_cast<std::size_t>(elapsed_ns / period_ns_), latest_chunk_.action_count - 1U);

  isaac::JointCommandCpuT selected;
  selected.position.resize(latest_chunk_.joint_count);
  selected.velocity.assign(latest_chunk_.joint_count, 0.0F);
  selected.effort.resize(latest_chunk_.joint_count);
  selected.kp = latest_chunk_.kp;
  selected.kd = latest_chunk_.kd;
  for (std::size_t joint_index = 0; joint_index < latest_chunk_.joint_count; ++joint_index) {
    const std::size_t value_index = joint_index * latest_chunk_.action_count + action_index;
    selected.position[joint_index] = latest_chunk_.position[value_index];
    selected.effort[joint_index] = latest_chunk_.effort[value_index];
  }
  return command.emit(std::move(selected), holoscan::EmitOptions{.capture_time = *now});
}

}  // namespace isaac::os::ops
