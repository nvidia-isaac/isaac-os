/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "applications/so101_reach/mock/mock_operators.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <holoscan/logger/logger.hpp>

namespace isaac::os {
namespace {

std::chrono::nanoseconds validated_state_period(double period_seconds) {
  if (!std::isfinite(period_seconds) || period_seconds <= 0.0) {
    HOLOSCAN_LOG_ERROR("The SO-101 mock state source requires a positive finite period.");
    throw std::invalid_argument("The SO-101 mock state source requires a positive finite period.");
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(period_seconds));
}

}  // namespace

void MockStateSourceOp::setup(holoscan::OperatorSpec& spec) { spec.output(state, "out_state"); }

holoscan::Contract MockStateSourceOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnClock{.period = validated_state_period(params().period)});
  return contract;
}

void MockStateSourceOp::start() { (void)validated_state_period(params().period); }

holoscan::expected<void, holoscan::Error> MockStateSourceOp::compute(
    holoscan::ExecutionContext& context) {
  const auto now = context.clock().now();
  if (!now) {
    return holoscan::make_unexpected(std::move(now).error());
  }
  auto output = state.allocate();
  if (!output) {
    return holoscan::make_unexpected(std::move(output).error());
  }
  isaac::StateSampleCpuT& message = output->value();
  constexpr std::array<float, 6U> kPosition{0.0F, 0.0F, 0.0F, 1.57000005F, 0.0F, 0.0F};
  message.position.assign(kPosition.begin(), kPosition.end());
  message.velocity = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  message.source_sequence = ++sequence_;
  message.fault_epoch = 0U;
  return state.emit(std::move(*output), holoscan::EmitOptions{
                                            .capture_time = *now,
                                            .trace = holoscan::TraceContext::root(sequence_),
                                        });
}

void MockCommandSinkOp::reset(std::size_t expected_count) {
  const std::lock_guard lock(mutex_);
  expected_count_ = expected_count;
  received_count_ = 0U;
  valid_ = true;
}

bool MockCommandSinkOp::wait_for_expected_count(std::chrono::duration<double> timeout) {
  std::unique_lock lock(mutex_);
  return changed_.wait_for(lock, timeout, [] { return received_count_ >= expected_count_; });
}

bool MockCommandSinkOp::valid() {
  const std::lock_guard lock(mutex_);
  return valid_;
}

void MockCommandSinkOp::setup(holoscan::OperatorSpec& spec) { spec.input(command, "in_command"); }

holoscan::Contract MockCommandSinkOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{command});
  return contract;
}

holoscan::expected<void, holoscan::Error> MockCommandSinkOp::compute(
    holoscan::ExecutionContext& context) {
  const auto sink_time = context.clock().now();
  if (!sink_time) {
    return holoscan::make_unexpected(std::move(sink_time).error());
  }
  auto received = command.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  const isaac::JointCommandCpuT& message = received->data;
  const holoscan::TimePoint capture_time = received->metadata.capture_time();
  const double end_to_end_latency_ms =
      capture_time.valid()
          ? static_cast<double>((*sink_time).timestamp_ns - capture_time.timestamp_ns) / 1'000'000.0
          : -1.0;
  const auto finite = [](float value) { return std::isfinite(value); };
  const bool message_valid =
      message.position.size() == 6U && message.velocity.size() == 6U &&
      message.effort.size() == 6U && message.kp.size() == 6U && message.kd.size() == 6U &&
      std::all_of(message.position.begin(), message.position.end(), finite) &&
      std::all_of(message.velocity.begin(), message.velocity.end(), finite) &&
      std::all_of(message.effort.begin(), message.effort.end(), finite) &&
      std::all_of(message.kp.begin(), message.kp.end(), finite) &&
      std::all_of(message.kd.begin(), message.kd.end(), finite);
  {
    const std::lock_guard lock(mutex_);
    valid_ = valid_ && message_valid;
    ++received_count_;
    if (message_valid && received_count_ % 10U == 0U) {
      HOLOSCAN_LOG_INFO(
          "SO-101 mock command {} received: trace {}, end-to-end latency {:.3f} ms, target "
          "[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}].",
          received_count_, received->metadata.trace_id, end_to_end_latency_ms, message.position[0],
          message.position[1], message.position[2], message.position[3], message.position[4],
          message.position[5]);
    }
  }
  changed_.notify_all();
  return {};
}

}  // namespace isaac::os
