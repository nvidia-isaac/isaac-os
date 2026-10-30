/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/so101/so101_io_ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/logger/logger.hpp>
#include <holoscan/time/time_point.hpp>

namespace isaac::os::ops {
namespace {

bool has_expected_joint_order(const so101::FeetechSerialDriver& driver) {
  const std::vector<std::string> names = driver.get_ordered_joint_names();
  return names.size() == so101::kSo101JointNames.size() &&
         std::equal(names.begin(), names.end(), so101::kSo101JointNames.begin(),
                    [](const std::string& actual, std::string_view expected) {
                      return actual == expected;
                    });
}

bool valid_command(const isaac::JointCommandCpuT& command) {
  constexpr std::size_t kJointCount = 6U;
  const auto finite = [](float value) { return std::isfinite(value); };
  return command.position.size() == kJointCount && command.velocity.size() == kJointCount &&
         command.effort.size() == kJointCount && command.kp.size() == kJointCount &&
         command.kd.size() == kJointCount &&
         std::all_of(command.position.begin(), command.position.end(), finite) &&
         std::all_of(command.velocity.begin(), command.velocity.end(), finite) &&
         std::all_of(command.effort.begin(), command.effort.end(), finite) &&
         std::all_of(command.kp.begin(), command.kp.end(), finite) &&
         std::all_of(command.kd.begin(), command.kd.end(), finite);
}

}  // namespace

So101StateSourceOp::So101StateSourceOp(std::shared_ptr<so101::FeetechSerialDriver> driver)
    : driver_(std::move(driver)) {}

void So101StateSourceOp::setup(holoscan::OperatorSpec& spec) { spec.output(state, "out_state"); }

holoscan::Contract So101StateSourceOp::contract() const {
  const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(params().period));
  holoscan::Contract contract;
  contract.trigger(holoscan::OnClock{.period = period});
  return contract;
}

void So101StateSourceOp::start() {
  read_failure_since_.reset();
  completion_requested_ = false;
  if (!driver_ || driver_->num_joints() != 6 || !has_expected_joint_order(*driver_)) {
    throw std::invalid_argument(
        "The SO-101 state source requires the expected six-joint calibration order.");
  }
  if (!std::isfinite(params().period) || params().period <= 0.0) {
    throw std::invalid_argument("The SO-101 state source requires a positive finite period.");
  }
  positions_.resize(6U);
  velocities_.resize(6U);
}

holoscan::expected<void, holoscan::Error> So101StateSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (completion_requested_) return {};
  const auto now = context.clock().now();
  if (!now) {
    return holoscan::make_unexpected(std::move(now).error());
  }
  if (!driver_->read_present_positions_and_velocities(positions_, velocities_)) {
    ++fault_epoch_;
    // Wall time bounds a hardware outage even when the graph uses a synthetic clock.
    const auto failed_at = std::chrono::steady_clock::now();
    if (!read_failure_since_) {
      read_failure_since_ = failed_at;
      HOLOSCAN_LOG_WARN("Unable to read the SO-101 joint state. Retrying for up to two seconds.");
    } else if (failed_at - *read_failure_since_ >= std::chrono::seconds(2)) {
      HOLOSCAN_LOG_ERROR(
          "Stopping the application after two seconds without SO-101 joint state. "
          "Check the arm power and USB connection before restarting.");
      context.request_completion();
      completion_requested_ = true;
    }
    return {};
  }
  read_failure_since_.reset();
  auto output = state.allocate();
  if (!output) {
    return holoscan::make_unexpected(std::move(output).error());
  }
  isaac::StateSampleCpuT& sample = output->value();
  sample.position = positions_;
  sample.velocity = velocities_;
  sample.source_sequence = ++sequence_;
  sample.fault_epoch = fault_epoch_;
  return state.emit(std::move(*output),
                    holoscan::EmitOptions{.capture_time = *now,
                                          .trace = holoscan::TraceContext::root(sequence_)});
}

So101CommandSinkOp::So101CommandSinkOp(std::shared_ptr<so101::FeetechSerialDriver> driver,
                                       std::size_t expected_count)
    : driver_(std::move(driver)), expected_count_(expected_count) {}

void So101CommandSinkOp::setup(holoscan::OperatorSpec& spec) { spec.input(command, "in_command"); }

holoscan::Contract So101CommandSinkOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{command});
  return contract;
}

void So101CommandSinkOp::start() {
  if (!driver_ || driver_->num_joints() != 6 || !has_expected_joint_order(*driver_)) {
    throw std::invalid_argument(
        "The SO-101 command sink requires the expected six-joint calibration order.");
  }
  received_count_ = 0U;
  completion_requested_ = false;
}

void So101CommandSinkOp::stop() {
  if (driver_) driver_->hold_present_position();
}

holoscan::expected<void, holoscan::Error> So101CommandSinkOp::compute(
    holoscan::ExecutionContext& context) {
  auto received = command.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  if (context.cancellation().stop_requested()) {
    driver_->hold_present_position();
    return {};
  }
  if (completion_requested_) return {};
  if (!valid_command(received->data)) {
    HOLOSCAN_LOG_ERROR("Not commanding the SO-101 arm: the joint command is invalid.");
    return {};
  }
  if (!driver_->send_action(received->data.position)) {
    HOLOSCAN_LOG_ERROR("Unable to send the SO-101 joint command.");
  }
  if (expected_count_ != 0U) {
    ++received_count_;
    if (received_count_ >= expected_count_) {
      context.request_completion();
      completion_requested_ = true;
    }
  }
  return {};
}

}  // namespace isaac::os::ops
