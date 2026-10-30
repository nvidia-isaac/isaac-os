/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "applications/gr00t_n1_7_wbc_g1/mock/mock_operators.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/logger/logger.hpp>

namespace isaac::os {
namespace {

using namespace std::chrono_literals;

constexpr DLDataType kImageDtype{kDLUInt, 8U, 1U};
constexpr std::array<std::int64_t, 3U> kImageShape{480, 640, 3};
constexpr std::size_t kImageBytes = 480U * 640U * 3U;
const holoscan::TensorOutputSpec kDeviceImage{
    .representation =
        {
            .memory_kind = holoscan::MemoryKind::kCudaDevice,
            .dtype = kImageDtype,
            .rank = 3U,
        },
    .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = kImageBytes},
    .storage = holoscan::TensorOutputStorage::kRuntimePool,
};

bool command_is_valid(const isaac::JointCommandCpuT& command, std::size_t count) {
  const auto valid_values = [count](const std::vector<float>& values) {
    return values.size() == count &&
           std::ranges::all_of(values, [](float value) { return std::isfinite(value); });
  };
  return valid_values(command.position) && valid_values(command.velocity) &&
         valid_values(command.effort) && valid_values(command.kp) && valid_values(command.kd);
}

bool command_matches(const isaac::JointCommandCpuT& expected, const isaac::JointCommandCpuT& actual,
                     const std::vector<bool>& overlay_indices, bool match_overlay) {
  const auto matches = [&overlay_indices](const std::vector<float>& expected_values,
                                          const std::vector<float>& actual_values,
                                          bool select_overlay) {
    if (expected_values.size() != overlay_indices.size() ||
        actual_values.size() != overlay_indices.size()) {
      return false;
    }
    for (std::size_t index = 0; index < overlay_indices.size(); ++index) {
      if (overlay_indices[index] == select_overlay &&
          expected_values[index] != actual_values[index]) {
        return false;
      }
    }
    return true;
  };
  return matches(expected.position, actual.position, match_overlay) &&
         matches(expected.velocity, actual.velocity, match_overlay) &&
         matches(expected.effort, actual.effort, match_overlay) &&
         matches(expected.kp, actual.kp, match_overlay) &&
         matches(expected.kd, actual.kd, match_overlay);
}

struct CommandEvidence {
  isaac::JointCommandCpuT command;
  std::int64_t capture_time_ns;
};

}  // namespace

void MockRobotStateOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(wbc_joint_state, "out_wbc_joint_state");
  spec.output(gr00t_joint_state, "out_gr00t_joint_state");
  spec.output(imu, "out_imu");
}

holoscan::Contract MockRobotStateOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = 20ms});
  return result;
}

void MockRobotStateOp::start() {
  joint_state_message_.position.resize(joint_count_);
  joint_state_message_.velocity.resize(joint_count_);
  for (std::size_t index = 0; index < joint_count_; ++index) {
    joint_state_message_.position[index] = static_cast<float>(index + 1U) * 0.01F;
    joint_state_message_.velocity[index] = -static_cast<float>(index + 1U) * 0.02F;
  }
  imu_message_.orientation = holoscan::schema::Quaternionf{1.0F, 0.0F, 0.0F, 0.0F};
  imu_message_.angular_velocity = holoscan::schema::Vector3f{0.0F, 0.0F, 0.0F};
  imu_message_.linear_acceleration = holoscan::schema::Vector3f{0.0F, 0.0F, 0.0F};
}

holoscan::expected<void, holoscan::Error> MockRobotStateOp::compute(
    holoscan::ExecutionContext& context) {
  const auto source_start = context.clock().now();
  if (!source_start) {
    HOLOSCAN_LOG_ERROR("Unable to read the mock state clock: {}",
                       holoscan::runtime_error_message(source_start.error()));
    return holoscan::make_unexpected(std::move(source_start).error());
  }
  const holoscan::EmitOptions options{.capture_time = *source_start};
  // Publish matching IMU evidence before joint state triggers WBC.
  auto emitted = imu.emit(imu_message_, options);
  if (!emitted) {
    HOLOSCAN_LOG_ERROR("Unable to publish mock IMU state: {}",
                       holoscan::runtime_error_message(emitted.error()));
    return emitted;
  }
  emitted = wbc_joint_state.emit(joint_state_message_, options);
  if (!emitted) {
    HOLOSCAN_LOG_ERROR("Unable to publish mock joint state to WBC: {}",
                       holoscan::runtime_error_message(emitted.error()));
    return emitted;
  }
  emitted = gr00t_joint_state.emit(joint_state_message_, options);
  if (!emitted) {
    HOLOSCAN_LOG_ERROR("Unable to publish mock joint state to GR00T: {}",
                       holoscan::runtime_error_message(emitted.error()));
    return emitted;
  }
  return emitted;
}

void MockImageOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(image, "out_image").produces_tensor(kDeviceImage);
}

holoscan::Contract MockImageOp::contract() const {
  const double frame_rate = params().frame_rate;
  const double period_seconds = 1.0 / frame_rate;
  const double max_period_seconds =
      static_cast<double>(std::chrono::nanoseconds::max().count()) * 1e-9;
  if (!std::isfinite(frame_rate) || frame_rate <= 0.0 || !std::isfinite(period_seconds) ||
      period_seconds >= max_period_seconds) {
    throw std::invalid_argument("The camera frame rate must produce a positive finite period.");
  }
  const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(period_seconds));
  if (period <= 0ns) {
    throw std::invalid_argument(
        "The camera frame rate must produce a period of at least one nanosecond.");
  }
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = period});
  return result;
}

holoscan::expected<void, holoscan::Error> MockImageOp::compute(
    holoscan::ExecutionContext& context) {
  // Action selection uses the frame's actual production time, not its nominal scheduled time.
  auto capture_time = context.clock().now();
  if (!capture_time) {
    return holoscan::make_unexpected(std::move(capture_time).error());
  }
  auto loan = image.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = kImageShape, .dtype = kImageDtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  const int value = static_cast<int>(++sequence_ % 255U);
  if (writer->byte_size() != kImageBytes ||
      cudaMemsetAsync(writer->data(), value, kImageBytes, context.cuda_stream().get()) !=
          cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return image.emit(std::move(*loan), holoscan::EmitOptions{.capture_time = *capture_time});
}

MockCommandSinkOp::MockCommandSinkOp(std::size_t expected_count,
                                     const std::vector<std::string>& joint_order,
                                     const std::vector<std::string>& overlay_joints)
    : expected_count_(expected_count), joint_count_(joint_order.size()) {
  const std::unordered_set<std::string> overlay_names(overlay_joints.begin(), overlay_joints.end());
  overlay_indices_.reserve(joint_order.size());
  for (const auto& joint : joint_order) {
    overlay_indices_.push_back(overlay_names.contains(joint));
  }
  const bool has_overlay_joints =
      std::ranges::any_of(overlay_indices_, [](bool from_overlay) { return from_overlay; });
  const bool has_primary_joints =
      std::ranges::any_of(overlay_indices_, [](bool from_overlay) { return !from_overlay; });
  if (!has_overlay_joints || !has_primary_joints) {
    throw std::invalid_argument(
        "The GR00T and WBC mock requires at least one WBC-only and GR00T-only joint.");
  }
}

void MockCommandSinkOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(command, "in_joint_command");
  spec.input(action_command, "in_action_command");
  spec.input(primary_command, "in_primary_joint_command");
}

holoscan::Contract MockCommandSinkOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{command});
  result.input(action_command, holoscan::LatestWithin{.max_age = std::chrono::nanoseconds::max()});
  result.on_missing(action_command, holoscan::SafeDefault{});
  result.input(primary_command, holoscan::LatestWithin{.max_age = std::chrono::nanoseconds::max()});
  result.on_missing(primary_command, holoscan::SafeDefault{});
  return result;
}

holoscan::expected<void, holoscan::Error> MockCommandSinkOp::compute(
    holoscan::ExecutionContext& context) {
  std::optional<CommandEvidence> received_action_command;
  if (!action_command.is_missing()) {
    auto received_action = action_command.receive();
    if (!received_action) {
      return holoscan::make_unexpected(std::move(received_action).error());
    }
    if (!received_action->metadata.valid() ||
        !command_is_valid(received_action->data, joint_count_)) {
      HOLOSCAN_LOG_ERROR("The GR00T action is invalid for {} joints.", joint_count_);
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    received_action_command = CommandEvidence{
        .command = std::move(received_action->data),
        .capture_time_ns = received_action->metadata.capture_time().timestamp_ns,
    };
  }
  std::optional<CommandEvidence> received_primary_command;
  if (!primary_command.is_missing()) {
    auto received_primary = primary_command.receive();
    if (!received_primary) {
      return holoscan::make_unexpected(std::move(received_primary).error());
    }
    if (!received_primary->metadata.valid() ||
        !command_is_valid(received_primary->data, joint_count_)) {
      HOLOSCAN_LOG_ERROR("The WBC command is invalid for {} joints.", joint_count_);
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    received_primary_command = CommandEvidence{
        .command = std::move(received_primary->data),
        .capture_time_ns = received_primary->metadata.capture_time().timestamp_ns,
    };
  }
  auto received = command.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  if (!command_is_valid(received->data, joint_count_)) {
    HOLOSCAN_LOG_ERROR("The combined command is invalid for {} joints.", joint_count_);
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
  }
  ++received_count_;
  const holoscan::TimePoint combined_time = received->metadata.capture_time();
  if (!composition_validated_ && combined_time.valid() && received_action_command &&
      received_primary_command &&
      received_action_command->capture_time_ns == combined_time.timestamp_ns &&
      received_primary_command->capture_time_ns == combined_time.timestamp_ns &&
      command_matches(received_action_command->command, received->data, overlay_indices_, true) &&
      command_matches(received_primary_command->command, received->data, overlay_indices_, false)) {
    composition_validated_ = true;
  }
  if (!completion_requested_ && expected_count_ != 0U && received_count_ >= expected_count_) {
    if (!composition_validated_) {
      HOLOSCAN_LOG_ERROR("The combined command was not validated after {} messages.",
                         expected_count_);
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    context.request_completion();
    completion_requested_ = true;
  }
  return {};
}

}  // namespace isaac::os
