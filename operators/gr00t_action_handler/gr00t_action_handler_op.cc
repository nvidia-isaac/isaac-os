/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/gr00t_action_handler/gr00t_action_handler_op.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {
namespace {

std::vector<float> sample_chunk(const std::vector<float>& values, std::uint32_t joint_count,
                                std::uint32_t action_count, double raw_step) {
  const double step = std::clamp(raw_step, 0.0, static_cast<double>(action_count - 1U));
  const std::size_t low = static_cast<std::size_t>(step);
  const std::size_t high = std::min(low + 1U, static_cast<std::size_t>(action_count - 1U));
  const float alpha = static_cast<float>(step - static_cast<double>(low));
  std::vector<float> result(joint_count);
  for (std::size_t joint = 0; joint < joint_count; ++joint) {
    const std::size_t offset = joint * action_count;
    result[joint] = values[offset + low] + (values[offset + high] - values[offset + low]) * alpha;
  }
  return result;
}

holoscan::EmitOptions forwarded_options(const holoscan::SampleMetadata& metadata) {
  holoscan::EmitOptions options{.flags = metadata.flags};
  const holoscan::TimePoint capture_time = metadata.capture_time();
  if (capture_time.valid()) {
    options.capture_time = capture_time;
  }
  if (metadata.frame_id != 0U) {
    options.frame_id = metadata.frame_id;
  }
  if (metadata.trace_id != 0U) {
    options.trace = holoscan::TraceContext::root(metadata.trace_id);
  }
  return options;
}

}  // namespace

void Gr00tActionHandlerOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(control_command, "in_control_command").queue_depth(1U);
  spec.input(action_chunk, "in_action_chunk").queue_depth(1U);
  spec.output(action_command, "out_action_command");
  spec.output(forwarded_control_command, "out_control_command");
}

holoscan::Contract Gr00tActionHandlerOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{control_command});
  contract.input(action_chunk, holoscan::LatestWithin{.max_age = std::chrono::nanoseconds::max()});
  contract.on_missing(action_chunk, holoscan::SafeDefault{});
  return contract;
}

void Gr00tActionHandlerOp::start() {
  if (!std::isfinite(params().action_interval_s) || params().action_interval_s <= 0.0) {
    throw std::runtime_error("The trained action interval must be a positive finite duration.");
  }
  if (!std::isfinite(params().handover_blend_steps) || params().handover_blend_steps < 0.0) {
    throw std::runtime_error(
        "The action handover blend length must be a finite nonnegative value.");
  }
  const double max_interval_seconds =
      static_cast<double>(std::chrono::nanoseconds::max().count()) * 1e-9;
  if (params().action_interval_s >= max_interval_seconds) {
    throw std::runtime_error("The trained action interval must fit in nanoseconds.");
  }
  action_interval_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::duration<double>(params().action_interval_s))
                            .count();
  if (action_interval_ns_ <= 0) {
    throw std::runtime_error("The trained action interval must be at least one nanosecond.");
  }
  latest_chunk_ = {};
  latest_image_time_ns_ = 0;
  last_command_time_ns_ = 0;
  last_position_.clear();
  handover_offset_.clear();
  handover_start_step_ = 0.0;
  started_ = false;
  warned_missing_image_time_ = false;
  warned_future_image_time_ = false;
  warned_expired_ = false;
}

holoscan::expected<void, holoscan::Error> Gr00tActionHandlerOp::compute(
    holoscan::ExecutionContext& context) {
  auto received_control = control_command.receive();
  if (!received_control) {
    return holoscan::make_unexpected(std::move(received_control).error());
  }
  auto now = context.clock().now();
  if (!now) {
    return holoscan::make_unexpected(std::move(now).error());
  }

  bool chunk_accepted = false;
  if (!action_chunk.is_missing()) {
    auto received = action_chunk.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    if (!received->metadata.valid() || !validate_joint_command_chunk(received->data)) {
      HOLOSCAN_LOG_ERROR(
          "The GR00T action chunk is invalid or its dimensions do not match its joint and action "
          "counts; retaining the previous valid chunk.");
    } else {
      latest_chunk_ = std::move(received->data);
      latest_image_time_ns_ = latest_chunk_.image_capture_time_ns;
      if (latest_image_time_ns_ <= 0) {
        latest_image_time_ns_ = now->timestamp_ns;
        if (!warned_missing_image_time_) {
          warned_missing_image_time_ = true;
          HOLOSCAN_LOG_WARN(
              "The GR00T action chunk has no image capture time. Action selection starts from "
              "chunk arrival and does not account for inference latency.");
        }
      } else if (latest_image_time_ns_ > now->timestamp_ns) {
        latest_image_time_ns_ = now->timestamp_ns;
        if (!warned_future_image_time_) {
          warned_future_image_time_ = true;
          HOLOSCAN_LOG_WARN(
              "The GR00T action chunk has an image capture time in the future. Action selection "
              "starts from chunk arrival.");
        }
      }
      started_ = true;
      chunk_accepted = true;
      warned_expired_ = false;
    }
  }

  if (started_) {
    const std::int64_t elapsed_ns =
        std::max<std::int64_t>(0, now->timestamp_ns - latest_image_time_ns_);
    const double raw_step =
        static_cast<double>(elapsed_ns) / static_cast<double>(action_interval_ns_);
    const double last_step = static_cast<double>(latest_chunk_.action_count - 1U);
    if (raw_step > last_step && !warned_expired_) {
      warned_expired_ = true;
      HOLOSCAN_LOG_WARN(
          "The GR00T action chunk expired at step {:.2f} of {:.0f}. The final action is held until "
          "the next chunk arrives.",
          raw_step, last_step);
    }

    if (chunk_accepted && params().handover_blend_steps > 0.0) {
      if (last_position_.size() == latest_chunk_.joint_count) {
        const double since_last_command = static_cast<double>(
            std::max<std::int64_t>(0, now->timestamp_ns - last_command_time_ns_));
        // The new plan has no value before step zero, so sampling clamps instead of extrapolating
        // when the image is newer than the previous control command.
        const auto previous_plan_position = sample_chunk(
            latest_chunk_.position, latest_chunk_.joint_count, latest_chunk_.action_count,
            raw_step - since_last_command / static_cast<double>(action_interval_ns_));
        handover_offset_.resize(latest_chunk_.joint_count);
        for (std::size_t joint = 0; joint < latest_chunk_.joint_count; ++joint) {
          handover_offset_[joint] = last_position_[joint] - previous_plan_position[joint];
        }
        handover_start_step_ = raw_step;
      } else {
        handover_offset_.clear();
      }
    }

    isaac::JointCommandCpuT selected;
    selected.position = sample_chunk(latest_chunk_.position, latest_chunk_.joint_count,
                                     latest_chunk_.action_count, raw_step);
    selected.velocity.assign(latest_chunk_.joint_count, 0.0F);
    selected.effort = params().feedforward_torque
                          ? sample_chunk(latest_chunk_.effort, latest_chunk_.joint_count,
                                         latest_chunk_.action_count, raw_step)
                          : std::vector<float>(latest_chunk_.joint_count, 0.0F);
    selected.kp = latest_chunk_.kp;
    selected.kd = latest_chunk_.kd;
    if (!handover_offset_.empty()) {
      const float weight = static_cast<float>(std::clamp(
          1.0 - (raw_step - handover_start_step_) / params().handover_blend_steps, 0.0, 1.0));
      for (std::size_t joint = 0; joint < selected.position.size(); ++joint) {
        selected.position[joint] += handover_offset_[joint] * weight;
      }
      if (weight == 0.0F) {
        handover_offset_.clear();
      }
    }
    if (params().handover_blend_steps > 0.0) {
      last_position_ = selected.position;
      last_command_time_ns_ = now->timestamp_ns;
    }

    holoscan::TimePoint action_time = received_control->metadata.capture_time();
    if (!action_time.valid()) {
      action_time = *now;
    }
    auto emitted = action_command.emit(std::move(selected),
                                       holoscan::EmitOptions{.capture_time = action_time});
    if (!emitted) {
      return emitted;
    }
  }

  return forwarded_control_command.emit(std::move(received_control->data),
                                        forwarded_options(received_control->metadata));
}

}  // namespace isaac::os::ops
