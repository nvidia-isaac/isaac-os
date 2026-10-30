/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/joint_command_combiner/joint_command_combiner_op.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {

void JointCommandCombinerOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(primary_command, "in_primary_joint_command").queue_depth(1U);
  spec.input(overlay_command, "in_overlay_joint_command").queue_depth(1U);
  spec.output(command, "out_joint_command");
}

holoscan::Contract JointCommandCombinerOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{primary_command});
  // LatestWithin consumes each overlay sample once. SafeDefault keeps primary ticks running between
  // samples, while the operator cache preserves the latest overlay command.
  contract.input(overlay_command,
                 holoscan::LatestWithin{.max_age = std::chrono::nanoseconds::max()});
  contract.on_missing(overlay_command, holoscan::SafeDefault{});
  return contract;
}

void JointCommandCombinerOp::start() {
  const auto& config = params();
  if (config.joint_order.empty()) {
    throw std::runtime_error("The joint order must not be empty.");
  }

  const std::unordered_set<std::string> primary_joints(config.joint_names_from_primary.begin(),
                                                       config.joint_names_from_primary.end());
  const std::unordered_set<std::string> overlay_joints(config.joint_names_from_overlay.begin(),
                                                       config.joint_names_from_overlay.end());
  if (primary_joints.size() != config.joint_names_from_primary.size() ||
      overlay_joints.size() != config.joint_names_from_overlay.size()) {
    throw std::runtime_error("Each configured joint must appear only once in its command group.");
  }

  take_from_overlay_.clear();
  take_from_overlay_.reserve(config.joint_order.size());
  for (const auto& name : config.joint_order) {
    const bool from_primary = primary_joints.contains(name);
    const bool from_overlay = overlay_joints.contains(name);
    if (from_primary == from_overlay) {
      throw std::runtime_error("Joint '" + name + "' must belong to exactly one command group.");
    }
    take_from_overlay_.push_back(from_overlay);
  }
  cached_overlay_command_ = {};
  cached_overlay_flags_ = holoscan::SampleFlags::kNone;
  overlay_seen_ = false;
  first_tick_time_ = {};
  first_tick_seen_ = false;
}

bool JointCommandCombinerOp::command_size_is_valid(
    const isaac::JointCommandCpuT& value) const noexcept {
  const std::size_t count = take_from_overlay_.size();
  return value.position.size() >= count && value.velocity.size() >= count &&
         value.effort.size() >= count && value.kp.size() >= count && value.kd.size() >= count;
}

holoscan::expected<void, holoscan::Error> JointCommandCombinerOp::compute(
    holoscan::ExecutionContext&) {
  if (!overlay_command.is_missing()) {
    auto selected = overlay_command.receive();
    if (!selected) {
      return holoscan::make_unexpected(std::move(selected).error());
    }
    if (selected->metadata.valid() && command_size_is_valid(selected->data)) {
      cached_overlay_command_ = std::move(selected->data);
      cached_overlay_flags_ = selected->metadata.flags;
      overlay_seen_ = true;
    } else {
      HOLOSCAN_LOG_ERROR(
          "The overlay joint command is invalid or has fewer values than the configured joint "
          "count; retaining the previous overlay command.");
    }
  }

  auto received_primary = primary_command.receive();
  if (!received_primary) {
    return holoscan::make_unexpected(std::move(received_primary).error());
  }
  if (!received_primary->metadata.valid() || !command_size_is_valid(received_primary->data)) {
    HOLOSCAN_LOG_ERROR(
        "The primary joint command is invalid or has fewer values than the configured joint count; "
        "no combined command was sent.");
    return {};
  }

  if (!first_tick_seen_) {
    // Enable the learned-policy overlay after real elapsed time, independent of the runtime clock.
    first_tick_seen_ = true;
    first_tick_time_ = std::chrono::steady_clock::now();
  }
  const double elapsed_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - first_tick_time_).count();
  const bool use_overlay = overlay_seen_ && elapsed_s >= params().overlay_startup_delay_s;

  isaac::JointCommandCpuT combined = std::move(received_primary->data);
  if (use_overlay) {
    for (std::size_t index = 0; index < take_from_overlay_.size(); ++index) {
      if (!take_from_overlay_[index]) {
        continue;
      }
      combined.position[index] = cached_overlay_command_.position[index];
      combined.velocity[index] = cached_overlay_command_.velocity[index];
      combined.effort[index] = cached_overlay_command_.effort[index];
      combined.kp[index] = cached_overlay_command_.kp[index];
      combined.kd[index] = cached_overlay_command_.kd[index];
    }
  }
  const holoscan::SampleMetadata& primary_metadata = received_primary->metadata;
  const holoscan::TimePoint command_source = primary_metadata.capture_time();
  holoscan::EmitOptions options{
      .flags = primary_metadata.flags |
               (use_overlay ? cached_overlay_flags_ : holoscan::SampleFlags::kNone)};
  // The fallback remains usable, but does not yet contain all configured command sources.
  if (!use_overlay && !params().joint_names_from_overlay.empty()) {
    options.flags |= holoscan::SampleFlags::kDegraded;
  }
  if (command_source.valid()) {
    options.capture_time = command_source;
  }
  if (primary_metadata.frame_id != 0U) {
    options.frame_id = primary_metadata.frame_id;
  }
  if (primary_metadata.trace_id != 0U) {
    options.trace = holoscan::TraceContext::root(primary_metadata.trace_id);
  }
  return command.emit(std::move(combined), options);
}

}  // namespace isaac::os::ops
