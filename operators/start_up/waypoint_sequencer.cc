/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/start_up/waypoint_sequencer.h"

#include <cmath>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {

WaypointSequencer::WaypointSequencer(std::vector<Waypoint> waypoints, double vel_settled_threshold,
                                     std::size_t joint_count)
    : waypoints_(std::move(waypoints)),
      vel_settled_threshold_(vel_settled_threshold),
      joint_count_(joint_count),
      start_pose_(joint_count, 0.0F),
      command_(joint_count, 0.0F) {
  if (waypoints_.empty()) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "The startup sequence must contain at least one waypoint.");
  }
  for (const Waypoint& waypoint : waypoints_) {
    if (waypoint.positions.size() != joint_count_) {
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kInvalidArgument,
          "Startup waypoint '" + waypoint.name + "' does not give a position for every joint.");
    }
    if (waypoint.interp_steps <= 0) {
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kInvalidArgument,
          "Startup waypoint '" + waypoint.name + "' must take at least one step to reach.");
    }
    if (waypoint.dwell_steps < 0) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                   "Startup waypoint '" + waypoint.name +
                                       "' cannot be held for a negative number of steps.");
    }
  }

  HOLOSCAN_LOG_INFO("The startup sequence has {} waypoints.", waypoints_.size());
}

const std::vector<float>& WaypointSequencer::step(const std::vector<float>* measured_position,
                                                  const std::vector<float>* measured_velocity,
                                                  std::optional<Handover> handover) {
  if (handover) {
    if (waypoints_.size() != 1U || handover->target.size() != joint_count_ ||
        !std::isfinite(handover->fraction)) {
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kInvalidArgument,
          "A startup handover needs one complete target and a finite progress fraction.");
    }
    waypoints_.front().positions = handover->target;
  }
  const bool position_valid =
      measured_position != nullptr && measured_position->size() == joint_count_;
  if (!start_pose_captured_) {
    if (!position_valid) {
      command_ = start_pose_;
      return command_;
    }
    start_pose_ = *measured_position;
    start_pose_captured_ = true;
  }

  if (at_final_waypoint()) {
    command_ = waypoints_.back().positions;
    return command_;
  }
  const Waypoint& waypoint = waypoints_[index_];
  const std::vector<float>& from = index_ == 0 ? start_pose_ : waypoints_[index_ - 1U].positions;
  const float fraction =
      smoothstep(handover ? handover->fraction
                          : static_cast<float>(step_) / static_cast<float>(waypoint.interp_steps));
  for (std::size_t joint = 0; joint < joint_count_; ++joint) {
    command_[joint] = from[joint] + fraction * (waypoint.positions[joint] - from[joint]);
  }
  if (handover) {
    if (handover->fraction >= 1.0F) index_ = waypoints_.size();
    return command_;
  }
  const bool velocity_valid =
      measured_velocity != nullptr && measured_velocity->size() == joint_count_;
  if (position_valid && velocity_valid) {
    float fastest_joint = 0.0F;
    for (const float velocity : *measured_velocity) {
      fastest_joint = std::max(fastest_joint, std::abs(velocity));
    }
    ++step_;
    if (step_ >= waypoint.interp_steps + waypoint.dwell_steps &&
        fastest_joint < static_cast<float>(vel_settled_threshold_)) {
      HOLOSCAN_LOG_DEBUG("The startup pose '{}' settled.", waypoint.name);
      ++index_;
      step_ = 0;
    }
  }
  return command_;
}

}  // namespace isaac::os::ops
