/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_START_UP_WAYPOINT_SEQUENCER_H
#define ISAAC_OS_OPERATORS_START_UP_WAYPOINT_SEQUENCER_H

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace isaac::os::ops {
inline float smoothstep(float fraction) {
  fraction = std::clamp(fraction, 0.0F, 1.0F);
  return fraction * fraction * (3.0F - 2.0F * fraction);
}

/// Smoothstep interpolation and velocity-based settling between joint-space waypoints.
class WaypointSequencer {
 public:
  struct Waypoint {
    std::string name;
    std::vector<float> positions;
    int interp_steps{};
    /// Minimum hold before checking joint velocity.
    int dwell_steps{};
  };
  WaypointSequencer(std::vector<Waypoint> waypoints, double vel_settled_threshold,
                    std::size_t joint_count);
  /// A one-waypoint handover follows a changing target at an externally timed fraction.
  /// Completion depends on the fraction reaching one, rather than measured settling.
  struct Handover {
    const std::vector<float>& target;
    float fraction;
  };
  /// Returns the commanded positions; the reference is valid until the next call.
  const std::vector<float>& step(const std::vector<float>* measured_position,
                                 const std::vector<float>* measured_velocity,
                                 std::optional<Handover> handover = std::nullopt);
  [[nodiscard]] bool at_final_waypoint() const { return index_ >= waypoints_.size(); }

 private:
  std::vector<Waypoint> waypoints_;
  double vel_settled_threshold_{};
  std::size_t joint_count_{};
  std::size_t index_{};
  int step_{};
  bool start_pose_captured_{};
  std::vector<float> start_pose_;
  std::vector<float> command_;
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_START_UP_WAYPOINT_SEQUENCER_H
