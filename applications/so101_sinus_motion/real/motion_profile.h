/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace isaac::os {

/// A configuration, feedback, or output failure in the bounded hardware diagnostic.
class MotionProfileError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Retry a failed position-hold attempt up to three times without releasing torque.
template <typename Driver>
bool hold_after_motion_profile(Driver& driver) {
  for (int attempt = 0; attempt < 3; ++attempt) {
    if (driver.hold_present_position()) return true;
  }
  return false;
}

/// Hold each shoulder-pan target for three seconds, writing it only once. Repeated nearby
/// sine targets otherwise mask the servo's own speed and acceleration profile.
template <typename Driver, typename Timing, typename Stop>
void measure_motion_profile(Driver& driver, std::ostream& output, Timing& timing, Stop stop) {
  using namespace std::chrono_literals;
  std::vector<float> origin(6U);
  if (!driver.read_present_positions(origin) || origin.size() != 6U) {
    throw MotionProfileError("Unable to read the six joint positions before the motion test.");
  }
  if (const auto names = driver.get_ordered_joint_names();
      names.size() != 6U || names.front() != "shoulder_pan") {
    throw MotionProfileError("Unable to identify the shoulder-pan joint for the motion test.");
  }
  output << "phase,elapsed_s,target_rad,position_rad,velocity_rad_s\n";
  output.flush();
  if (!output) {
    throw MotionProfileError("Unable to save the feedback header. The motion test will not start.");
  }
  // Three out-and-back trials; the other five joints keep their measured starting targets.
  for (int phase = 0; phase < 6 && !stop(); ++phase) {
    auto target = origin;
    target.front() += phase % 2 == 0 ? 0.20F : 0.0F;
    const auto start = timing.now();
    if (!driver.send_action(target)) {
      throw MotionProfileError("Unable to send the motion-test target. Stopping the test.");
    }
    auto next = start;
    float final_position = origin.front();
    while (!stop() && timing.now() - start < 3s) {
      std::vector<float> position(6U);
      std::vector<float> velocity(6U);
      if (!driver.read_present_positions_and_velocities(position, velocity) ||
          position.size() != 6U || velocity.size() != 6U) {
        throw MotionProfileError("Unable to read joint feedback. Stopping the motion test.");
      }
      final_position = position.front();
      const double elapsed = std::chrono::duration<double>(timing.now() - start).count();
      output << phase << ',' << elapsed << ',' << target.front() << ',' << position.front() << ','
             << velocity.front() << '\n';
      if (!output) {
        throw MotionProfileError("Unable to save joint feedback. Stopping the motion test.");
      }
      next += 10ms;
      timing.sleep_until(next);
    }
    if (!stop() && std::abs(final_position - target.front()) > 0.02F) {
      throw MotionProfileError(
          "The shoulder-pan target was not reached within three seconds. "
          "Check the feedback, calibrated limits, and motion settings before comparing profiles.");
    }
  }
}

}  // namespace isaac::os
