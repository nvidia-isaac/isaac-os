/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "common/robot_control_benchmark/arm_kinematics.hpp"
#include "common/robot_control_benchmark/joint_safety.hpp"

namespace isaac::os::benchmark::gpu_ik {

/// Joint count accepted by the fixed GPU IK workload.
inline constexpr int kJoints = 7;

/// Number of independent initial joint configurations evaluated on every control tick.
inline constexpr int kSeeds = 1024;

/// Fixed local search for a servo tick, not a global IK or collision-free planner.
inline constexpr int kIterations = 4;
inline constexpr float kSeedRadiusRad = 0.08F;
inline constexpr float kOrientationWeight = 0.01F;
inline constexpr float kContinuityWeight = 0.0001F;

/// One target and the feasible command interval shared by all GPU candidates.
struct SolveInput {
  float measured_q[kJoints];  // NOSONAR: CUDA device access; std::array methods are host-only.
  float target_p[3];          // NOSONAR: CUDA device access; std::array methods are host-only.
  float target_R[9];          // NOSONAR: CUDA device access; std::array methods are host-only.
  float lower[kJoints];       // NOSONAR: CUDA device access; std::array methods are host-only.
  float upper[kJoints];       // NOSONAR: CUDA device access; std::array methods are host-only.
};

/// Packs the host input and intersects joint limits with the allowed step from measured joints.
inline SolveInput make_solve_input(const double* measured_q, const double* target_p,
                                   const double* target_R) {
  SolveInput input{};
  for (int j = 0; j < kJoints; ++j) {
    input.measured_q[j] = static_cast<float>(measured_q[j]);
    const double low =
        std::max(JointSafety::kMinPositionRad[j], measured_q[j] - JointSafety::kMaxStepRad);
    const double high =
        std::min(JointSafety::kMaxPositionRad[j], measured_q[j] + JointSafety::kMaxStepRad);
    if (low > high) {
      // Match the safety stage's bounded recovery when measured joints are out of range.
      const double recovery = measured_q[j] < JointSafety::kMinPositionRad[j]
                                  ? measured_q[j] + JointSafety::kMaxStepRad
                                  : measured_q[j] - JointSafety::kMaxStepRad;
      input.lower[j] = input.upper[j] = static_cast<float>(recovery);
    } else {
      input.lower[j] = static_cast<float>(low);
      input.upper[j] = static_cast<float>(high);
      // Round inward so float-to-double output conversion does not trigger safety clamps.
      if (input.lower[j] < low) {
        input.lower[j] = std::nextafter(input.lower[j], INFINITY);
      }
      if (input.upper[j] > high) {
        input.upper[j] = std::nextafter(input.upper[j], -INFINITY);
      }
    }
  }
  for (int i = 0; i < 3; ++i) {
    input.target_p[i] = static_cast<float>(target_p[i]);
  }
  for (int i = 0; i < 9; ++i) {
    input.target_R[i] = static_cast<float>(target_R[i]);
  }
  return input;
}

/// Reproducible perturbations without mutable RNG state or per-tick host seed transfers.
RCB_HD inline float seed_offset(int seed, int joint) {
  if (seed == 0) {
    return 0.0F;
  }
  auto bits = static_cast<std::uint32_t>(seed * kJoints + joint);
  bits ^= bits >> 16;
  bits *= 0x7feb352dU;
  bits ^= bits >> 15;
  bits *= 0x846ca68bU;
  bits ^= bits >> 16;
  const float unit = static_cast<float>(bits >> 8) * (1.0F / 16777216.0F);
  return kSeedRadiusRad * (2.0F * unit - 1.0F);
}

/// Scores the command that can actually be applied, including a joint-continuity penalty.
RCB_HD inline float command_score(const SolveInput& input, const float* q) {
  float continuity = 0.0F;
  for (int j = 0; j < kJoints; ++j) {
    if (!(q[j] >= input.lower[j] && q[j] <= input.upper[j])) {
      return INFINITY;
    }
    const float delta = q[j] - input.measured_q[j];
    continuity += delta * delta;
  }
  const ArmModelT<float> model;
  Mat4T<float> ee{};
  Mat4T<float> frames[kJoints + 1]{};  // NOSONAR: fixed local storage in host/device code.
  model.forward(q, ee, frames);
  float position = 0.0F;
  float rotation = 0.0F;
  for (int row = 0; row < 3; ++row) {
    const float delta = ee[row * 4 + 3] - input.target_p[row];
    position += delta * delta;
    for (int col = 0; col < 3; ++col) {
      const float dr = ee[row * 4 + col] - input.target_R[row * 3 + col];
      rotation += dr * dr;
    }
  }
  // Half the squared Frobenius distance is ~angle^2 near zero and does not vanish at pi.
  const float score =
      position + kOrientationWeight * 0.5F * rotation + kContinuityWeight * continuity;
  return score >= 0.0F ? score : INFINITY;
}

/// Evaluates one of the 1,024 independent, fixed-iteration local IK candidates.
RCB_HD inline float solve_candidate(const SolveInput& input, int seed, float* q) {
  for (int j = 0; j < kJoints; ++j) {
    q[j] = detail::kin_clamp(input.measured_q[j] + seed_offset(seed, j), input.lower[j],
                             input.upper[j]);
  }
  const ArmModelT<float> model;
  model.ik_step(q, input.target_p, input.target_R, kIterations);
  for (int j = 0; j < kJoints; ++j) {
    q[j] = detail::kin_clamp(q[j], input.lower[j], input.upper[j]);
  }
  return command_score(input, q);
}

}  // namespace isaac::os::benchmark::gpu_ik
