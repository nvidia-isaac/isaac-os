/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Padding-free POD tick messages. Sizes are asserted so the Holoscan 5 POD
// layout validator can cover every object byte.

#include <cstdint>

namespace isaac::os::benchmark {

inline constexpr int kNumJoints = 7;
inline constexpr int kPacketBytes = 8;
inline constexpr int kBatchBytes = kNumJoints * kPacketBytes;

struct Pose6 {
  std::int64_t seq;
  std::int64_t t_src_ns;
  double x, y, z;
  double rx, ry, rz;
  std::uint64_t version;
  std::int64_t produced_ns;
};

struct JointStateSample {
  std::int64_t sequence;
  std::int64_t scheduled_ns;
  std::int64_t callback_start_ns;
  double position[kNumJoints];
};

struct JointSetpoints {
  std::int64_t seq;
  std::int64_t t_src_ns;
  double q[kNumJoints];
  std::int64_t callback_start_ns;
  double measured_q[kNumJoints];
  std::int64_t target_produced_ns;
  std::int64_t target_use_ns;
  std::uint64_t target_version;
};

struct MotorCommandBatch {
  std::int64_t seq;
  std::int64_t t_src_ns;
  std::uint8_t packets[kBatchBytes];
  std::int64_t callback_start_ns;
  std::int64_t target_produced_ns;
  std::int64_t target_use_ns;
  std::uint64_t target_version;
  double safe_q[kNumJoints];
  std::uint32_t safety_non_finite;
  std::uint32_t safety_range_clamps;
  std::uint32_t safety_step_clamps;
  std::uint32_t safety_recovery_steps;
};

static_assert(sizeof(Pose6) == 80, "Pose6 must be padding-free for the wire POD layout");
static_assert(sizeof(JointStateSample) == 80, "JointStateSample must be padding-free");
static_assert(sizeof(JointSetpoints) == 160, "JointSetpoints must be padding-free");
static_assert(sizeof(MotorCommandBatch) == 176, "MotorCommandBatch must be padding-free");

}  // namespace isaac::os::benchmark
