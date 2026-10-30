/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Compare the parallel GPU selection with an exhaustive serial candidate search.
// Host/device multiply-add rounding may change the winning seed, so compare the
// objective and physical command constraints rather than requiring identical joints.

#include "common/robot_control_benchmark/gpu_motion_ik.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "common/robot_control_benchmark/arm_kinematics.hpp"
#include "common/robot_control_benchmark/gpu_ik_workload.hpp"
#include "common/robot_control_benchmark/trajectory.hpp"

namespace isaac::os::benchmark {
namespace {

constexpr int kJoints = gpu_ik::kJoints;

// Covers host/device float contraction while remaining far below the score change
// between the single-seed and full-search solutions exercised below.
constexpr float kScoreTolerance = 2e-6F;

/// Returns whether at least one CUDA device is available to the test process.
bool device_is_present() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// Inputs for one GPU IK solve.
struct Pose {
  std::array<double, kJoints> q;
  std::array<double, 3> target_p;
  std::array<double, 9> target_R;
};

/// Returns a reachable benchmark pose with a nontrivial warm start.
Pose representative_pose() {
  Pose pose{};
  pose.q = ArmModel::home();
  pose.target_p = {0.45, 0.05, 0.50};
  pose.target_R = {1, 0, 0, 0, -1, 0, 0, 0, -1};
  return pose;
}

/// Runs one command through the public motion backend interface.
std::array<double, kJoints> compute_command(GpuDlsIkBackend& backend, const Pose& pose,
                                            CudaStreamHandle stream = nullptr) {
  MotionInput input{};
  input.measured_q = pose.q;
  std::ranges::copy(pose.target_p, input.target_p);
  input.target_R = pose.target_R;
  backend.compute(input, stream);
  return input.q_warm_start;
}

/// Cardinal points of the commanded trajectory, including its X and Y extrema.
constexpr std::array<std::array<double, 3>, 4> kTrajectoryExtrema{{
    {0.30, 0.00, 0.50},
    {0.45, -0.15, 0.50},
    {0.45, 0.15, 0.50},
    {0.60, 0.00, 0.50},
}};

/// Exhaustively finds the best candidate on the host, including the bounded hold fallback.
float host_best_score(const gpu_ik::SolveInput& input) {
  std::array<float, kJoints> q{};
  for (int j = 0; j < kJoints; ++j) {
    q[j] = std::clamp(input.measured_q[j], input.lower[j], input.upper[j]);
  }
  float best = gpu_ik::command_score(input, q.data());
  for (int seed = 0; seed < gpu_ik::kSeeds; ++seed) {
    best = std::min(best, gpu_ik::solve_candidate(input, seed, q.data()));
  }
  return best;
}

/// Computes the selection objective for a downloaded GPU result.
float score_result(const gpu_ik::SolveInput& input, const std::array<double, kJoints>& q) {
  std::array<float, kJoints> q_float{};
  std::ranges::transform(q, q_float.begin(), [](double v) { return static_cast<float>(v); });
  return gpu_ik::command_score(input, q_float.data());
}

/// Checks the returned command is already feasible for the downstream safety stage.
void expect_safe_command(const std::array<double, kJoints>& measured,
                         const std::array<double, kJoints>& command) {
  JointState state{};
  state.position = measured;
  auto checked = command;
  EXPECT_FALSE(JointSafety{}.validate_and_clamp(checked, state).modified());
  for (int j = 0; j < kJoints; ++j) {
    EXPECT_TRUE(std::isfinite(command[j]));
    EXPECT_EQ(checked[j], command[j]);
  }
}

TEST(GpuIkWorkload, SeedsAreDistinctLocalAndDeterministic) {
  std::set<std::array<float, kJoints>> offsets;
  for (int seed = 0; seed < gpu_ik::kSeeds; ++seed) {
    std::array<float, kJoints> offset{};
    for (int j = 0; j < kJoints; ++j) {
      offset[j] = gpu_ik::seed_offset(seed, j);
      EXPECT_LE(std::abs(offset[j]), gpu_ik::kSeedRadiusRad);
      EXPECT_EQ(offset[j], gpu_ik::seed_offset(seed, j));
      if (seed == 0) {
        EXPECT_EQ(offset[j], 0.0F);
      }
    }
    offsets.insert(offset);
  }
  EXPECT_EQ(offsets.size(), 1024);
}

TEST(GpuIkWorkload, OppositeOrientationHasNonzeroCost) {
  const auto q = ArmModelT<float>::home();
  Mat4T<float> ee{};
  Mat4T<float> frames[kJoints + 1]{};  // NOSONAR: forward requires a fixed-size array reference.
  ArmModelT<float>{}.forward(q.data(), ee, frames);
  const std::array<double, 3> p{ee[3], ee[7], ee[11]};
  std::array<double, 9> rotation{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      rotation[r * 3 + c] = ee[r * 4 + c] * (c == 0 ? 1.0 : -1.0);
    }
  }
  std::array<double, kJoints> measured{};
  std::ranges::copy(q, measured.begin());
  const auto input = gpu_ik::make_solve_input(measured.data(), p.data(), rotation.data());
  EXPECT_NEAR(gpu_ik::command_score(input, q.data()), 4 * gpu_ik::kOrientationWeight, 1e-6);
}

/// Returns Cartesian position error after forward kinematics.
double position_error(const std::array<double, kJoints>& q, const std::array<double, 3>& target_p) {
  const ArmModel model;
  Mat4 ee{};
  Mat4 frames[ArmModel::N + 1]{};  // NOSONAR: forward requires a fixed-size array reference.
  model.forward(q.data(), ee, frames);
  const double dx = target_p[0] - ee[3];
  const double dy = target_p[1] - ee[7];
  const double dz = target_p[2] - ee[11];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

TEST(GpuMotionIk, SelectsBestCandidateAcrossTrajectoryExtrema) {
  if (!device_is_present()) {
    GTEST_SKIP() << "No CUDA device is available.";
  }

  GpuDlsIkBackend backend;
  bool improved_over_seed_zero = false;
  for (const auto& target : kTrajectoryExtrema) {
    Pose pose = representative_pose();
    pose.target_p = target;
    const auto input =
        gpu_ik::make_solve_input(pose.q.data(), pose.target_p.data(), pose.target_R.data());
    const float expected = host_best_score(input);
    const auto actual = compute_command(backend, pose);
    const float actual_score = score_result(input, actual);
    EXPECT_NEAR(actual_score, expected, kScoreTolerance) << "target x=" << target[0];
    expect_safe_command(pose.q, actual);

    std::array<float, kJoints> single{};
    const float single_score = gpu_ik::solve_candidate(input, 0, single.data());
    improved_over_seed_zero |= actual_score < single_score - 10 * kScoreTolerance;
  }
  EXPECT_TRUE(improved_over_seed_zero) << "The parallel search must contribute useful candidates.";
}

TEST(GpuMotionIk, RespectsJointLimitsAndStepBounds) {
  if (!device_is_present()) {
    GTEST_SKIP() << "No CUDA device is available.";
  }
  GpuDlsIkBackend backend;
  for (const auto& measured : {JointSafety::kMinPositionRad, JointSafety::kMaxPositionRad}) {
    Pose pose = representative_pose();
    pose.q = measured;
    const auto actual = compute_command(backend, pose);
    expect_safe_command(measured, actual);
  }
}

TEST(GpuMotionIk, HoldsWhenEveryCandidateHasInvalidScore) {
  if (!device_is_present()) {
    GTEST_SKIP() << "No CUDA device is available.";
  }
  Pose pose = representative_pose();
  pose.target_p[0] = std::numeric_limits<double>::quiet_NaN();
  GpuDlsIkBackend backend;
  const auto actual = compute_command(backend, pose);
  for (int j = 0; j < kJoints; ++j) {
    EXPECT_EQ(actual[j], static_cast<double>(static_cast<float>(pose.q[j])));
  }
  expect_safe_command(pose.q, actual);
}

TEST(GpuMotionIk, BackendCentersSearchOnMeasuredState) {
  if (!device_is_present()) {
    GTEST_SKIP() << "No CUDA device is available.";
  }
  const Pose pose = representative_pose();
  MotionInput input{};
  input.measured_q = pose.q;
  input.q_warm_start.fill(2.0);
  std::ranges::copy(pose.target_p, input.target_p);
  input.target_R = pose.target_R;
  GpuDlsIkBackend backend;
  backend.compute(input, nullptr);
  expect_safe_command(pose.q, input.q_warm_start);
}

TEST(GpuMotionIk, TracksTheCircularTrajectoryWithBoundedCommands) {
  if (!device_is_present()) {
    GTEST_SKIP() << "No CUDA device is available.";
  }
  GpuDlsIkBackend backend;
  auto q = ArmModel::home();
  double worst_error = 0.0;
  // One two-second revolution at 1 kHz, after half a second of settling.
  for (int tick = 0; tick < 2500; ++tick) {
    const Pose6 target = trajectory_pose(tick * 0.001);
    const std::array<double, 3> p{target.x, target.y, target.z};
    const auto rotation = rpy_to_rotation(target.rx, target.ry, target.rz);
    const auto next = compute_command(backend, Pose{q, p, rotation});
    expect_safe_command(q, next);
    if (tick >= 500) {
      worst_error = std::max(worst_error, position_error(next, p));
    }
    q = next;
  }
  EXPECT_LT(worst_error, 0.005);
}

TEST(GpuMotionIk, MovesTheFlangeTowardTheTarget) {
  if (!device_is_present()) {
    GTEST_SKIP() << "No CUDA device is available.";
  }

  const Pose pose = representative_pose();
  const double before = position_error(pose.q, pose.target_p);

  GpuDlsIkBackend backend;
  const auto solved = compute_command(backend, pose);

  EXPECT_LT(position_error(solved, pose.target_p), before);
}

TEST(GpuMotionIk, RepeatsItselfExactly) {
  if (!device_is_present()) {
    GTEST_SKIP() << "No CUDA device is available.";
  }

  const Pose pose = representative_pose();
  GpuDlsIkBackend backend;

  const auto first = compute_command(backend, pose);
  const auto second = compute_command(backend, pose);

  for (int j = 0; j < kJoints; ++j) {
    EXPECT_DOUBLE_EQ(first[j], second[j]) << "joint " << j;
  }
}

// The isolated scenario runs each solve on its HSDK-created Green Context stream, which
// the backend borrows for that call without taking ownership.
TEST(GpuMotionIk, RunsOnABorrowedStream) {
  if (!device_is_present()) {
    GTEST_SKIP() << "No CUDA device is available.";
  }

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  const Pose pose = representative_pose();
  {
    GpuDlsIkBackend backend;
    const auto solved = compute_command(backend, pose, stream);
    for (const double value : solved) {
      EXPECT_TRUE(std::isfinite(value));
    }
  }

  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(GpuMotionIk, RefreshesInputsAcrossGraphReplaysAndStreamChanges) {
  if (!device_is_present()) {
    GTEST_SKIP() << "No CUDA device is available.";
  }

  GpuDlsIkBackend backend;
  // Exercise replay with changing inputs and borrowed stream lifetimes. Output checks
  // verify correctness; they do not establish Green Context SM placement.
  for (int round = 0; round < 2; ++round) {
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    Pose pose = representative_pose();
    for (const auto& target : kTrajectoryExtrema) {
      SCOPED_TRACE(::testing::Message()
                   << "round=" << round << " target x=" << target[0] << " y=" << target[1]);
      pose.target_p = target;
      // Vary orientation as well as position and measured joints to cover every input
      // staged into pinned memory before replay.
      pose.target_R = rpy_to_rotation(3.0 + 0.1 * round, target[0], target[1]);
      const auto input =
          gpu_ik::make_solve_input(pose.q.data(), pose.target_p.data(), pose.target_R.data());
      const auto actual = compute_command(backend, pose, stream);
      EXPECT_NEAR(score_result(input, actual), host_best_score(input), kScoreTolerance);
      expect_safe_command(pose.q, actual);
      pose.q = actual;
    }
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  }
  // Check that returning to the owned stream still produces the current result.
  const Pose pose = representative_pose();
  const auto input =
      gpu_ik::make_solve_input(pose.q.data(), pose.target_p.data(), pose.target_R.data());
  const auto actual = compute_command(backend, pose);
  EXPECT_NEAR(score_result(input, actual), host_best_score(input), kScoreTolerance);
  expect_safe_command(pose.q, actual);
}

}  // namespace
}  // namespace isaac::os::benchmark
