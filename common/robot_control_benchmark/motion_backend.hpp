/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "common/robot_control_benchmark/arm_kinematics.hpp"
#include "common/robot_control_benchmark/payloads.hpp"

struct CUstream_st;

namespace isaac::os::benchmark {

/// Non-owning CUDA stream handle without requiring CUDA headers in CPU-only builds.
using CudaStreamHandle = CUstream_st*;

/// Joint state and Cartesian target consumed by one motion computation.
struct MotionInput {
  std::array<double, kNumJoints> q_warm_start{};
  std::array<double, kNumJoints> measured_q{};
  double target_p[3]{};
  std::array<double, 9> target_R{};
};

/// Interface implemented by each fixed robot-control motion workload.
class MotionBackend {
 public:
  /// Destroys the selected motion backend.
  virtual ~MotionBackend() = default;
  /// Returns the stable backend name used in labels and configuration.
  [[nodiscard]] virtual const char* name() const = 0;
  /// Updates the warm start in place and optionally uses a caller-owned CUDA stream.
  virtual void compute(MotionInput& in, CudaStreamHandle solve_stream = nullptr) = 0;
};

// The CPU arm of the benchmark, and the one the cross-runtime comparison is built on.
//
// One four-iteration damped-least-squares solve per tick. The GPU backend uses the
// same ArmModelT definition as float for each of its 1,024 local seeds.
class CpuDlsIkBackend final : public MotionBackend {
 public:
  /// Returns the CPU IK backend name.
  [[nodiscard]] const char* name() const override { return "cpu-ik"; }

  /// Runs one double-precision DLS IK step on the CPU.
  void compute(MotionInput& in, CudaStreamHandle solve_stream) override {
    static_cast<void>(solve_stream);
    model_.ik_step(in.q_warm_start.data(), in.target_p, in.target_R.data());
  }

 private:
  [[no_unique_address]] ArmModel model_;
};

/// Creates the named fixed motion workload.
std::unique_ptr<MotionBackend> create_motion_backend(std::string_view backend);

}  // namespace isaac::os::benchmark
