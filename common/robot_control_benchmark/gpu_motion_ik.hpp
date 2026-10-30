/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// GPU inverse kinematics for the Motion stage: 1,024 parallel single-precision damped
// least-squares solves and GPU-side command selection per control tick.
//
// This is the benchmark's GPU workload. It consumes the commanded target and produces a
// real IK solution, which is what makes it a demonstration of useful work inside a
// 1 kHz frame rather than a synthetic placeholder.

#include <memory>

#include "common/robot_control_benchmark/motion_backend.hpp"

namespace isaac::os::benchmark {

/// Motion backend that searches 1,024 local IK seeds on CUDA per tick.
class GpuDlsIkBackend final : public MotionBackend {
 public:
  /// Creates and warms the fixed GPU IK solver.
  GpuDlsIkBackend();

  /// Releases the CUDA resources owned by this backend.
  ~GpuDlsIkBackend() override;

  GpuDlsIkBackend(const GpuDlsIkBackend&) = delete;
  GpuDlsIkBackend& operator=(const GpuDlsIkBackend&) = delete;
  GpuDlsIkBackend(GpuDlsIkBackend&&) = delete;
  GpuDlsIkBackend& operator=(GpuDlsIkBackend&&) = delete;

  /// Returns the GPU IK backend name.
  [[nodiscard]] const char* name() const override { return "gpu-ik"; }

  /// Stages inputs in pinned memory, replays the GPU IK graph, and waits for its command.
  /// The first call on a new stream captures the graph; use warmup calls before measuring.
  /// A non-null stream is borrowed for this call only.
  void compute(MotionInput& in, CudaStreamHandle solve_stream = nullptr) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace isaac::os::benchmark
