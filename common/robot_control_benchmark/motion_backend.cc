/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/robot_control_benchmark/motion_backend.hpp"

#include "common/robot_control_benchmark/gpu_motion_ik.hpp"

namespace isaac::os::benchmark {

/// Creates the configured fixed motion workload.
std::unique_ptr<MotionBackend> create_motion_backend(std::string_view backend) {
  if (backend == "cpu-ik") {
    return std::make_unique<CpuDlsIkBackend>();
  }
  if (backend == "gpu-ik") {
    return std::make_unique<GpuDlsIkBackend>();
  }
  throw std::invalid_argument("unknown --motion-backend '" + std::string(backend) +
                              "' (expected cpu-ik or gpu-ik)");
}

}  // namespace isaac::os::benchmark
