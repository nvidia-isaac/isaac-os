/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cublas_v2.h>

#include <chrono>
#include <cstddef>

#include <cuda_bf16.h>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace isaac::os::ops {

/// Generates the benchmark's fixed GPU contention workload on a CUDA stream.
///
/// Each activation repeats 4096 x 4096 BF16 matrix multiplication with FP32 accumulation,
/// reusing fixed pseudorandom inputs and overwriting one output buffer.
/// A coalescing 10 us clock keeps bursts ready without catching up missed ticks.
/// Stream completion bounds GPU work to one burst in flight.
/// Contended and isolated scenarios differ only in their CUDA resource placement.
class GpuContentionOp final : public holoscan::Operator<> {
 public:
  /// Fixed workload definition shared by every contention scenario.
  static constexpr auto kPeriod = std::chrono::microseconds{10};
  static constexpr int kMatrixOrder = 4096;
  static constexpr int kGemmCount = 42;

  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

 private:
  static constexpr std::size_t kMatrixElements =
      static_cast<std::size_t>(kMatrixOrder) * static_cast<std::size_t>(kMatrixOrder);
  static constexpr std::size_t kMatrixBytes = kMatrixElements * sizeof(__nv_bfloat16);

  cublasHandle_t handle_{};
  __nv_bfloat16* left_{};
  __nv_bfloat16* right_{};
  __nv_bfloat16* product_{};
};

}  // namespace isaac::os::ops
