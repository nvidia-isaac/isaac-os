/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "applications/robot_control_benchmark/gpu_contention_op.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {
namespace {

[[nodiscard]] std::string cuda_error(std::string_view action, cudaError_t status) {
  return std::format("{}: {}.", action, cudaGetErrorString(status));
}

[[nodiscard]] std::string cublas_error(std::string_view action, cublasStatus_t status) {
  return std::format("{} (cuBLAS status {}).", action, static_cast<int>(status));
}

void require_cuda(cudaError_t status, std::string_view action,
                  holoscan::ErrorCode code = holoscan::ErrorCode::kFailure) {
  if (status != cudaSuccess) {
    throw holoscan::RuntimeError(code, cuda_error(action, status));
  }
}

void require_cublas(cublasStatus_t status, std::string_view action) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure, cublas_error(action, status));
  }
}

void release_buffer(__nv_bfloat16*& buffer, std::string_view name) noexcept {
  if (buffer == nullptr) {
    return;
  }
  const cudaError_t status = cudaFree(buffer);
  if (status != cudaSuccess) {
    HOLOSCAN_LOG_WARN("Unable to release the GPU contention {} buffer: {}.", name,
                      cudaGetErrorString(status));
  }
  buffer = nullptr;
}

}  // namespace

holoscan::Contract GpuContentionOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnClock{.period = kPeriod,
                                     .tick_policy = holoscan::ClockTickPolicy::kCoalesceLateTicks});
  return contract;
}

void GpuContentionOp::start() {
  try {
    require_cuda(cudaMalloc(&left_, kMatrixBytes),
                 "Unable to allocate the first GPU contention input",
                 holoscan::ErrorCode::kResourceExhausted);
    require_cuda(cudaMalloc(&right_, kMatrixBytes),
                 "Unable to allocate the second GPU contention input",
                 holoscan::ErrorCode::kResourceExhausted);
    require_cuda(cudaMalloc(&product_, kMatrixBytes),
                 "Unable to allocate the GPU contention output",
                 holoscan::ErrorCode::kResourceExhausted);
    // Initialize reproducible, zero-centered inputs once. Scale the weights to keep output
    // magnitudes bounded; every GEMM reuses these inputs and overwrites the output.
    std::mt19937 generator{0};
    std::uniform_real_distribution<float> values{-1.0F, 1.0F};
    std::vector<__nv_bfloat16> initial(kMatrixElements);
    std::ranges::generate(initial, [&] { return __float2bfloat16(values(generator)); });
    require_cuda(cudaMemcpy(left_, initial.data(), kMatrixBytes, cudaMemcpyHostToDevice),
                 "Unable to initialize the first GPU contention input");
    const float weight_scale = 1.0F / std::sqrt(static_cast<float>(kMatrixOrder));
    std::ranges::generate(initial,
                          [&] { return __float2bfloat16(values(generator) * weight_scale); });
    require_cuda(cudaMemcpy(right_, initial.data(), kMatrixBytes, cudaMemcpyHostToDevice),
                 "Unable to initialize the second GPU contention input");
    require_cuda(cudaMemset(product_, 0, kMatrixBytes),
                 "Unable to initialize the GPU contention output");
    require_cuda(cudaDeviceSynchronize(), "Unable to complete GPU contention initialization");
    require_cublas(cublasCreate(&handle_), "Unable to create the GPU contention workload");
    HOLOSCAN_LOG_INFO(
        "GPU contention runs {} repeated {} x {} BF16 matrix multiplications with FP32 "
        "accumulation per burst using fixed pseudorandom inputs, with at most one burst in flight.",
        kGemmCount, kMatrixOrder, kMatrixOrder);
  } catch (...) {
    stop();
    throw;
  }
}

void GpuContentionOp::stop() {
  if (handle_ != nullptr) {
    const cublasStatus_t status = cublasDestroy(handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
      HOLOSCAN_LOG_WARN("Unable to release the GPU contention workload (cuBLAS status {}).",
                        static_cast<int>(status));
    }
    handle_ = nullptr;
  }
  release_buffer(product_, "output");
  release_buffer(right_, "second input");
  release_buffer(left_, "first input");
}

holoscan::expected<void, holoscan::Error> GpuContentionOp::compute(
    holoscan::ExecutionContext& context) {
  if (handle_ == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kNotReady, "The GPU contention workload is not initialized."});
  }

  const cudaStream_t stream = context.cuda_stream().get();
  cublasStatus_t cublas_status = cublasSetStream(handle_, stream);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        cublas_error("Unable to assign the GPU contention stream", cublas_status)});
  }

  constexpr float kAlpha = 1.0F;
  constexpr float kBeta = 0.0F;
  for (int gemm = 0; gemm < kGemmCount; ++gemm) {
    cublas_status = cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, kMatrixOrder, kMatrixOrder,
                                 kMatrixOrder, &kAlpha, left_, CUDA_R_16BF, kMatrixOrder, right_,
                                 CUDA_R_16BF, kMatrixOrder, &kBeta, product_, CUDA_R_16BF,
                                 kMatrixOrder, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          cublas_error("Unable to launch the GPU contention workload", cublas_status)});
    }
  }

  const cudaError_t cuda_status = cudaStreamSynchronize(stream);
  if (cuda_status != cudaSuccess) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        cuda_error("Unable to complete the GPU contention workload", cuda_status)});
  }
  return {};
}

}  // namespace isaac::os::ops
