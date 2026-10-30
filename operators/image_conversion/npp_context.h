/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_IMAGE_CONVERSION_NPP_CONTEXT_H
#define ISAAC_OS_OPERATORS_IMAGE_CONVERSION_NPP_CONTEXT_H

#include <nppdefs.h>

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <holoscan/core/execution_context.hpp>

namespace isaac::os::ops::detail {

class NppContext {
 public:
  holoscan::expected<NppStreamContext, holoscan::Error> for_stream(
      holoscan::ExecutionContext& context) {
    const int device = context.cuda_stream().device().get();
    if (!context_) {
      NppStreamContext value{};
      value.nCudaDeviceId = device;
      int shared_memory_per_block{};
      const std::array<std::pair<cudaDeviceAttr, int*>, 5> attributes{{
          {cudaDevAttrMultiProcessorCount, &value.nMultiProcessorCount},
          {cudaDevAttrMaxThreadsPerMultiProcessor, &value.nMaxThreadsPerMultiProcessor},
          {cudaDevAttrMaxThreadsPerBlock, &value.nMaxThreadsPerBlock},
          {cudaDevAttrComputeCapabilityMajor, &value.nCudaDevAttrComputeCapabilityMajor},
          {cudaDevAttrComputeCapabilityMinor, &value.nCudaDevAttrComputeCapabilityMinor},
      }};
      cudaError_t status = cudaSuccess;
      for (const auto& [attribute, output] : attributes) {
        status = cudaDeviceGetAttribute(output, attribute, device);
        if (status != cudaSuccess) break;
      }
      if (status == cudaSuccess) {
        status = cudaDeviceGetAttribute(&shared_memory_per_block,
                                        cudaDevAttrMaxSharedMemoryPerBlock, device);
      }
      if (status != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure, "Unable to read GPU properties for image conversion: " +
                                               std::string{cudaGetErrorString(status)}});
      }
      value.nSharedMemPerBlock = static_cast<std::size_t>(shared_memory_per_block);
      context_ = value;
    }
    if (context_->nCudaDeviceId != device) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kIncompatible,
                          "Image conversion received work on a different CUDA device."});
    }
    NppStreamContext value = *context_;
    value.hStream = context.cuda_stream().get();
    if (const cudaError_t status = cudaStreamGetFlags(value.hStream, &value.nStreamFlags);
        status != cudaSuccess) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure,
                          "Unable to prepare the CUDA stream for image conversion: " +
                              std::string{cudaGetErrorString(status)}});
    }
    return value;
  }

 private:
  std::optional<NppStreamContext> context_;
};

}  // namespace isaac::os::ops::detail

#endif  // ISAAC_OS_OPERATORS_IMAGE_CONVERSION_NPP_CONTEXT_H
