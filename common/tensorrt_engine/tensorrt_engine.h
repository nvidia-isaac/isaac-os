/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>

#include <NvInfer.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace isaac {

/// Owns a TensorRT engine, runtime, and execution context.
///
/// Use the static factory TensorRTEngine::create() — the private constructor is only accessible
/// from there. Calling initLibNvInferPlugins() is handled internally (idempotent).
///
/// Typical usage:
///   auto result = TensorRTEngine::create(log_prefix, onnx_path);
///   if (!result) throw std::runtime_error(result.error());
///   auto trt = std::move(*result);
///   trt->context().setTensorAddress(name, ptr);
///   trt->context().enqueueV3(stream);
class TensorRTEngine {
 public:
  /// Load a cached engine or build from ONNX and cache it alongside the ONNX file.
  /// @param log_prefix  Prefix prepended to all TRT log messages.
  /// @param onnx_path   Path to the ONNX model; engine cached next to it keyed on GPU+version.
  /// @return            The engine on success, or an error message string on failure.
  static holoscan::expected<std::unique_ptr<TensorRTEngine>, holoscan::RuntimeError> create(
      const std::string& log_prefix, const std::string& onnx_path);

  ~TensorRTEngine() = default;

  TensorRTEngine(const TensorRTEngine&) = delete;
  TensorRTEngine& operator=(const TensorRTEngine&) = delete;

  nvinfer1::ICudaEngine& engine() const;
  nvinfer1::IExecutionContext& context() const;

 private:
  TensorRTEngine() = default;

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

}  // namespace isaac
