/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/tensorrt_engine/tensorrt_engine.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <holoscan/holoscan.hpp>

namespace isaac {

namespace {

class TensorRTLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        HOLOSCAN_LOG_ERROR("TensorRT reported an inference engine error: {}", msg);
        break;
      case Severity::kWARNING:
        HOLOSCAN_LOG_WARN("TensorRT reported an inference engine warning: {}", msg);
        break;
      case Severity::kINFO:
        HOLOSCAN_LOG_DEBUG("TensorRT reported inference engine details: {}", msg);
        break;
      default:
        break;
    }
  }
};

// TensorRT registers the first logger used to create a builder, runtime, or refitter for the
// process. Keep this logger alive for the process lifetime and pass it to every TensorRT object.
nvinfer1::ILogger& tensorrt_logger() {
  static auto* const logger = new TensorRTLogger();
  return *logger;
}

// Engine cache filename placed next to the ONNX, keyed on active GPU, TRT version, and ONNX
// file size + mtime so the cache is invalidated when the model is updated in-place.
holoscan::expected<std::string, holoscan::RuntimeError> make_tensorrt_engine_path(
    const std::string& onnx_path) {
  int device{};
  if (cudaGetDevice(&device) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, "Unable to identify the active GPU device."));
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, "Unable to read properties of the active GPU device."));
  }
  std::error_code ec;
  const auto model_size = std::filesystem::file_size(onnx_path, ec);
  if (ec) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, "Unable to determine the ONNX model file size."));
  }
  const auto model_mtime = std::filesystem::last_write_time(onnx_path, ec);
  if (ec) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, "Unable to read the ONNX model modification time."));
  }
  std::string gpu(prop.name);
  gpu.erase(std::remove(gpu.begin(), gpu.end(), ' '), gpu.end());
  return std::filesystem::path(onnx_path).replace_extension("").string() + ".direct." + gpu + "." +
         std::to_string(prop.major) + "." + std::to_string(prop.minor) + ".tensorrt." +
         std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "." +
         std::to_string(NV_TENSORRT_PATCH) + "." + std::to_string(NV_TENSORRT_BUILD) + "." +
         std::to_string(model_size) + "." + std::to_string(model_mtime.time_since_epoch().count()) +
         ".engine";
}

// Builds a TensorRT engine from ONNX, writes it to engine_path, and returns the serialized bytes.
holoscan::expected<std::vector<char>, holoscan::RuntimeError> build_engine_data(
    nvinfer1::ILogger& logger, const std::string& log_prefix, const std::string& onnx_path,
    const std::string& engine_path) {
  HOLOSCAN_LOG_INFO("Building the TensorRT engine for {} from {} (this may take a minute).",
                    log_prefix, onnx_path);
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, log_prefix + ": createInferBuilder failed"));
  }
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
  if (!network) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, log_prefix + ": createNetworkV2 failed"));
  }
  auto parser =
      std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
  if (!parser) {
    return holoscan::make_unexpected(holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                                            log_prefix + ": createParser failed"));
  }
  if (!parser->parseFromFile(onnx_path.c_str(),
                             static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, log_prefix + ": parseFromFile failed for " + onnx_path));
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, log_prefix + ": createBuilderConfig failed"));
  }

  auto* profile = builder->createOptimizationProfile();
  if (!profile) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, log_prefix + ": createOptimizationProfile failed"));
  }
  for (int i = 0; i < network->getNbInputs(); ++i) {
    auto* in = network->getInput(i);
    auto dims = in->getDimensions();
    profile->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
    profile->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
    profile->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
  }
  if (config->addOptimizationProfile(profile) < 0) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, log_prefix + ": addOptimizationProfile failed"));
  }

  auto plan =
      std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, log_prefix + ": buildSerializedNetwork failed"));
  }

  std::ofstream out(engine_path, std::ios::binary);
  if (!out) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, log_prefix + ": cannot write engine cache: " + engine_path));
  }
  out.write(static_cast<const char*>(plan->data()), plan->size());
  if (!out.good()) {
    std::filesystem::remove(engine_path);
    return holoscan::make_unexpected(
        holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                               log_prefix + ": failed writing engine cache: " + engine_path));
  }
  HOLOSCAN_LOG_INFO("Saved the TensorRT engine for {} to {}.", log_prefix, engine_path);

  return std::vector<char>(static_cast<const char*>(plan->data()),
                           static_cast<const char*>(plan->data()) + plan->size());
}

}  // namespace

holoscan::expected<std::unique_ptr<TensorRTEngine>, holoscan::RuntimeError> TensorRTEngine::create(
    const std::string& log_prefix, const std::string& onnx_path) {
  auto self = std::unique_ptr<TensorRTEngine>(new TensorRTEngine());
  auto& logger = tensorrt_logger();

  initLibNvInferPlugins(&logger, "");

  auto engine_path_result = make_tensorrt_engine_path(onnx_path);
  if (!engine_path_result) return holoscan::forward_error(engine_path_result);
  const std::string& engine_path = *engine_path_result;

  std::vector<char> engine_data;
  bool loaded_from_cache = false;

  if (std::filesystem::exists(engine_path)) {
    std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
    if (!f) {
      return holoscan::make_unexpected(
          holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                 log_prefix + ": cannot open cached engine: " + engine_path));
    }
    const auto sz = f.tellg();
    if (sz <= std::streampos{0}) {
      return holoscan::make_unexpected(
          holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                 log_prefix + ": cached engine has invalid size: " + engine_path));
    }
    f.seekg(0, std::ios::beg);
    engine_data.resize(static_cast<size_t>(sz));
    if (!f.read(engine_data.data(), sz)) {
      return holoscan::make_unexpected(
          holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                 log_prefix + ": cached engine read failed: " + engine_path));
    }
    HOLOSCAN_LOG_INFO("Loaded the cached TensorRT engine for {}: {}", log_prefix, engine_path);
    loaded_from_cache = true;
  } else {
    auto result = build_engine_data(logger, log_prefix, onnx_path, engine_path);
    if (!result) return holoscan::forward_error(result);
    engine_data = std::move(*result);
  }

  self->runtime_.reset(nvinfer1::createInferRuntime(logger));
  if (!self->runtime_) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, log_prefix + ": createInferRuntime failed"));
  }
  self->engine_.reset(
      self->runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!self->engine_ && loaded_from_cache) {
    HOLOSCAN_LOG_WARN("The cached TensorRT engine for {} is invalid; rebuilding: {}", log_prefix,
                      engine_path);
    std::filesystem::remove(engine_path);
    auto result = build_engine_data(logger, log_prefix, onnx_path, engine_path);
    if (!result) return holoscan::forward_error(result);
    engine_data = std::move(*result);
    self->engine_.reset(
        self->runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  }
  if (!self->engine_) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, log_prefix + ": deserializeCudaEngine failed"));
  }
  self->context_.reset(self->engine_->createExecutionContext());
  if (!self->context_) {
    return holoscan::make_unexpected(holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure, log_prefix + ": createExecutionContext failed"));
  }

  return self;
}

nvinfer1::ICudaEngine& TensorRTEngine::engine() const { return *engine_; }
nvinfer1::IExecutionContext& TensorRTEngine::context() const { return *context_; }

}  // namespace isaac
