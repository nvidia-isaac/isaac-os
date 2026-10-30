/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <CLI/CLI.hpp>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <sipl_capture_op/sipl_capture_op.hpp>
#include <sipl_capture_op/sipl_capture_service.hpp>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "applications/sipl_camera/sipl_metadata_sink_op.h"

namespace {

namespace camera = holoscan::camera;

using isaac::os::ops::SiplMetadataSinkOp;

constexpr std::uint32_t kCaptureQueueDepth = 4U;
constexpr std::uint32_t kCaptureTimeoutUs = 1'000'000U;
constexpr std::uint32_t kMaxHeight = 1984U;
constexpr std::uint32_t kMaxWidth = 2560U;
constexpr std::string_view kNitoBasePath = "/var/nvidia/nvcam/settings/sipl";
constexpr bool kRawOutput = false;

struct Options {
  std::string camera_config{"VB1940"};
  std::filesystem::path json_config;
  std::uint32_t camera_index{};
  std::int32_t cuda_device{};
  bool compile_only{};
  bool stereo{};
};

int run(const Options& options) {
  auto service = std::make_shared<camera::SIPLCaptureService>(
      options.camera_config, options.json_config.string(), kRawOutput, kCaptureQueueDepth,
      std::string{kNitoBasePath}, kCaptureTimeoutUs, options.cuda_device);

  holoscan::Graph graph{"sipl-camera"};
  const std::uint32_t camera_count = options.stereo ? 2U : 1U;

  holoscan::visualization::LayerStack layers;
  const float share = 1.0F / static_cast<float>(camera_count);
  for (std::uint32_t index = 0; index < camera_count; ++index) {
    layers.add_image_layer(
        std::format("camera_{}", index),
        {
            .format = holoscan::visualization::ImageFormat::kNV12,
            .max_width = kMaxWidth,
            .max_height = kMaxHeight,
            .yuv_model = holoscan::visualization::YUVModelConversion::kBT709,
            .yuv_range = holoscan::visualization::YUVRange::kFull,
            .view =
                holoscan::visualization::ViewTransform{
                    .offset_x = static_cast<float>(index) * share, .scale_x = share},
        });
  }
  const holoscan::visualization::SinkDesc window{holoscan::visualization::WindowSinkDesc{
      .title = "SIPL Camera",
      .width = 1280U,
      .height = options.stereo ? 496U : 992U,
  }};
  constexpr auto display_period = std::chrono::milliseconds{33};
  constexpr auto max_frame_age = std::chrono::milliseconds{100};
  auto display = graph.op<holoscan::visualization::HolovizOp>(
      "holoviz", std::move(layers), window,
      holoscan::visualization::ClockTrigger{.period = display_period}, max_frame_age);

  const holoscan::ConnectionOptions latest{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  holoscan::CompileOptions compile_options;
  const auto bind_output = [&](std::string operator_path, std::string output_port) {
    compile_options.deployment.bind_tensor_output_device({
        .operator_path = std::move(operator_path),
        .output_port = std::move(output_port),
        .device = holoscan::DeviceId{.value = options.cuda_device},
    });
  };
  for (std::uint32_t index = 0; index < camera_count; ++index) {
    const std::string capture_name = std::format("capture_{}", index);
    const std::string metadata_sink_name = std::format("metadata_sink_{}", index);
    auto capture = graph.op<camera::SIPLCaptureOp>(capture_name, service,
                                                   options.stereo ? index : options.camera_index,
                                                   holoscan::MemoryKind::kCudaDevice);
    auto metadata_sink = graph.op<SiplMetadataSinkOp>(metadata_sink_name);

    graph.add_flow(capture->frame, display->image_layer_input(index), latest);
    graph.add_flow(capture->sensor_data, metadata_sink->input, latest);
    bind_output(capture_name, "frame");
  }
  graph.partition("pipeline").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("Unable to compile the SIPL camera graph.");
    for (const holoscan::Diagnostic& diagnostic : plan.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }
  if (options.compile_only) return 0;

  auto shutdown = holoscan::ManagedShutdownSignals::create();
  if (!shutdown) {
    HOLOSCAN_LOG_ERROR("Unable to install signal handling for clean camera shutdown.");
    return 1;
  }
  auto session = holoscan::run_async(plan, *shutdown);
  session.wait();
  if (session.termination() == holoscan::RunTermination::kFailed) {
    HOLOSCAN_LOG_ERROR("The SIPL camera graph failed.");
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App cli{"Display a SIPL camera through Holoviz"};
  Options options;
  cli.add_option("--camera-config", options.camera_config,
                 "SIPL camera configuration name for a 2560x1984 NV12 stream");
  auto* camera_index =
      cli.add_option("--camera-index", options.camera_index, "Camera index within the SIPL rig");
  cli.add_flag("--stereo", options.stereo, "Display cameras 0 and 1 side by side")
      ->excludes(camera_index);
  cli.add_flag("--compile-only", options.compile_only,
               "Compile the graph without opening a camera");
  cli.add_option("--cuda-device", options.cuda_device, "CUDA device ordinal")
      ->check(CLI::NonNegativeNumber);
  cli.add_option("--json-config", options.json_config,
                 "SIPL platform JSON configuration for a 2560x1984 NV12 stream")
      ->required()
      ->check(CLI::ExistingFile);
  try {
    cli.parse(argc, argv);
    return run(options);
  } catch (const CLI::ParseError& error) {
    return cli.exit(error);
  } catch (const std::exception& error) {
    HOLOSCAN_LOG_ERROR("Unable to run the SIPL camera sample: {}", error.what());
    return 1;
  }
}
