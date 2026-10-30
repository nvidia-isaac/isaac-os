/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <CLI/CLI.hpp>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <cuda_runtime.h>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/so101_driver/so101_driver_config.h"
#include "operators/so101/so101_delta_motion_control_op.h"
#include "operators/so101/so101_io_ops.h"
#include "operators/so101_reach/so101_reach_fused_op.h"

namespace isaac::os {
namespace {

struct Options {
  std::filesystem::path config_path{
      "applications/so101_reach/config/"
      "so101_reach_config.yaml"};
  std::optional<std::size_t> message_count;
};

void parse_options(CLI::App& app, Options& options, int argc, char** argv) {
  app.add_option("--config-path", options.config_path, "Application YAML configuration")
      ->capture_default_str();
  app.add_option("--message-count", options.message_count, "Stop after this many commands")
      ->check(CLI::PositiveNumber);
  app.parse(argc, argv);
}

void log_driver(so101::DriverLogLevel level, std::string_view message) {
  switch (level) {
    case so101::DriverLogLevel::kInfo:
      HOLOSCAN_LOG_INFO("{}", message);
      break;
    case so101::DriverLogLevel::kWarning:
      HOLOSCAN_LOG_WARN("{}", message);
      break;
    case so101::DriverLogLevel::kError:
      HOLOSCAN_LOG_ERROR("{}", message);
      break;
  }
}

int run(const Options& options) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA device is available for the SO-101 reach application.\n";
    return 1;
  }

  auto document =
      holoscan::config_io::load(options.config_path, holoscan::config_io::Format::kYaml);
  auto state_params =
      holoscan::params::decode<ops::So101StateSourceParams>(document, "/state_source");
  auto policy_params = holoscan::params::decode<ops::So101ReachParams>(document, "/policy");
  auto control_params =
      holoscan::params::decode<ops::So101DeltaMotionParams>(document, "/motion_control");
  const so101::FeetechDriverOptions driver_options = decode_so101_driver_options(document);
  const auto driver = std::make_shared<so101::FeetechSerialDriver>();

  holoscan::Graph graph{"so101-reach"};
  auto source = graph.op<ops::So101StateSourceOp>("so101_state_source", driver);
  auto policy = graph.op<ops::So101ReachFusedOp>("so101_reach_policy");
  auto motion_control = graph.op<ops::So101DeltaMotionControlOp>("so101_motion_control");
  const auto sink = graph.op<ops::So101CommandSinkOp>("so101_command_sink", driver,
                                                      options.message_count.value_or(0U));
  source.set_params(std::move(state_params));
  policy.set_params(std::move(policy_params));
  motion_control.set_params(std::move(control_params));

  const holoscan::ConnectionOptions latest{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  graph.add_flow(source->state, policy->state, latest);
  graph.add_flow(source->state, motion_control->state, latest);
  graph.add_flow(policy->delta, motion_control->delta, latest);
  graph.add_flow(motion_control->command, sink->command, latest);
  graph.partition("state_source").add(source);
  graph.partition("reach_policy").add(policy);
  graph.partition("motion_control").add(motion_control);
  graph.partition("command_sink").add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("Unable to compile the SO-101 reach graph.");
    for (const holoscan::Diagnostic& diagnostic : plan.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }
  driver->initialize(driver_options, log_driver);
  try {
    return holoscan::run(plan);
  } catch (...) {
    driver->hold_present_position();
    throw;
  }
}

}  // namespace
}  // namespace isaac::os

int main(int argc, char** argv) {
  // CLI11 keeps references to option values, so they must outlive the parser.
  isaac::os::Options options;
  CLI::App app{"Run the SO-101 reach application on real hardware"};
  try {
    isaac::os::parse_options(app, options, argc, argv);
    auto signals = holoscan::ManagedShutdownSignals::create();
    if (!signals) {
      std::cerr << "Unable to prepare managed shutdown before starting the SO-101 arm: "
                << holoscan::runtime_error_message(signals.error()) << ".\n";
      return 1;
    }
    return isaac::os::run(options);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
