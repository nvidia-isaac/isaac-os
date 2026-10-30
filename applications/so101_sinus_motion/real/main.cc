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
#include "operators/simple_motion_planner/simple_motion_planner.h"
#include "operators/sinus_motion/sinus_motion_op.h"
#include "operators/so101/so101_io_ops.h"

namespace isaac::os {
namespace {

/// Command-line settings for a hardware run.
struct Options {
  /// Application YAML supplying the `/real_robot` driver settings and the operator parameters.
  std::filesystem::path config_path{
      "applications/so101_sinus_motion/config/"
      "so101_sinus_motion_config.yaml"};
  /// Stop after this many commands. Unset drives the arm until the process is interrupted.
  std::optional<std::size_t> message_count;
};

/// Binds the supported flags to `options`, throwing `CLI::ParseError` on malformed input.
///
void parse_options(CLI::App& app, Options& options, int argc, char** argv) {
  app.add_option("--config-path", options.config_path, "Application YAML configuration")
      ->capture_default_str();
  app.add_option("--message-count", options.message_count, "Stop after this many commands")
      ->check(CLI::PositiveNumber);
  app.parse(argc, argv);
}

/// Forwards a driver diagnostic to the Holoscan logger at the matching severity.
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

/// Drives a real SO-101 through the sine pipeline over the Feetech serial driver.
///
/// The configuration is decoded and validated and the graph is compiled before the driver is
/// initialized, so a bad configuration fails without energizing a servo. If the run throws, the
/// arm is commanded to hold its present position before the exception propagates, leaving it
/// stationary rather than limp or still chasing a stale target.
///
/// Returns 0 on a clean run, or 1 if the configuration cannot be loaded, bound, or compiled.
int run(const Options& options) {
  auto document =
      holoscan::config_io::load(options.config_path, holoscan::config_io::Format::kYaml);
  auto state_params =
      holoscan::params::decode<ops::So101StateSourceParams>(document, "/state_source");
  auto sinus_params =
      holoscan::params::decode<ops::SinusMotionParams>(document, "/sinus_motion_op");
  auto planner_params =
      holoscan::params::decode<ops::SimpleMotionPlannerParams>(document, "/motion_planner_op");
  const so101::FeetechDriverOptions driver_options = decode_so101_driver_options(document);
  const auto driver = std::make_shared<so101::FeetechSerialDriver>();

  holoscan::Graph graph{"so101-sinus"};
  auto source = graph.op<ops::So101StateSourceOp>("so101_state_source", driver);
  auto sinus = graph.op<ops::SinusMotionOp>("sinus_op");
  auto motion_planner = graph.op<ops::SimpleMotionPlannerOp>("motion_planner");
  const auto sink = graph.op<ops::So101CommandSinkOp>("so101_command_sink", driver,
                                                      options.message_count.value_or(0U));
  source.set_params(std::move(state_params));
  sinus.set_params(std::move(sinus_params));
  motion_planner.set_params(std::move(planner_params));

  const holoscan::ConnectionOptions latest{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  graph.add_flow(source->state, sinus->state, latest);
  graph.add_flow(source->state, motion_planner->state, latest);
  graph.add_flow(sinus->command, motion_planner->target, latest);
  graph.add_flow(motion_planner->command, sink->command, latest);
  graph.partition("state_source").add(source);
  graph.partition("sinus_op").add(sinus);
  graph.partition("motion_planner").add(motion_planner);
  graph.partition("command_sink").add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("Unable to compile the SO101 sinus graph.");
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

/// Parses the command line, installs managed shutdown handling so an interrupt can stop the arm
/// cleanly, and then runs the graph.
int main(int argc, char** argv) {
  isaac::os::Options options;
  CLI::App app{"Run the SO101 sinus motion application on real hardware"};
  try {
    isaac::os::parse_options(app, options, argc, argv);
    auto signals = holoscan::ManagedShutdownSignals::create();
    if (!signals) {
      std::cerr << "Unable to prepare managed shutdown before starting the SO101 arm: "
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
