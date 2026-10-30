/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <CLI/CLI.hpp>
#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <utility>

#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "applications/so101_sinus_motion/mock/mock_operators.h"
#include "operators/simple_motion_planner/simple_motion_planner.h"
#include "operators/sinus_motion/sinus_motion_op.h"

namespace isaac::os {
namespace {

/// Command-line settings for a mock run.
struct Options {
  /// Application YAML supplying the state-source, sine, and planner parameters.
  std::filesystem::path config_path{
      "applications/so101_sinus_motion/config/"
      "so101_sinus_motion_config.yaml"};
  /// Number of joint commands to observe before the graph is stopped.
  std::size_t message_count{100U};
  /// Wait budget for those commands. Unset means derive one from the configured state rate.
  std::optional<double> completion_timeout_seconds;
};

/// Binds the supported flags to `options`, throwing `CLI::ParseError` on malformed input.
void parse_options(CLI::App& app, Options& options, int argc, char** argv) {
  app.add_option("--config-path", options.config_path, "Application YAML configuration")
      ->capture_default_str();
  app.add_option("--message-count", options.message_count, "Stop after this many commands")
      ->check(CLI::PositiveNumber)
      ->capture_default_str();
  app.add_option("--completion-timeout-seconds", options.completion_timeout_seconds,
                 "Maximum time to wait for the requested commands. Defaults to three times the "
                 "time the configured state rate needs to produce them")
      ->check(CLI::PositiveNumber);
  app.parse(argc, argv);
}

/// Runs the mock state source, sine, planner, and sink graph until `message_count` commands
/// arrive or the wait budget expires.
///
/// Returns 0 only if the graph stopped cleanly, delivered the requested number of commands, and
/// every command validated; otherwise logs the failure with any graph diagnostics and returns 1.
int run(const Options& options) {
  auto document =
      holoscan::config_io::load(options.config_path, holoscan::config_io::Format::kYaml);
  auto state_params =
      holoscan::params::decode<ops::So101StateSourceParams>(document, "/state_source");
  auto sinus_params =
      holoscan::params::decode<ops::SinusMotionParams>(document, "/sinus_motion_op");
  auto planner_params =
      holoscan::params::decode<ops::SimpleMotionPlannerParams>(document, "/motion_planner_op");
  const double completion_timeout = options.completion_timeout_seconds.value_or(
      10.0 + 3.0 * static_cast<double>(options.message_count) * state_params->period);
  HOLOSCAN_LOG_INFO("Waiting up to {:.1f} s for {} SO101 sinus commands.", completion_timeout,
                    options.message_count);
  MockCommandSinkOp::reset(options.message_count);
  holoscan::Graph graph{"mock-so101-sinus"};
  auto source = graph.op<MockStateSourceOp>("mock_state_source");
  auto sinus = graph.op<ops::SinusMotionOp>("sinus_op");
  auto motion_planner = graph.op<ops::SimpleMotionPlannerOp>("motion_planner");
  const auto sink = graph.op<MockCommandSinkOp>("mock_command_sink");
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
  graph.add_flow(motion_planner->command, sink->command);
  graph.partition("state_source").add(source);
  graph.partition("sinus_op").add(sinus);
  graph.partition("motion_planner").add(motion_planner);
  graph.partition("command_sink").add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("Unable to compile the SO101 sinus mock graph.");
    for (const holoscan::Diagnostic& diagnostic : plan.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool completed =
      MockCommandSinkOp::wait_for_expected_count(std::chrono::duration<double>{completion_timeout});
  session.request_stop();
  session.wait();
  if (session.termination() != holoscan::RunTermination::kStopped || !completed ||
      !MockCommandSinkOp::valid()) {
    HOLOSCAN_LOG_ERROR("The SO101 sinus mock graph did not produce valid joint commands.");
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }
  return 0;
}

}  // namespace
}  // namespace isaac::os

/// Parses the command line, runs the mock graph, and maps parse and runtime failures to an
/// exit status.
int main(int argc, char** argv) {
  isaac::os::Options options;
  CLI::App app{"Run the SO101 sinus motion mock application"};
  try {
    isaac::os::parse_options(app, options, argc, argv);
    return isaac::os::run(options);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
