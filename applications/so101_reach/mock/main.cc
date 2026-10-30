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
#include <utility>

#include <cuda_runtime.h>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "applications/so101_reach/mock/mock_operators.h"
#include "operators/so101/so101_delta_motion_control_op.h"
#include "operators/so101_reach/so101_reach_fused_op.h"

namespace isaac::os {
namespace {

struct Options {
  std::filesystem::path config_path{
      "applications/so101_reach/config/"
      "so101_reach_config.yaml"};
  std::size_t message_count{100U};
  double completion_timeout_seconds{30.0};
};

void parse_options(CLI::App& app, Options& options, int argc, char** argv) {
  app.add_option("--config-path", options.config_path, "Application YAML configuration")
      ->capture_default_str();
  app.add_option("--message-count", options.message_count, "Stop after this many commands")
      ->check(CLI::PositiveNumber)
      ->capture_default_str();
  app.add_option("--completion-timeout-seconds", options.completion_timeout_seconds,
                 "Maximum time to wait for the requested commands")
      ->check(CLI::PositiveNumber)
      ->capture_default_str();
  app.parse(argc, argv);
}

int run(const Options& options) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA device is available for the SO-101 TensorRT mock application.\n";
    return 77;
  }

  auto document =
      holoscan::config_io::load(options.config_path, holoscan::config_io::Format::kYaml);
  auto state_params =
      holoscan::params::decode<ops::So101StateSourceParams>(document, "/state_source");
  auto policy_params = holoscan::params::decode<ops::So101ReachParams>(document, "/policy");
  auto control_params =
      holoscan::params::decode<ops::So101DeltaMotionParams>(document, "/motion_control");
  MockCommandSinkOp::reset(options.message_count);
  holoscan::Graph graph{"mock-so101-reach"};
  auto source = graph.op<MockStateSourceOp>("mock_state_source");
  auto policy = graph.op<ops::So101ReachFusedOp>("so101_reach_policy");
  auto motion_control = graph.op<ops::So101DeltaMotionControlOp>("so101_motion_control");
  const auto sink = graph.op<MockCommandSinkOp>("mock_command_sink");
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
  graph.add_flow(motion_control->command, sink->command);
  graph.partition("state_source").add(source);
  graph.partition("reach_policy").add(policy);
  graph.partition("motion_control").add(motion_control);
  graph.partition("command_sink").add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("Unable to compile the SO-101 TensorRT mock graph.");
    for (const holoscan::Diagnostic& diagnostic : plan.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool completed = MockCommandSinkOp::wait_for_expected_count(
      std::chrono::duration<double>{options.completion_timeout_seconds});
  session.request_stop();
  session.wait();
  if (session.termination() != holoscan::RunTermination::kStopped || !completed ||
      !MockCommandSinkOp::valid()) {
    HOLOSCAN_LOG_ERROR("The SO-101 TensorRT mock graph did not produce valid joint commands.");
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }
  return 0;
}

}  // namespace
}  // namespace isaac::os

int main(int argc, char** argv) {
  // CLI11 keeps references to option values, so they must outlive the parser.
  isaac::os::Options options;
  CLI::App app{"Run the SO-101 reach mock application"};
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
