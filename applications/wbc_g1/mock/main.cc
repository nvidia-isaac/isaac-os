/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <CLI/CLI.hpp>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <utility>

#include <cuda_runtime.h>
#include <holoscan/config/document.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "operators/agile_wbc/agile_wbc_op.h"
#include "operators/agile_wbc/history_term_config_codec.h"
#include "operators/mock/mock_joint_state_op.h"
#include "operators/mock/mock_sink_op.h"

namespace isaac::os {
namespace {

struct Options {
  std::filesystem::path config_path;
  std::optional<std::size_t> message_count;
};

void parse_options(CLI::App& app, Options& options, int argc, char** argv) {
  app.add_option("--config-path", options.config_path, "Application YAML configuration")
      ->required();
  app.add_option("--message-count", options.message_count, "Stop after this many messages")
      ->check(CLI::PositiveNumber);
  app.parse(argc, argv);
}

bool has_cuda_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

int run(const Options& options) {
  if (!has_cuda_device()) {
    std::cerr << "No CUDA device is available; skipping the WBC application.\n";
    return ISAAC_CTEST_SKIP_RETURN_CODE;
  }

  // Parse operator parameters.
  auto application_config =
      holoscan::config_io::load(options.config_path, holoscan::config_io::Format::kYaml);
  auto wbc_input = holoscan::params::decode<ops::WbcParams>(application_config, "/gr00t_wbc");
  const std::size_t joint_count = wbc_input->sdk_joint_order.size();

  // Instantiate operators.
  holoscan::Graph graph{"mock-wbc-g1"};
  const auto source = graph.op<ops::MockJointStateOp>("mock_joint_state_source", joint_count,
                                                      options.message_count);
  auto wbc = graph.op<ops::AgileWbcOp>("agile_wbc");
  const auto sink = graph.op<ops::MockSinkOp>("mock_joint_command_sink", joint_count);

  // Set operator parameters.
  wbc.set_params(std::move(wbc_input));

  // Define graph connections.
  graph.add_flow(source->joint_state, wbc->joint_state);
  graph.add_flow(source->imu, wbc->imu);
  graph.add_flow(wbc->command, sink->command);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // Compile and run.
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << "Unable to compile the WBC graph.\n" << plan.json() << '\n';
    return 1;
  }
  if (!options.message_count.has_value()) {
    return holoscan::run(plan);
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  session.wait();

  if (session.termination() != holoscan::RunTermination::kCompleted) {
    HOLOSCAN_LOG_ERROR("The WBC graph did not complete successfully.");
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
  CLI::App app{"Run the WBC mock application"};
  try {
    isaac::os::parse_options(app, options, argc, argv);
    if (!options.message_count.has_value()) {
      // Establish signal handling before CUDA and graph setup can start worker threads.
      auto signals = holoscan::ManagedShutdownSignals::create();
      if (!signals) {
        std::cerr << "Unable to initialize shutdown signal handling.\n";
        return 1;
      }
      const int result = isaac::os::run(options);
      if (!signals->close()) {
        std::cerr << "Unable to restore shutdown signal handling.\n";
        return 1;
      }
      return result;
    }
    return isaac::os::run(options);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
