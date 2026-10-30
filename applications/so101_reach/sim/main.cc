/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <CLI/CLI.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
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

#include "applications/so101_reach/sim/sim_adapters.h"
#include "operators/so101/so101_delta_motion_control_op.h"
#include "operators/so101_reach/so101_reach_fused_op.h"
#include "operators/ucxx_ops/ucxx_connect_op.h"
#include "operators/ucxx_ops/ucxx_endpoint.h"
#include "operators/ucxx_ops/ucxx_receiver_op.h"
#include "operators/ucxx_ops/ucxx_sender_op.h"

namespace isaac::os {
namespace {

// Default tags of Isaac Sim's UCX joint state and joint command nodes; the scene keeps them.
constexpr std::uint64_t kJointStateTag = 0xe2e1a077;
constexpr std::uint64_t kJointCommandTag = 0x3ad708d3;

struct Options {
  std::filesystem::path config_path{"applications/so101_reach/config/so101_reach_config.yaml"};
  bool compile_only{};
  ops::UcxxConnectOptions connection;
  std::size_t message_count{};
};

int run(const Options& options, holoscan::ManagedShutdownSignals& signals) {
  const auto config =
      holoscan::config_io::load(options.config_path, holoscan::config_io::Format::kYaml);
  auto policy_params = holoscan::params::decode<ops::So101ReachParams>(config, "/policy");
  auto control_params =
      holoscan::params::decode<ops::So101DeltaMotionParams>(config, "/motion_control");
  if (!ops::validate_so101_reach_params(*policy_params) ||
      !ops::validate_so101_delta_motion_params(*control_params)) {
    throw std::invalid_argument("The SO-101 reach policy or motion configuration is invalid.");
  }
  const auto hostname = config.get<std::string>("/sim_so101/hostname");
  const auto port = config.get<int>("/sim_so101/port");

  // Prime CUDA before UCX probes its CUDA transport. Graph-only validation needs no GPU.
  if (!options.compile_only) {
    cudaError_t status = cudaSetDevice(0);
    if (status == cudaSuccess) status = cudaFree(nullptr);
    if (status != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Unable to initialize CUDA for SO-101 simulation: {}.",
                         cudaGetErrorString(status));
      return 1;
    }
  }
  ops::UcxxEndpoint endpoint{hostname, port, false};
  holoscan::Graph graph{"sim-so101-reach"};
  const auto receiver = graph.op<ops::UcxxReceiverOp<isaac::JointStateCpuT>>(
      "joint_state_receiver", endpoint, kJointStateTag, 16384U, 0U, TimestampSource::kGraphClock);
  const auto source = graph.op<ops::So101SimStateOp>("sim_state", options.message_count);
  const auto connect = graph.op<ops::UcxxConnectOp>("connect", endpoint, options.connection);
  auto policy = graph.op<ops::So101ReachFusedOp>("so101_reach_policy");
  auto controller = graph.op<ops::So101DeltaMotionControlOp>("so101_motion_control");
  const auto sender = graph.op<ops::UcxxSenderOp<isaac::JointCommandCpuT>>(
      "joint_command_sender", endpoint, kJointCommandTag);
  policy.set_params(std::move(policy_params));
  controller.set_params(std::move(control_params));
  const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                           .queue_depth = 1U};
  graph.add_flow(receiver->out_message, source->joints, latest);
  graph.add_flow(source->state, policy->state, latest);
  graph.add_flow(source->state, controller->state, latest);
  graph.add_flow(policy->delta, controller->delta, latest);
  graph.add_flow(controller->command, sender->in_message, latest);
  graph.partition("joint_state_rx").add(receiver);
  graph.partition("sim_state").add(source);
  graph.partition("connect").add(connect);
  graph.partition("reach_policy").add(policy);
  graph.partition("motion_control").add(controller);
  graph.partition("joint_command_tx").add(sender);

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto plan = holoscan::compile(graph);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("Unable to compile the SO-101 reach simulation graph.");
    for (const auto& diagnostic : plan.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }
  if (options.compile_only) {
    HOLOSCAN_LOG_INFO(
        "SO-101 reach simulation graph compiled. No simulator connection was opened.");
    return 0;
  }
  HOLOSCAN_LOG_INFO("Running SO-101 reach simulation against {}:{}.", hostname, port);
  auto session = holoscan::run_async(plan, signals);
  session.wait();
  if (session.termination() == holoscan::RunTermination::kRequestedCompleted ||
      session.termination() == holoscan::RunTermination::kStopped)
    return 0;
  HOLOSCAN_LOG_ERROR("SO-101 reach simulation stopped with an error.");
  for (const auto& diagnostic : session.diagnostics()) {
    HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
  }
  return 1;
}

}  // namespace
}  // namespace isaac::os

int main(int argc, char** argv) {
  isaac::os::Options options;
  CLI::App app{"Run the SO-101 reach policy in Isaac Sim over UCX"};
  app.add_option("--config-path", options.config_path, "Application YAML configuration")
      ->take_last();
  app.add_flag("--compile-only", options.compile_only,
               "Compile the graph without connecting or running inference");
  app.add_option_function<std::int64_t>(
         "--connect-timeout-ms",
         [&options](std::int64_t ms) {
           options.connection.timeout = std::chrono::milliseconds{ms};
         },
         "Initial connection timeout in milliseconds (0: wait indefinitely)")
      ->check(CLI::NonNegativeNumber);
  app.add_option_function<std::int64_t>(
         "--connect-retry-interval-ms",
         [&options](std::int64_t ms) {
           options.connection.retry_interval = std::chrono::milliseconds{ms};
         },
         "Delay between connection attempts in milliseconds (default: 1000)")
      ->check(CLI::PositiveNumber);
  app.add_option("--message-count", options.message_count,
                 "Stop after this many received joint states")
      ->check(CLI::PositiveNumber);
  try {
    app.parse(argc, argv);
    auto signals = holoscan::ManagedShutdownSignals::create();
    if (!signals) {
      std::cerr << "Unable to prepare managed shutdown for the SO-101 simulation: "
                << holoscan::runtime_error_message(signals.error()) << ".\n";
      return 1;
    }
    return isaac::os::run(options, *signals);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
