/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <CLI/CLI.hpp>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/time/synthetic_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "operators/agile_wbc/agile_wbc_op.h"
#include "operators/agile_wbc/history_term_config_codec.h"
#include "operators/message_drop_sink/message_drop_sink_op.h"
#include "operators/teleop/teleop_params_config_codec.h"
#include "operators/teleop/teleop_twist_command_op.h"
#include "operators/unitree_sim_ucx/unitree_sim_ucx_subgraph.h"

namespace isaac::os {
namespace {

struct Options {
  std::filesystem::path config_path;
  std::string hostname{"127.0.0.1"};
  int port{50008};
  ops::UcxxConnectOptions connection;
  bool compile_only{};
  bool holoviz{};
  bool gamepad{};
};

void parse_options(CLI::App& app, Options& options, int argc, char** argv) {
  app.add_option("--config-path", options.config_path, "Application YAML configuration")
      ->required()
      ->take_last();
  app.add_option("--hostname", options.hostname, "Simulator host");
  app.add_option("--port", options.port, "Simulator UCX port (1-65535)")
      ->check(CLI::Range(1, 65535));
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
  app.add_flag("--compile-only", options.compile_only, "Validate the graph without connecting");
  app.add_flag("--holoviz", options.holoviz, "Display the simulator camera");
  app.add_flag("--gamepad", options.gamepad, "Enable keyboard/gamepad velocity input");
  app.parse(argc, argv);
}

int run(const Options& options) {
  auto shutdown = holoscan::ManagedShutdownSignals::create();
  if (!shutdown) throw std::runtime_error("Unable to initialize managed shutdown.");
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess)
    throw std::runtime_error("Unable to initialize CUDA before UCX startup.");

  // Parse operator parameters.
  auto application_config =
      holoscan::config_io::load(options.config_path, holoscan::config_io::Format::kYaml);
  auto wbc_params = holoscan::params::decode<ops::WbcParams>(application_config, "/gr00t_wbc");

  // Instantiate operators.
  auto clock = std::make_shared<holoscan::SyntheticClock>(0);
  holoscan::Graph graph{"unitree-sim-wbc"};
  graph.set_default_clock(graph.bind_shared_clock("simulation_clock", clock));
  const auto sim = ops::add_unitree_sim_ucx_subgraph(graph, options.hostname, options.port, clock,
                                                     options.connection);
  auto wbc = graph.op<ops::AgileWbcOp>("wbc", /*velocity_from_teleop=*/options.gamepad);

  // Set operator parameters.
  wbc.set_params(std::move(wbc_params));

  // Define graph connections.
  // Same compute chain and WBC contract as the real app; pairing belongs to the sensor adapter.
  graph.add_flow(sim.state->joint_state, wbc->joint_state);
  graph.add_flow(sim.state->imu, wbc->imu);
  graph.add_flow(wbc->command, sim.sender->in_message);
  graph.partition("control").add(sim.state).add(wbc).add(sim.sender);
  if (options.gamepad) {
    auto teleop = graph.op<ops::TeleopTwistCommandOp>("teleop", /*emit_enable_signals=*/false);
    teleop.set_params(holoscan::params::decode<ops::TeleopParams>(application_config, "/gamepad"));
    graph.add_flow(teleop->velocity_commands, wbc->velocity_commands,
                   {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U});
    graph.partition("teleop").add(teleop);
  }
  if (options.holoviz) {
    holoscan::visualization::LayerStack layers;
    layers.add_image_layer("camera", {.format = holoscan::visualization::ImageFormat::kR8G8B8Unorm,
                                      .max_width = 1920,
                                      .max_height = 1080});
    auto viz = graph.op<holoscan::visualization::HolovizOp>(
        "holoviz", std::move(layers),
        holoscan::visualization::SinkDesc{holoscan::visualization::WindowSinkDesc{
            .title = "G1 Simulator Camera", .width = 640, .height = 480}});
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    graph.add_flow(sim.image->out_message, viz->image_layer_input(0), latest);
    graph.partition("holoviz").add(viz);
    graph.partition("image_transport").add(sim.image);
  } else {
    auto drain = graph.op<ops::MessageDropSinkOp>("image_drain");
    graph.add_flow(sim.image->out_message, drain->in_message);
    graph.partition("image_transport").add(sim.image);
    graph.partition("image_sink").add(drain);
  }

  // Compile and run.
  holoscan::CompileOptions compile_options;
  compile_options.deployment.bind_tensor_output_device(
      {.operator_path = "image_rx", .output_port = "out_message", .device = holoscan::DeviceId{0}});
  auto plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (options.compile_only) return 0;
  auto session = holoscan::run_async(plan, *shutdown);
  session.wait();
  if (session.termination() == holoscan::RunTermination::kFailed) {
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << holoscan::format_diagnostic(diagnostic, plan.graph()) << '\n';
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
  CLI::App app{"Run the HSDK5 WBC simulation application"};
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
