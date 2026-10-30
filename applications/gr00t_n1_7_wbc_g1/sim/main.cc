/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <CLI/CLI.hpp>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <holoscan/config/document.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/time/synthetic_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "applications/gr00t_n1_7_wbc_g1/green_context.h"
#include "applications/gr00t_n1_7_wbc_g1/shutdown_signals.h"
#include "operators/agile_wbc/agile_wbc_op.h"
#include "operators/agile_wbc/history_term_config_codec.h"
#include "operators/gr00t_action_handler/gr00t_action_handler_op.h"
#include "operators/gr00t_n1_7/gr00t_n1_7_fused_op.h"
#include "operators/gr00t_n1_7/gr00t_n1_7_model_config_codec.h"
#include "operators/joint_command_combiner/joint_command_combiner_op.h"
#include "operators/start_up/start_up_op.h"
#include "operators/start_up/start_up_params_config_codec.h"
#include "operators/teleop/teleop_params_config_codec.h"
#include "operators/teleop/teleop_twist_command_op.h"
#include "operators/unitree_sim_ucx/unitree_sim_ucx_subgraph.h"

namespace isaac::os {
namespace {

struct Options {
  std::filesystem::path application_config_path;
  std::filesystem::path model_config_path;
  std::string hostname{"127.0.0.1"};
  int port = 50008;
  ops::UcxxConnectOptions connection;
  bool compile_only = false;
  bool holoviz = false;
  bool gamepad = false;
  std::optional<std::uint32_t> wbc_sms;
};

void parse_options(CLI::App& cli, Options& options, int argc, char** argv) {
  cli.add_option("--application-config-path,--config-path", options.application_config_path)
      ->required()
      ->take_last();
  cli.add_option("--model-config-path", options.model_config_path)->required()->take_last();
  cli.add_option("--hostname", options.hostname);
  cli.add_option("--port", options.port, "Simulator UCX port (1-65535)")
      ->check(CLI::Range(1, 65535));
  cli.add_option_function<std::int64_t>(
         "--connect-timeout-ms",
         [&options](std::int64_t ms) {
           options.connection.timeout = std::chrono::milliseconds{ms};
         },
         "Initial connection timeout in milliseconds (0: wait indefinitely)")
      ->check(CLI::NonNegativeNumber);
  cli.add_option_function<std::int64_t>(
         "--connect-retry-interval-ms",
         [&options](std::int64_t ms) {
           options.connection.retry_interval = std::chrono::milliseconds{ms};
         },
         "Delay between connection attempts in milliseconds (default: 1000)")
      ->check(CLI::PositiveNumber);
  cli.add_flag("--compile-only", options.compile_only);
  cli.add_flag("--holoviz", options.holoviz);
  cli.add_option("--green-context-wbc-sms", options.wbc_sms,
                 "WBC SM count (CLI overrides YAML; default: device minimum; 0: off)")
      ->check(CLI::NonNegativeNumber)
      ->take_last();
  cli.add_flag("--gamepad", options.gamepad, "Enable keyboard/gamepad velocity input");
  cli.parse(argc, argv);
}

int run(const Options& app_options, const ShutdownSignals& shutdown) {
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess)
    throw std::runtime_error("Unable to initialize CUDA before UCX startup.");
  // Parse operator parameters.
  auto application_config = holoscan::config_io::load(app_options.application_config_path,
                                                      holoscan::config_io::Format::kYaml);
  const auto requested_wbc_sms =
      load_wbc_green_context_sms(application_config, app_options.wbc_sms);
  auto model_config =
      holoscan::config_io::load(app_options.model_config_path, holoscan::config_io::Format::kYaml);
  // The shared model document carries two scheduler/allocation fields that are not part of the
  // fused inference schema. Remove exactly those fields before strict binding.
  auto patched_model_config = model_config.patched(std::array{
      holoscan::config::Edit::remove("/gr00t_n1_7/sync_after_compute"),
      holoscan::config::Edit::remove("/gr00t_n1_7/pool_blocks"),
  });
  auto inference = patched_model_config.at("/gr00t_n1_7");
  auto config = application_config.patched(
      holoscan::config::Edit::add("/sim/gr00t_n1_7/inference", inference.to_value()));
  // Instantiate operators.
  auto clock = std::make_shared<holoscan::SyntheticClock>(0);
  holoscan::Graph graph{"unitree-sim-n17-wbc"};
  graph.set_default_clock(graph.bind_shared_clock("simulation_clock", clock));
  const auto sim = ops::add_unitree_sim_ucx_subgraph(graph, app_options.hostname, app_options.port,
                                                     clock, app_options.connection);
  auto image = graph.op<ops::UnitreeSimImageToTensorOp>("camera_pixels");
  auto wbc = graph.op<ops::AgileWbcOp>("wbc", /*velocity_from_teleop=*/app_options.gamepad);
  auto gr00t = graph.op<ops::Gr00tN17FusedOp>("gr00t");
  auto handler = graph.op<ops::Gr00tActionHandlerOp>("action_handler");
  auto combiner = graph.op<ops::JointCommandCombinerOp>("combiner");
  auto start_up =
      graph.op<ops::StartUpOp>("start_up", /*enable_signal_connected=*/app_options.gamepad);
  // Set operator parameters. Retain the simulation model and its trained action interval.
  wbc.set_params(holoscan::params::decode<ops::WbcParams>(config, "/gr00t_wbc"));
  gr00t.set_params(holoscan::params::decode<ops::Gr00tN17Params>(config, "/sim/gr00t_n1_7"));
  handler.set_params(
      holoscan::params::decode<ops::Gr00tActionHandlerParams>(config, "/sim/gr00t_action_handler"));
  combiner.set_params(holoscan::params::decode<ops::JointCommandCombinerParams>(
      config, "/sim/joint_command_combiner"));

  start_up.set_params(holoscan::params::decode<ops::StartUpParams>(config, "/sim/start_up"));

  // Define graph connections.
  const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                           .queue_depth = 1U};
  graph.add_flow(sim.image->out_message, image->input, latest);
  graph.partition("image_transport").add(sim.image).add(image);
  graph.add_flow(sim.state->joint_state, wbc->joint_state);
  graph.add_flow(sim.state->imu, wbc->imu);
  // Retain the matching state until its camera frame arrives for aligned inference.
  graph.add_flow(sim.state->joint_state, gr00t->joint_state,
                 {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 4U});
  graph.add_flow(image->output, gr00t->image, latest);
  graph.add_flow(gr00t->action_chunk, handler->action_chunk);
  graph.add_flow(sim.state->joint_state, start_up->joint_state, latest);
  graph.add_flow(wbc->command, handler->control_command);
  // Select the overlay before forwarding its matching WBC command, as in the real app.
  graph.add_flow(handler->action_command, combiner->overlay_command, latest);
  graph.add_flow(handler->forwarded_control_command, combiner->primary_command);
  graph.add_flow(combiner->command, start_up->joint_command);
  graph.add_flow(start_up->command, sim.sender->in_message, latest);
  graph.partition("control").add(sim.state).add(wbc).add(handler).add(combiner).add(start_up).add(
      sim.sender);
  graph.partition("gr00t").add(gr00t);
  if (app_options.gamepad) {
    auto teleop = graph.op<ops::TeleopTwistCommandOp>("teleop");
    teleop.set_params(holoscan::params::decode<ops::TeleopParams>(config, "/gamepad"));
    graph.add_flow(teleop->velocity_commands, wbc->velocity_commands,
                   {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U});
    graph.add_flow(teleop->enable_lower_body, start_up->enable_lower, latest);
    graph.add_flow(teleop->enable_upper_body, start_up->enable_upper, latest);
    graph.partition("teleop").add(teleop);
  }
  if (app_options.holoviz) {
    const auto width = config.get<std::uint32_t>("/sim/holoviz/width");
    const auto height = config.get<std::uint32_t>("/sim/holoviz/height");
    const auto window_title = config.get<std::string>("/sim/holoviz/window_title");
    holoscan::visualization::LayerStack layers;
    layers.add_image_layer("camera", {.format = holoscan::visualization::ImageFormat::kR8G8B8Unorm,
                                      .max_width = 1920,
                                      .max_height = 1080});
    auto viz = graph.op<holoscan::visualization::HolovizOp>(
        "holoviz", std::move(layers),
        holoscan::visualization::SinkDesc{holoscan::visualization::WindowSinkDesc{
            .title = window_title, .width = width, .height = height}});
    graph.add_flow(sim.image->out_message, viz->image_layer_input(0), latest);
    graph.partition("holoviz").add(viz);
  }
  // Compile and run.
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      {.operator_path = "image_rx", .output_port = "out_message", .device = holoscan::DeviceId{0}});
  options.deployment.bind_tensor_output_device(
      {.operator_path = "camera_pixels", .output_port = "output", .device = holoscan::DeviceId{0}});
  const auto resolved_wbc_sms = configure_wbc_green_context(options, requested_wbc_sms, "control");
  auto plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  validate_wbc_green_context(plan, resolved_wbc_sms, "control");
  if (app_options.compile_only) return 0;
  if (shutdown.stop_requested()) {
    return 0;
  }
  auto session = holoscan::run_async(plan);
  shutdown.wait(session);
  for (const auto& diagnostic : session.diagnostics()) {
    std::cerr << holoscan::format_diagnostic(diagnostic, plan.graph()) << '\n';
  }
  return session.termination() == holoscan::RunTermination::kFailed ? 1 : 0;
}

}  // namespace
}  // namespace isaac::os

int main(int argc, char** argv) {
  setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 0);
  isaac::os::Options options;
  CLI::App cli{"Run the HSDK5 GR00T N1.7 and WBC simulation application"};
  try {
    isaac::os::parse_options(cli, options, argc, argv);
    const isaac::os::ShutdownSignals shutdown;
    return isaac::os::run(options, shutdown);
  } catch (const CLI::ParseError& error) {
    return cli.exit(error);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
