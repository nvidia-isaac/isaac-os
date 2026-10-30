/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <CLI/CLI.hpp>
#include <array>
#include <cstddef>
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
#include <vector>

#include <cuda_runtime.h>
#include <holoscan/config/document.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/threading/thread_placement.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "applications/gr00t_n1_7_wbc_g1/green_context.h"
#include "applications/gr00t_n1_7_wbc_g1/shutdown_signals.h"
#include "operators/agile_wbc/agile_wbc_op.h"
#include "operators/agile_wbc/history_term_config_codec.h"
#include "operators/g1_low_level/g1_low_level_joint_command_publisher.h"
#include "operators/g1_low_level/g1_low_level_joint_state_subscriber.h"
#include "operators/g1_low_level/mode_machine.h"
#include "operators/gr00t_action_handler/gr00t_action_handler_op.h"
#include "operators/gr00t_n1_7/gr00t_n1_7_fused_op.h"
#include "operators/gr00t_n1_7/gr00t_n1_7_model_config_codec.h"
#include "operators/joint_command_combiner/joint_command_combiner_op.h"
#include "operators/start_up/start_up_op.h"
#include "operators/start_up/start_up_params_config_codec.h"
#include "operators/teleop/teleop_params_config_codec.h"
#include "operators/teleop/teleop_twist_command_op.h"
#include "operators/v4l2_camera/v4l2_camera_op.h"

namespace isaac::os {
namespace {

struct ThreadPinningParams {
  bool enable{};
  std::uint32_t wbc_core{};
  int wbc_priority{};
  std::uint32_t gr00t_core{};
  int gr00t_priority{};
};

HOLOSCAN_PARAM_SCHEMA(ThreadPinningParams,
                      holoscan::param_schema("com.nvidia.isaac.Gr00tWbcThreadPinningParams", 1),
                      holoscan::field<&ThreadPinningParams::enable>("enable"),
                      holoscan::field<&ThreadPinningParams::wbc_core>("wbc_core"),
                      holoscan::field<&ThreadPinningParams::wbc_priority>("wbc_priority"),
                      holoscan::field<&ThreadPinningParams::gr00t_core>("gr00t_core"),
                      holoscan::field<&ThreadPinningParams::gr00t_priority>("gr00t_priority"));

struct Options {
  std::filesystem::path config_path;
  std::filesystem::path model_config_path;
  std::optional<std::uint32_t> green_context_wbc_sms;
  int dds_domain_id{};
  std::string network_interface;
  bool gamepad{};
};

void parse_options(CLI::App& app, Options& options, int argc, char** argv) {
  app.add_option("--config-path", options.config_path, "Application YAML configuration")
      ->required()
      ->take_last();
  app.add_option("--model-config-path", options.model_config_path, "GR00T model YAML configuration")
      ->required()
      ->take_last();
  app.add_option("--green-context-wbc-sms", options.green_context_wbc_sms,
                 "WBC SM count (CLI overrides YAML; default: device minimum; 0: off)")
      ->check(CLI::NonNegativeNumber)
      ->take_last();
  app.add_option("--domain-id", options.dds_domain_id, "DDS domain ID (0 = physical robot)")
      ->check(CLI::NonNegativeNumber)
      ->take_last();
  app.add_option("--network-interface", options.network_interface, "Robot network interface");
  app.add_flag("--gamepad", options.gamepad, "Enable keyboard/gamepad velocity input");
  app.parse(argc, argv);
}

bool has_cuda_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void report_diagnostics(const holoscan::ExecutionPlan& plan, const holoscan::RunSession& session) {
  const holoscan::ExecutionGraphView graph = plan.graph();
  for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
    HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, graph));
  }
}

int run(const Options& options, const ShutdownSignals& signals) {
  if (!has_cuda_device()) {
    std::cerr << "No GPU is available, so the robot cannot be controlled.\n";
    return 77;
  }

  const auto application_config =
      holoscan::config_io::load(options.config_path, holoscan::config_io::Format::kYaml);
  const auto model_config =
      holoscan::config_io::load(options.model_config_path, holoscan::config_io::Format::kYaml);
  const auto startup_config =
      holoscan::config_io::load(application_config.get<std::string>("/start_up/config_file"),
                                holoscan::config_io::Format::kYaml);
  const auto inference_params = model_config.at("/gr00t_n1_7");
  std::vector<holoscan::config::ConfigObjectEntry> dds_overrides{
      {"dds_domain_id", holoscan::config::ConfigValue::signed_integer(options.dds_domain_id)},
  };
  if (!options.network_interface.empty()) {
    dds_overrides.push_back({
        "network_interface",
        holoscan::config::ConfigValue::string(options.network_interface),
    });
  }
  const auto operator_config =
      application_config.patched(holoscan::config::Edit::remove("/start_up/config_file"));
  const auto composed_config = operator_config.patched(std::array{
      holoscan::config::Edit::merge_object("/start_up", startup_config.root().to_value()),
      holoscan::config::Edit::add("/gr00t_n1_7/inference", inference_params.to_value()),
      holoscan::config::Edit::merge_object("/unitree_g1_robot",
                                           holoscan::config::ConfigValue::object(dds_overrides)),
  });
  auto dds_params =
      holoscan::params::decode<ops::G1DdsParams>(composed_config, "/unitree_g1_robot");
  auto startup_params = holoscan::params::decode<ops::StartUpParams>(composed_config, "/start_up");
  if (static_cast<std::size_t>(dds_params->g1_num_motors) !=
      startup_params->sdk_joint_order.size()) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "The motor count must match the startup joint order.");
  }
  const auto requested_wbc_sms =
      load_wbc_green_context_sms(application_config, options.green_context_wbc_sms);
  const auto pinning =
      holoscan::params::decode<ThreadPinningParams>(application_config, "/operator_pinning");

  holoscan::Graph graph{"gr00t-n1-7-wbc-g1"};
  auto mode_machine = std::make_shared<ModeMachine>();
  auto joint_state =
      graph.op<ops::G1JointStateSubscriberOp>("joint_state_subscriber", mode_machine);
  auto camera = graph.op<ops::V4l2CameraOp>("camera");
  auto wbc = graph.op<ops::AgileWbcOp>("agile_wbc", /*velocity_from_teleop=*/options.gamepad);
  auto gr00t = graph.op<ops::Gr00tN17FusedOp>("gr00t_n1_7");
  auto action_handler = graph.op<ops::Gr00tActionHandlerOp>("gr00t_action_handler");
  auto combiner = graph.op<ops::JointCommandCombinerOp>("joint_command_combiner");
  auto start_up = graph.op<ops::StartUpOp>("start_up", /*enable_signal_connected=*/options.gamepad);
  auto publisher =
      graph.op<ops::G1JointCommandPublisherOp>("joint_command_publisher", mode_machine);

  joint_state.set_params(dds_params);
  publisher.set_params(std::move(dds_params));
  camera.set_params(holoscan::params::decode<ops::V4l2CameraParams>(composed_config, "/camera"));
  gr00t.set_params(holoscan::params::decode<ops::Gr00tN17Params>(composed_config, "/gr00t_n1_7"));
  action_handler.set_params(holoscan::params::decode<ops::Gr00tActionHandlerParams>(
      composed_config, "/gr00t_action_handler"));
  combiner.set_params(holoscan::params::decode<ops::JointCommandCombinerParams>(
      composed_config, "/joint_command_combiner"));
  start_up.set_params(std::move(startup_params));
  wbc.set_params(holoscan::params::decode<ops::WbcParams>(composed_config, "/gr00t_wbc"));
  const holoscan::ConnectionOptions latest_sample{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };

  graph.add_flow(joint_state->joint_state, wbc->joint_state);
  graph.add_flow(joint_state->imu, wbc->imu);
  // Retain state history for delayed camera frames.
  graph.add_flow(joint_state->joint_state, gr00t->joint_state,
                 {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 4U});
  graph.add_flow(joint_state->joint_state, start_up->joint_state, latest_sample);
  graph.add_flow(camera->image, gr00t->image, latest_sample);
  graph.add_flow(gr00t->action_chunk, action_handler->action_chunk);
  graph.add_flow(wbc->command, action_handler->control_command);
  // Select the overlay before forwarding its matching WBC command.
  graph.add_flow(action_handler->action_command, combiner->overlay_command, latest_sample);
  graph.add_flow(action_handler->forwarded_control_command, combiner->primary_command);
  graph.add_flow(combiner->command, start_up->joint_command);
  graph.add_flow(start_up->command, publisher->command, latest_sample);

  if (options.gamepad) {
    auto teleop = graph.op<ops::TeleopTwistCommandOp>("teleop");
    teleop.set_params(holoscan::params::decode<ops::TeleopParams>(composed_config, "/gamepad"));
    graph.add_flow(teleop->velocity_commands, wbc->velocity_commands, latest_sample);
    graph.add_flow(teleop->enable_lower_body, start_up->enable_lower, latest_sample);
    graph.add_flow(teleop->enable_upper_body, start_up->enable_upper, latest_sample);
    graph.partition("teleop").add(teleop);
  }

  graph.partition("wbc")
      .add(joint_state)
      .add(wbc)
      .add(action_handler)
      .add(combiner)
      .add(start_up)
      .add(publisher);
  graph.partition("gr00t").add(gr00t);
  graph.partition("camera").add(camera);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions compile_options;
  compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "camera",
      .output_port = "out_image",
      .device = holoscan::DeviceId{0},
  });
  const auto wbc_sms = configure_wbc_green_context(compile_options, requested_wbc_sms);

  if (pinning->enable) {
    compile_options.deployment
        .bind_thread_placement(
            "wbc",
            holoscan::ThreadPlacement{
                .affinity = holoscan::CPUAffinity{pinning->wbc_core},
                .scheduling =
                    holoscan::ThreadScheduling{.policy = holoscan::SchedulingPolicy::kFIFO,
                                               .priority = pinning->wbc_priority},
            })
        .bind_thread_placement(
            "gr00t", holoscan::ThreadPlacement{
                         .affinity = holoscan::CPUAffinity{pinning->gr00t_core},
                         .scheduling =
                             holoscan::ThreadScheduling{.policy = holoscan::SchedulingPolicy::kFIFO,
                                                        .priority = pinning->gr00t_priority},
                     });
  }

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("Unable to compile the GR00T and WBC graph.");
    for (const holoscan::Diagnostic& diagnostic : plan.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }

  validate_wbc_green_context(plan, wbc_sms);
  if (signals.stop_requested()) {
    return 0;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  signals.wait(session);
  if (session.termination() != holoscan::RunTermination::kStopped) {
    HOLOSCAN_LOG_ERROR("The robot control graph stopped unexpectedly.");
    report_diagnostics(plan, session);
    return 1;
  }
  return 0;
}

}  // namespace
}  // namespace isaac::os

int main(int argc, char** argv) {
  // Set before CUDA initialization.
  setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", /*overwrite=*/0);
  isaac::os::Options options;
  CLI::App app{"Run the GR00T and WBC real application"};
  try {
    isaac::os::parse_options(app, options, argc, argv);
    const isaac::os::ShutdownSignals signals;
    return isaac::os::run(options, signals);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
