/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <CLI/CLI.hpp>
#include <array>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <cuda_runtime.h>
#include <holoscan/config/document.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "operators/agile_wbc/agile_wbc_op.h"
// Supplies ConfigCodec<HistoryTerm> for WbcParams::history_terms.
#include "operators/agile_wbc/history_term_config_codec.h"
#include "operators/g1_low_level/g1_low_level_joint_command_publisher.h"
#include "operators/g1_low_level/g1_low_level_joint_state_subscriber.h"
#include "operators/g1_low_level/mode_machine.h"
#include "operators/start_up/start_up_op.h"
#include "operators/start_up/start_up_params_config_codec.h"
#include "operators/teleop/teleop_params_config_codec.h"
#include "operators/teleop/teleop_twist_command_op.h"

namespace isaac::os {

namespace {

/// Applies the command-line DDS domain to the loaded document, so every consumer of the
/// `unitree_g1_robot` section sees one value rather than a struct patched after binding.
holoscan::config::Document with_domain_override(const holoscan::config::Document& document,
                                                int domain_id) {
  return document.patched(holoscan::config::Edit::add(
      "/unitree_g1_robot/dds_domain_id", holoscan::config::ConfigValue::signed_integer(domain_id)));
}

struct Options {
  std::filesystem::path config_path;
  std::optional<int> domain_id;
  std::optional<std::size_t> message_count;
  bool gamepad{};
};

void parse_options(CLI::App& app, Options& options, int argc, char** argv) {
  app.add_option("--config-path", options.config_path, "Application YAML configuration")
      ->required();
  app.add_option("--domain-id", options.domain_id,
                 "Unitree DDS domain; overrides the configured value")
      ->check(CLI::NonNegativeNumber);
  app.add_option("--message-count", options.message_count, "Stop after this many messages")
      ->check(CLI::PositiveNumber);
  app.add_flag("--gamepad", options.gamepad, "Enable keyboard/gamepad velocity input");
  app.parse(argc, argv);
}

bool has_cuda_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

int run(const Options& options, holoscan::ManagedShutdownSignals& signals) {
  if (!has_cuda_device()) {
    std::cerr << "No CUDA device is available; the Unitree G1 WBC policy cannot run.\n";
    return 1;
  }

  // Parse operator parameters.
  auto loaded = holoscan::config_io::load(options.config_path, holoscan::config_io::Format::kYaml);
  const holoscan::config::Document application_config =
      options.domain_id ? with_domain_override(loaded, *options.domain_id) : std::move(loaded);
  // StartUpParams carries the waypoint routine itself, so the file named by `config_file` is
  // merged into the section and the path key removed before decoding.
  const auto startup_routine =
      holoscan::config_io::load(application_config.get<std::string>("/start_up/config_file"),
                                holoscan::config_io::Format::kYaml);
  const holoscan::config::Document config =
      application_config.patched(holoscan::config::Edit::remove("/start_up/config_file"))
          .patched(
              holoscan::config::Edit::merge_object("/start_up", startup_routine.root().to_value()));
  auto dds_params = holoscan::params::decode<ops::G1DdsParams>(config, "/unitree_g1_robot");
  auto wbc_params = holoscan::params::decode<ops::WbcParams>(config, "/gr00t_wbc");
  auto startup_params = holoscan::params::decode<ops::StartUpParams>(config, "/start_up");

  // The DDS operators size their motor arrays in start(), so their count must match the policy.
  if (wbc_params->sdk_joint_order.size() != static_cast<std::size_t>(dds_params->g1_num_motors)) {
    throw std::runtime_error("The WBC configuration lists " +
                             std::to_string(wbc_params->sdk_joint_order.size()) +
                             " joints but the Unitree G1 DDS configuration expects " +
                             std::to_string(dds_params->g1_num_motors) + ".");
  }

  // Instantiate operators.
  holoscan::Graph graph{"unitree-g1-wbc"};
  auto mode_machine = std::make_shared<ModeMachine>();
  auto subscriber = graph.op<ops::G1JointStateSubscriberOp>("joint_state_subscriber", mode_machine);
  auto wbc = graph.op<ops::AgileWbcOp>("agile_wbc", /*velocity_from_teleop=*/options.gamepad);
  auto start_up = graph.op<ops::StartUpOp>("start_up", /*enable_signal_connected=*/options.gamepad);
  auto publisher = graph.op<ops::G1JointCommandPublisherOp>("joint_command_publisher", mode_machine,
                                                            options.message_count.value_or(0U));

  // Set operator parameters.
  subscriber.set_params(dds_params);
  publisher.set_params(dds_params);
  wbc.set_params(std::move(wbc_params));
  start_up.set_params(std::move(startup_params));

  // Define graph connections.
  const holoscan::ConnectionOptions latest_sample{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  graph.add_flow(subscriber->joint_state, wbc->joint_state);
  graph.add_flow(subscriber->imu, wbc->imu);
  // The startup ramp sits between the policy and the robot: it slews from the measured pose to
  // the routine's final pose before handing the joints over to the policy.
  graph.add_flow(subscriber->joint_state, start_up->joint_state, latest_sample);
  graph.add_flow(wbc->command, start_up->joint_command);
  graph.add_flow(start_up->command, publisher->command, latest_sample);

  if (options.gamepad) {
    auto teleop = graph.op<ops::TeleopTwistCommandOp>("teleop");
    teleop.set_params(holoscan::params::decode<ops::TeleopParams>(config, "/gamepad"));
    graph.add_flow(teleop->velocity_commands, wbc->velocity_commands, latest_sample);
    graph.add_flow(teleop->enable_lower_body, start_up->enable_lower, latest_sample);
    graph.add_flow(teleop->enable_upper_body, start_up->enable_upper, latest_sample);
  }
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("Unable to compile the Unitree G1 WBC graph.");
    for (const holoscan::Diagnostic& diagnostic : plan.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }
  if (!options.message_count) {
    return holoscan::run(plan);
  }

  // The publisher requests completion once it has written the requested number of commands.
  holoscan::RunSession session = holoscan::run_async(plan, signals);
  session.wait();
  if (session.termination() != holoscan::RunTermination::kRequestedCompleted) {
    HOLOSCAN_LOG_ERROR("The Unitree G1 WBC graph did not complete successfully.");
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
  // Establish the managed signal domain first: it must own the process signal mask before CUDA,
  // DDS, or the runtime start any participant thread, and the blocking run adopts it from here.
  auto signals = holoscan::ManagedShutdownSignals::create();
  if (!signals) {
    std::cerr << "Unable to establish managed shutdown signals: "
              << holoscan::runtime_error_message(signals.error()) << '\n';
    return 1;
  }

  // CLI11 keeps references to option values, so they must outlive the parser.
  isaac::os::Options options;
  CLI::App app{"Run the Unitree G1 whole-body control application"};
  try {
    isaac::os::parse_options(app, options, argc, argv);
    return isaac::os::run(options, *signals);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
