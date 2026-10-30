/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <CLI/CLI.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <holoscan/config/document.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/threading/thread_placement.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "applications/gr00t_n1_7_wbc_g1/green_context.h"
#include "applications/gr00t_n1_7_wbc_g1/mock/mock_operators.h"
#include "applications/gr00t_n1_7_wbc_g1/shutdown_signals.h"
#include "common/benchmark/core_roles.h"
#include "operators/agile_wbc/agile_wbc_op.h"
#include "operators/agile_wbc/history_term_config_codec.h"
#include "operators/gr00t_action_handler/gr00t_action_handler_op.h"
#include "operators/gr00t_n1_7/gr00t_n1_7_fused_op.h"
#include "operators/gr00t_n1_7/gr00t_n1_7_model_config_codec.h"
#include "operators/joint_command_combiner/joint_command_combiner_op.h"

namespace isaac::os {
namespace {

struct Options {
  std::filesystem::path application_config_path;
  std::filesystem::path model_config_path;
  std::optional<std::uint32_t> green_context_wbc_sms;
  std::optional<std::size_t> message_count;
  benchmark::CoreRoles benchmark_roles;
  bool benchmark_affinity_only{};
};

holoscan::ThreadPlacement benchmark_placement(std::uint32_t core, holoscan::SchedulingPolicy policy,
                                              std::optional<int> priority) {
  return holoscan::ThreadPlacement{
      .affinity = holoscan::CPUAffinity{core},
      .scheduling = holoscan::ThreadScheduling{.policy = policy, .priority = priority},
      .wake_mode = holoscan::WakeMode::kSleepingWaitset,
      .irq_isolation = holoscan::IRQIsolation::kNotRequested,
  };
}

void validate_benchmark_placements(const holoscan::ExecutionPlan& plan,
                                   const benchmark::CoreRoles& roles, bool affinity_only) {
  struct Expected {
    std::string_view name;
    std::uint32_t core;
    holoscan::SchedulingPolicy policy;
    std::optional<int> priority;
  };
  const auto realtime_policy =
      affinity_only ? holoscan::SchedulingPolicy::kOther : holoscan::SchedulingPolicy::kFIFO;
  const std::array expected{
      Expected{"image_source", *roles.background, holoscan::SchedulingPolicy::kOther, std::nullopt},
      Expected{"gr00t", *roles.gr00t, realtime_policy,
               affinity_only ? std::nullopt : std::optional<int>{benchmark::kGr00tFifoPriority}},
      Expected{"wbc", *roles.wbc, realtime_policy,
               affinity_only ? std::nullopt : std::optional<int>{benchmark::kWbcFifoPriority}},
  };
  for (const auto& requirement : expected) {
    const auto placement = std::find_if(
        plan.partition_placements().begin(), plan.partition_placements().end(),
        [&](const auto& candidate) { return candidate.authored_name == requirement.name; });
    if (placement == plan.partition_placements().end() || !placement->accepted ||
        !placement->accepted->affinity ||
        placement->accepted->affinity->cores != std::vector<std::uint32_t>{requirement.core} ||
        placement->accepted->scheduling.policy != requirement.policy ||
        placement->accepted->scheduling.priority != requirement.priority) {
      throw std::runtime_error("Benchmark placement was not accepted for partition '" +
                               std::string{requirement.name} + "'.");
    }
  }
}

void parse_options(CLI::App& app, Options& options, int argc, char** argv) {
  app.add_option("--application-config-path", options.application_config_path,
                 "Application YAML configuration")
      ->required()
      ->take_last();
  app.add_option("--model-config-path", options.model_config_path, "GR00T model YAML configuration")
      ->required();
  app.add_option("--green-context-wbc-sms", options.green_context_wbc_sms,
                 "WBC SM count (CLI overrides YAML; default: device minimum; 0: off)")
      ->check(CLI::NonNegativeNumber)
      ->take_last();
  app.add_option("--message-count", options.message_count, "Stop after this many messages")
      ->check(CLI::PositiveNumber);
  auto* controller_core =
      app.add_option("--benchmark-controller-core", options.benchmark_roles.controller,
                     "CPU core for the graph controller")
          ->check(CLI::NonNegativeNumber);
  auto* background_core =
      app.add_option("--benchmark-background-core", options.benchmark_roles.background,
                     "CPU core for background work")
          ->check(CLI::NonNegativeNumber);
  auto* gr00t_core =
      app.add_option("--benchmark-gr00t-core", options.benchmark_roles.gr00t, "CPU core for GR00T")
          ->check(CLI::NonNegativeNumber);
  auto* wbc_core =
      app.add_option("--benchmark-wbc-core", options.benchmark_roles.wbc, "CPU core for WBC")
          ->check(CLI::NonNegativeNumber);
  app.add_flag("--benchmark-affinity-only", options.benchmark_affinity_only,
               "Set benchmark CPU affinity without real-time scheduling")
      ->needs(controller_core, background_core, gr00t_core, wbc_core);

  app.parse(argc, argv);
  try {
    options.benchmark_roles.validate();
  } catch (const std::invalid_argument& error) {
    throw CLI::ValidationError{error.what()};
  }
}

bool has_cuda_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

int run(const Options& options, const ShutdownSignals& signals) {
  if (!has_cuda_device()) {
    std::cerr << "No CUDA device is available; skipping the GR00T and WBC application.\n";
    return ISAAC_CTEST_SKIP_RETURN_CODE;
  }
  auto application_config = holoscan::config_io::load(options.application_config_path,
                                                      holoscan::config_io::Format::kYaml);
  const auto requested_wbc_sms =
      load_wbc_green_context_sms(application_config, options.green_context_wbc_sms);
  auto model_config =
      holoscan::config_io::load(options.model_config_path, holoscan::config_io::Format::kYaml);
  auto inference_params = model_config.at("/gr00t_n1_7");
  auto composed_config = application_config.patched(
      holoscan::config::Edit::add("/gr00t_n1_7/inference", inference_params.to_value()));
  auto wbc_params = holoscan::params::decode<ops::WbcParams>(composed_config, "/gr00t_wbc");
  auto gr00t_params = holoscan::params::decode<ops::Gr00tN17Params>(composed_config, "/gr00t_n1_7");
  auto action_handler_params = holoscan::params::decode<ops::Gr00tActionHandlerParams>(
      composed_config, "/gr00t_action_handler");
  auto combiner_params = holoscan::params::decode<ops::JointCommandCombinerParams>(
      composed_config, "/joint_command_combiner");
  auto image_params =
      holoscan::params::decode<MockImageParams>(composed_config, "/mock_image_source");
  const std::size_t joint_count = combiner_params->joint_order.size();

  holoscan::Graph graph{"mock-gr00t-n1-7-wbc-g1"};
  const auto state_source = graph.op<MockRobotStateOp>("mock_joint_state_source", joint_count);
  auto image_source = graph.op<MockImageOp>("mock_image_source");
  auto wbc = graph.op<ops::AgileWbcOp>("agile_wbc");
  auto gr00t = graph.op<ops::Gr00tN17FusedOp>("gr00t_n1_7");
  auto action_handler = graph.op<ops::Gr00tActionHandlerOp>("gr00t_action_handler");
  auto combiner = graph.op<ops::JointCommandCombinerOp>("joint_command_combiner");
  const auto command_sink = graph.op<MockCommandSinkOp>(
      "mock_joint_command_sink", options.message_count.value_or(0U), combiner_params->joint_order,
      combiner_params->joint_names_from_overlay);
  wbc.set_params(std::move(wbc_params));
  gr00t.set_params(std::move(gr00t_params));
  action_handler.set_params(std::move(action_handler_params));
  combiner.set_params(std::move(combiner_params));
  image_source.set_params(std::move(image_params));
  graph.add_flow(state_source->wbc_joint_state, wbc->joint_state);
  graph.add_flow(state_source->imu, wbc->imu);
  const holoscan::ConnectionOptions latest_sample{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  // Retain state history for delayed camera frames.
  graph.add_flow(state_source->gr00t_joint_state, gr00t->joint_state,
                 {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 4U});
  graph.add_flow(image_source->image, gr00t->image, latest_sample);
  graph.add_flow(gr00t->action_chunk, action_handler->action_chunk);
  graph.add_flow(wbc->command, action_handler->control_command);
  // Select the overlay before forwarding its matching WBC command.
  // Keep two overlay samples for fanout to the combiner and validation sink.
  const holoscan::ConnectionOptions overlay_history{
      .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 2U};
  graph.add_flow(action_handler->action_command, combiner->overlay_command, overlay_history);
  graph.add_flow(action_handler->action_command, command_sink->action_command, latest_sample);
  graph.add_flow(action_handler->forwarded_control_command, combiner->primary_command);
  graph.add_flow(action_handler->forwarded_control_command, command_sink->primary_command,
                 latest_sample);
  graph.add_flow(combiner->command, command_sink->command);
  graph.partition("image_source").add(image_source);
  graph.partition("wbc")
      .add(state_source)
      .add(wbc)
      .add(action_handler)
      .add(combiner)
      .add(command_sink);
  graph.partition("gr00t").add(gr00t);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  holoscan::CompileOptions compile_options;
  compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "mock_image_source",
      .output_port = "out_image",
      .device = holoscan::DeviceId{0},
  });
  const auto wbc_sms = configure_wbc_green_context(compile_options, requested_wbc_sms);
  if (options.benchmark_roles.enabled()) {
    const auto realtime_policy = options.benchmark_affinity_only
                                     ? holoscan::SchedulingPolicy::kOther
                                     : holoscan::SchedulingPolicy::kFIFO;
    const auto gr00t_priority = options.benchmark_affinity_only
                                    ? std::nullopt
                                    : std::optional<int>{benchmark::kGr00tFifoPriority};
    const auto wbc_priority = options.benchmark_affinity_only
                                  ? std::nullopt
                                  : std::optional<int>{benchmark::kWbcFifoPriority};
    compile_options.deployment
        .bind_thread_placement(
            "image_source", benchmark_placement(*options.benchmark_roles.background,
                                                holoscan::SchedulingPolicy::kOther, std::nullopt))
        .bind_thread_placement("gr00t", benchmark_placement(*options.benchmark_roles.gr00t,
                                                            realtime_policy, gr00t_priority))
        .bind_thread_placement("wbc", benchmark_placement(*options.benchmark_roles.wbc,
                                                          realtime_policy, wbc_priority));
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("Unable to compile the GR00T and WBC graph.");
    for (const holoscan::Diagnostic& diagnostic : plan.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }
  if (options.benchmark_roles.enabled()) {
    validate_benchmark_placements(plan, options.benchmark_roles, options.benchmark_affinity_only);
  }
  validate_wbc_green_context(plan, wbc_sms);

  if (options.benchmark_roles.enabled()) {
    benchmark::pin_current_thread(*options.benchmark_roles.controller);
  }
  if (signals.stop_requested()) {
    return 0;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  signals.wait(session);

  if (signals.stop_requested() && session.termination() == holoscan::RunTermination::kStopped) {
    return 0;
  }
  const auto expected_termination = options.message_count
                                        ? holoscan::RunTermination::kRequestedCompleted
                                        : holoscan::RunTermination::kStopped;
  if (session.termination() != expected_termination) {
    HOLOSCAN_LOG_ERROR("The GR00T and WBC graph did not complete successfully.");
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
  // Set before CUDA initialization.
  setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 0);
  isaac::os::Options options;
  CLI::App app{"Run the GR00T and WBC mock application"};
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
