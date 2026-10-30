/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <charconv>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include "holoscan/core/compile.hpp"
#include "holoscan/core/diagnostic.hpp"
#include "holoscan/core/run.hpp"
#include <fmt/format.h>

#include "common/mcap_logger/mcap_logger.h"
#include "reference_graph.h"

namespace isaac::os::mcap_logger_reference {
namespace {

struct Options {
  std::chrono::seconds duration{10};
  std::filesystem::path output_path;
  bool enable_write_file{true};
  bool enable_stream{false};
  std::string websocket_host{"127.0.0.1"};
  std::uint16_t websocket_port{8765};
  bool help{};
};

std::uint16_t parse_port(std::string_view value) {
  long long port{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), port);
  if (error != std::errc{} || end != value.data() + value.size() || port < 0 || port > 65535) {
    throw std::invalid_argument("--websocket-port must be an integer between 0 and 65535.");
  }
  return static_cast<std::uint16_t>(port);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--help" || argument == "-h") {
      options.help = true;
    } else if (argument == "--stream") {
      options.enable_stream = true;
    } else if (argument == "--no-file") {
      options.enable_write_file = false;
    } else if (argument.starts_with("--duration-s=")) {
      const std::string_view value = argument.substr(std::string_view{"--duration-s="}.size());
      long long seconds{};
      const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), seconds);
      if (error != std::errc{} || end != value.data() + value.size() || seconds <= 0) {
        throw std::invalid_argument("--duration-s must be a positive integer.");
      }
      options.duration = std::chrono::seconds(seconds);
    } else if (argument.starts_with("--output-path=")) {
      options.output_path = argument.substr(std::string_view{"--output-path="}.size());
    } else if (argument.starts_with("--websocket-host=")) {
      options.websocket_host = argument.substr(std::string_view{"--websocket-host="}.size());
    } else if (argument.starts_with("--websocket-port=")) {
      options.websocket_port =
          parse_port(argument.substr(std::string_view{"--websocket-port="}.size()));
    } else {
      throw std::invalid_argument("Unknown argument: " + std::string{argument} + ".");
    }
  }
  if (!options.enable_write_file && !options.enable_stream) {
    throw std::invalid_argument(
        "--no-file requires --stream -- otherwise there is no output at all.");
  }
  if (options.output_path.empty()) {
    // Relative, not a temp directory: resolves against the current working directory, which
    // metadata.json's "workdir": "isaac_root" makes the (bind-mounted, host-persistent) repo
    // root when run via ./isaac_os run -- unlike a temp directory, which --rm deletes on exit.
    options.output_path = "mcap_logger_output";
  }
  return options;
}

void print_usage(std::string_view executable) {
  std::cout << "Usage: " << executable
            << " [--duration-s=<positive integer, default 10>]"
               " [--output-path=<path, default \"mcap_logger_output\" under the current "
               "directory>]"
               " [--stream] [--no-file]"
               " [--websocket-host=<address, default 127.0.0.1>]"
               " [--websocket-port=<0-65535, default 8765; 0 lets the OS pick>]\n";
}

int run(const Options& options) {
  using isaac::os::mcap_logger::HsdkMcapLogger;
  using isaac::os::mcap_logger::HsdkMcapLoggerOptions;

  HsdkMcapLogger logger(HsdkMcapLoggerOptions{.output_path = options.output_path,
                                              .enable_write_file = options.enable_write_file,
                                              .enable_stream = options.enable_stream,
                                              .websocket_host = options.websocket_host,
                                              .websocket_port = options.websocket_port});
  ReferenceGraph graph(logger);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << "Unable to compile the mcap_logger_reference graph.\n";
    for (const holoscan::Diagnostic& diagnostic : plan.diagnostics()) {
      std::cerr << "  " << diagnostic.token << " (service " << diagnostic.service << ")\n";
    }
    return 1;
  }

  logger.start(plan.endpoint_catalog());
  holoscan::RunSession run_session = holoscan::run_async(plan);

  // The same counter pattern is recorded at two resolutions when this host can encode:
  // counter_image raw at 640x480, and counter_image_encoded with video encoding on at
  // 1920x1080, which writes two channels -- the encoded video itself and a
  // counter_image_encoded/metadata sidecar carrying the holoscan.schema.Image fields
  // foxglove.CompressedVideo cannot hold. On a host with no usable NVENC the graph leaves the
  // encoded topic out, so only counter_image is recorded.
  std::cout << "Logging sine_joint_state, cosine_joint_state, circular_pose, circular_twist, "
               "circular_transform, camera_info, counter_image"
            << (graph.records_encoded_topic()
                    ? ", counter_image_encoded, counter_image_encoded/metadata"
                    : " (counter_image_encoded is skipped: this host has no usable hardware video "
                      "encoder)")
            << " for " << options.duration.count() << "s"
            << (options.enable_stream
                    ? (options.websocket_port == 0
                           ? " (streaming live on an OS-assigned port; see the startup log above "
                             "for the bound port)"
                           : fmt::format(" (streaming live on {}:{})", options.websocket_host,
                                         options.websocket_port))
                    : "")
            << "...\n"
            << std::flush;
  std::this_thread::sleep_for(options.duration);

  run_session.request_stop();
  run_session.wait();
  logger.shutdown();

  if (logger.output_path()) {
    std::cout << "Wrote " << *logger.output_path() << "\n";
  } else {
    std::cout << "Stream-only run complete; no file was written.\n";
  }
  return 0;
}

}  // namespace
}  // namespace isaac::os::mcap_logger_reference

int main(int argc, char** argv) {
  try {
    const isaac::os::mcap_logger_reference::Options options =
        isaac::os::mcap_logger_reference::parse_options(argc, argv);
    if (options.help) {
      isaac::os::mcap_logger_reference::print_usage(argv[0]);
      return 0;
    }
    return isaac::os::mcap_logger_reference::run(options);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    isaac::os::mcap_logger_reference::print_usage(argv[0]);
    return 1;
  }
}
