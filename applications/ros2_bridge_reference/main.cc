/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
#include <rcutils/logging.h>

#include <cuda_runtime.h>
#endif

#include "holoscan/core/compile.hpp"
#include "holoscan/core/diagnostic.hpp"
#include "holoscan/core/run.hpp"
#include "holoscan/logger/logger.hpp"

#include "applications/ros2_bridge_reference/image_check_op.h"
#include "applications/ros2_bridge_reference/image_relay_node.h"
#include "applications/ros2_bridge_reference/pose_relay_node.h"
#include "applications/ros2_bridge_reference/reference_graph.h"
#include "common/ros2_bridge/ros2_bridge.h"

// A signal handler is a function with C language linkage, so these two live at global scope.
// The flag is a function-local static, set only by the handler.
volatile std::sig_atomic_t& stop_signalled() {
  static volatile std::sig_atomic_t flag = 0;
  return flag;
}

// Only records the request, and puts the signal's default action back so a second one is fatal.
extern "C" void request_stop_on_signal(int signal_number) {
  stop_signalled() = 1;
  std::signal(signal_number, SIG_DFL);
}

namespace isaac::os::ros2_bridge_reference {
namespace {

constexpr auto kSignalPollInterval = std::chrono::milliseconds(50);

struct Options {
  std::chrono::seconds duration{10};
  bool help{};
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--help" || argument == "-h") {
      options.help = true;
    } else if (argument.starts_with("--duration-s=")) {
      const std::string_view value = argument.substr(std::string_view{"--duration-s="}.size());
      long long seconds{};
      const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), seconds);
      if (error != std::errc{} || end != value.data() + value.size() || seconds <= 0) {
        throw std::invalid_argument("--duration-s must be a positive integer.");
      }
      options.duration = std::chrono::seconds(seconds);
    } else {
      throw std::invalid_argument("Unknown argument: " + std::string{argument} + ".");
    }
  }
  return options;
}

void print_usage(std::string_view executable) {
  std::cout << "Usage: " << executable << " [--duration-s=<positive integer, default 10>]\n";
}

int run(const Options& options) {
  // The GPU image path runs automatically when ros2_bridge was built with the CUDA buffer backend
  // and this machine has a CUDA device; otherwise the app warns and is the pose round trip alone.
  bool cuda_image = false;
#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
  int cuda_device_count = 0;
  cuda_image = cudaGetDeviceCount(&cuda_device_count) == cudaSuccess && cuda_device_count > 0;
  if (!cuda_image) {
    (void)cudaGetLastError();
    HOLOSCAN_LOG_WARN("No CUDA device found; skipping the GPU image path.");
  }
#else
  HOLOSCAN_LOG_WARN(
      "ros2_bridge was built without the CUDA buffer backend; skipping the GPU image path.");
#endif

  // The relay node below shares this process, so give both nodes intra-process communication: the
  // GPU images between them are then handed over in place, instead of through the ROS CUDA buffer
  // backend's same-process IPC path, which blocks every publish on the GPU.
  isaac::os::ros2_bridge::Ros2Bridge<> bridge(
      "ros2_bridge_reference_bridge_node", 0,
      rclcpp::NodeOptions().use_intra_process_comms(cuda_image));

  ReferenceGraph graph(bridge, cuda_image);
  // A device topic needs its device bound at compile time: the generator's output (the tensor it
  // emits) and the imported topic (where the received tensor lands), both on the bridge's pool GPU.
  holoscan::CompileOptions compile_options;
#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
  if (cuda_image) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = "image_generator",
                                              .output_port = "out_image",
                                              .device = holoscan::DeviceId{bridge.pool_device()}});
    compile_options.deployment.bind_imported_topic_device(
        {.endpoint_name = "in_image", .device = holoscan::DeviceId{bridge.pool_device()}});
  }
#endif
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, compile_options);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("Unable to compile the ros2_bridge_reference graph.");
    for (const holoscan::Diagnostic& diagnostic : plan.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }

  bridge.start(plan.endpoint_catalog());
#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
  // ROS logs every image buffer it serializes or deserializes at info level, which buries this
  // app's own output at the image rate. Warnings and errors from those loggers still show.
  for (const char* logger :
       {"serialize_buffer_with_endpoint", "deserialize_buffer_with_endpoint"}) {
    rcutils_logging_set_logger_level(logger, RCUTILS_LOG_SEVERITY_WARN);
  }
#endif
  // The external ROS2 node this app talks to: subscribes what the bridge publishes out
  // (kPoseOutRos2Topic) and republishes it on the topic the bridge subscribes back in
  // (kPoseInRos2Topic). Created after bridge.start() because rclcpp::init() is handled internally
  // by the bridge's start(), process-wide, so any in-process rclcpp node created after it shares
  // that context and needs no init call of its own. It is still created before the graph runs, so
  // its subscription exists by the time the generator's first tick publishes.
  PoseRelayNode relay("ros2_bridge_reference_external_node", kPoseOutRos2Topic, kPoseInRos2Topic);
#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
  std::unique_ptr<ImageRelayNode> image_relay;
  if (cuda_image) {
    image_relay = std::make_unique<ImageRelayNode>("ros2_bridge_reference_image_relay_node",
                                                   kImageOutRos2Topic, kImageInRos2Topic);
  }
#endif
  // Give ROS2 discovery a moment to match the bridge's publisher/subscription against the relay
  // node's before the first tick publishes -- a message published before discovery completes can
  // be silently dropped (see test_ros2_bridge_roundtrip.cc's own comment on the same race).
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  holoscan::RunSession run_session = holoscan::run_async(plan);

  // The bridge leaves signals to the app, so until this point Ctrl+C kills the process, including
  // if startup hangs. From here a signal ends the run early and takes the same orderly stop as the
  // timer; a second one is not caught, so it still kills a shutdown that hangs.
  std::signal(SIGINT, request_stop_on_signal);
  std::signal(SIGTERM, request_stop_on_signal);

  // Printed after the handlers are installed, so a signal sent once this line appears is handled;
  // the SIGINT test waits for it.
  std::cout << "Publishing circular_pose to " << kPoseOutRos2Topic
            << ", relaying through an external ROS2 node, and reading it back on "
            << kPoseInRos2Topic << " for " << options.duration.count()
            << "s; Ctrl+C stops it early.\n"
            << std::flush;
#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
  if (cuda_image) {
    std::cout << "Also round-tripping a 640x480 image as a CUDA buffer through "
              << kImageOutRos2Topic << " and " << kImageInRos2Topic << ".\n"
              << std::flush;
  }
#endif
  const auto deadline = std::chrono::steady_clock::now() + options.duration;
  while (!stop_signalled() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(kSignalPollInterval);
  }
  if (stop_signalled()) {
    HOLOSCAN_LOG_INFO("Stop requested; stopping the ros2_bridge_reference graph.");
  }

  run_session.request_stop();
  run_session.wait();
  if (run_session.termination() != holoscan::RunTermination::kStopped) {
    HOLOSCAN_LOG_ERROR("The ros2_bridge_reference graph did not stop cleanly.");
    for (const holoscan::Diagnostic& diagnostic : run_session.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    bridge.stop();
    return 1;
  }
  bridge.stop();

  // PoseGeneratorOp/PoseLoggerOp already logged their own sent/received tallies from their
  // stop() hooks, run as part of request_stop()/wait() above; this is the third leg of that same
  // count, from the one participant that isn't a Holoscan operator. Same HOLOSCAN_LOG_INFO as
  // those two, not std::cout, so all three tallies land in the same log stream for comparison.
  const std::uint64_t relayed_count = relay.relayed_count();
  HOLOSCAN_LOG_INFO("External relay node relayed {} poses.", relayed_count);
  if (relayed_count == 0) {
    HOLOSCAN_LOG_ERROR(
        "No pose completed the round trip through the external relay node -- ros2_bridge did not "
        "deliver anything.");
    return 1;
  }

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
  if (cuda_image) {
    const std::uint64_t cuda_images = image_relay->cuda_count();
    const std::uint64_t host_images = image_relay->host_count();
    HOLOSCAN_LOG_INFO("External relay node relayed {} images as CUDA buffers.", cuda_images);
    if (host_images > 0 || cuda_images == 0) {
      HOLOSCAN_LOG_ERROR(
          "{} images reached the external relay node in host memory and {} as CUDA buffers. The "
          "ROS "
          "CUDA buffer backend plugin is probably not installed, so frames are copied to the host "
          "instead of staying on the GPU.",
          host_images, cuda_images);
      return 1;
    }
    const ImageCheckTally& tally = image_check_tally();
    const std::uint64_t sent_images = image_sent_count().load();
    const std::uint64_t received_images = tally.received.load();
    if (received_images == 0 || tally.bad_images.load() > 0) {
      HOLOSCAN_LOG_ERROR(
          "The graph received {} images, {} of them with the wrong memory, size, encoding or "
          "pixels.",
          received_images, tally.bad_images.load());
      return 1;
    }
    // Every image should make the round trip. Frames still in transit when the graph stops are not
    // delivered, so a couple may be missing.
    constexpr std::uint64_t kInFlightImages = 2;
    if (sent_images < received_images || sent_images - received_images > kInFlightImages) {
      HOLOSCAN_LOG_ERROR(
          "The generator sent {} images and the graph received {}; more than {} are missing.",
          sent_images, received_images, kInFlightImages);
      return 1;
    }
  }
#endif
  return 0;
}

}  // namespace
}  // namespace isaac::os::ros2_bridge_reference

int main(int argc, char** argv) {
  try {
    const isaac::os::ros2_bridge_reference::Options options =
        isaac::os::ros2_bridge_reference::parse_options(argc, argv);
    if (options.help) {
      isaac::os::ros2_bridge_reference::print_usage(argv[0]);
      return 0;
    }
    return isaac::os::ros2_bridge_reference::run(options);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    isaac::os::ros2_bridge_reference::print_usage(argv[0]);
    return 1;
  }
}
