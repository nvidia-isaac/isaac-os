/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Drives the SO-101 arm in Isaac Sim with a sinusoidal joint-position command, receiving
// joint state and sending joint commands back over UCX. Sine phase and the
// emitted command.header.stamp are derived from the simulator timestamp
// carried in each incoming JointState header. Uses a private SinusMotionOp
// (co-located in this mode's sinus_motion_op.{h,cc}) that consumes
// isaac::JointStateCpuT directly on the UCX path, and loads its configuration
// from YAML. Three partitions by default (connection / joint_state_rx / control_and_tx),
// plus optional image_transport and holoviz partitions when enable_camera is true.

#include <CLI/CLI.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <holoscan/config/codec.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "applications/so101_sinus_motion/sim/sinus_motion_op.h"
#include "operators/ucxx_ops/message_traits.h"
#include "operators/ucxx_ops/ucxx_connect_op.h"
#include "operators/ucxx_ops/ucxx_endpoint.h"
#include "operators/ucxx_ops/ucxx_receiver_op.h"
#include "operators/ucxx_ops/ucxx_sender_op.h"

namespace isaac::os::apps::so101_sinus_motion_sim {

// FNV-1a 32-bit hashes of the FBS schema names used by Isaac Sim's isaacsim.ucx.core.
constexpr std::uint64_t kJointStateTag = 0xe2e1a077;    // isaac.JointState
constexpr std::uint64_t kJointCommandTag = 0x3ad708d3;  // isaac.JointCommand
constexpr std::uint64_t kImageTag = 0x1018c86b;         // isaac.Image

template <typename T>
T get_or(const holoscan::config::Document& config, std::string_view path, T default_value) {
  const auto value = config.find(path);
  return value ? holoscan::config::decode_config<T>(*value) : std::move(default_value);
}

// A tiny stop-after-N helper. Sits inline between joint_state_rx and the
// sinus generator: forwards each received JointState downstream unchanged,
// and once `stop_after_messages` samples have passed through, requests
// completion. Placing it inline (rather than as a second consumer on
// joint_state_rx->out_message) keeps every output port with a single
// consumer, as required by the compiled graph's PARTITION_COVER validation.
class TickCounter final : public holoscan::Operator<> {
 public:
  explicit TickCounter(std::size_t stop_after_messages)
      : stop_after_messages_(stop_after_messages) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(joint_state_in, "joint_state_in");
    spec.output(joint_state_out, "joint_state_out");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{joint_state_in});
    return contract;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto received = joint_state_in.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    auto emitted = joint_state_out.emit(
        std::move(received->data),
        holoscan::EmitOptions{.capture_time = received->metadata.capture_time()});
    if (!emitted) {
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    // Increment unconditionally so the RHS of the completion check has no
    // side effect (Sonar CPP:S3616 / CERT MSC15-C: keep short-circuited
    // operands pure).
    ++messages_seen_;
    if (stop_after_messages_ != 0U && messages_seen_ >= stop_after_messages_) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<isaac::JointStateCpuT> joint_state_in;
  holoscan::Output<isaac::JointStateCpuT> joint_state_out;

 private:
  std::size_t stop_after_messages_{};
  std::size_t messages_seen_{};
};

struct Options {
  std::filesystem::path config_path;
  bool connect_now{true};  // default on — this app is for running against Sim
  isaac::os::ops::UcxxConnectOptions connection;
  std::optional<std::size_t> message_count;
};

void parse_options(CLI::App& app, Options& options, int argc, char** argv) {
  // take_last(): the isaac_os launcher prepends its own --config-path when
  // wrapping the run command; the second one (the user's) is what we want.
  app.add_option("--config-path", options.config_path, "Application YAML configuration")
      ->required()
      ->take_last();
  app.add_flag("--connect,!--no-connect", options.connect_now,
               "Connect to Sim at startup (default). Use --no-connect for dry-run compile.");
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
                 "Terminate after receiving this many joint-state samples (default: run forever)")
      ->check(CLI::PositiveNumber);
  app.parse(argc, argv);
}

int run(const Options& options) {
  const auto config =
      holoscan::config_io::load(options.config_path, holoscan::config_io::Format::kYaml);
  const auto sim_hostname = config.get<std::string>("/sim_so101/hostname");
  const auto sim_port = config.get<int>("/sim_so101/port");
  const auto joint_state_buffer_size =
      get_or(config, "/sim_so101/joint_state_flatbuffer_buffer_size", std::size_t{16U * 1024U});
  const auto image_buffer_size =
      get_or(config, "/sim_so101/image_flatbuffer_buffer_size", std::size_t{4U * 1024U});
  const auto camera_buffer_size =
      get_or(config, "/sim_so101/camera_tensor_buffer_size", std::size_t{921600U});
  const auto enable_camera = get_or(config, "/sim_so101/enable_camera", true);
  SinusMotionParams sinus_params{
      .sin_frequency = config.get<double>("/sinus_motion_op/sin_frequency"),
      .sin_amplitude = config.get<double>("/sinus_motion_op/sin_amplitude"),
      .num_joints = config.get<int>("/sinus_motion_op/num_joints"),
      .default_position =
          get_or(config, "/sinus_motion_op/default_position", std::vector<double>{}),
      .kp = get_or(config, "/sinus_motion_op/kp", std::vector<double>{}),
      .kd = get_or(config, "/sinus_motion_op/kd", std::vector<double>{}),
  };

  // Prime the CUDA driver primary context on this thread before the UcxxEndpoint
  // spins up. Without this UCX's CUDA-transport probe logs
  // `sys_device = unknown` in the worker address, the peer unpacks that as
  // `bandwidth = 0.00`, and wireup is rejected with UCS_ERR_REJECTED.
  // Holoscan initializes CUDA later in startup. Fail fast if any step misfires so we do not
  // proceed to the endpoint constructor with a half-initialised driver state.
  //
  // The RAII guard restores the previous current context and releases the
  // primary-ctx retain in its destructor, so the endpoint/session teardown
  // finishes with the caller's driver state unchanged.
  class PrimaryContextGuard {
   public:
    PrimaryContextGuard() = default;
    PrimaryContextGuard(CUdevice device, CUcontext primary_ctx, CUcontext previous_ctx)
        : device_(device), primary_ctx_(primary_ctx), previous_ctx_(previous_ctx), engaged_(true) {}
    PrimaryContextGuard(const PrimaryContextGuard&) = delete;
    PrimaryContextGuard& operator=(const PrimaryContextGuard&) = delete;
    PrimaryContextGuard(PrimaryContextGuard&& other) noexcept
        : device_(other.device_),
          primary_ctx_(other.primary_ctx_),
          previous_ctx_(other.previous_ctx_),
          engaged_(other.engaged_) {
      other.engaged_ = false;
    }
    PrimaryContextGuard& operator=(PrimaryContextGuard&& other) noexcept {
      if (this != &other) {
        // Release ours first so we don't leak if reassigned.
        if (engaged_) {
          cuCtxSetCurrent(previous_ctx_);
          cuDevicePrimaryCtxRelease(device_);
        }
        device_ = other.device_;
        primary_ctx_ = other.primary_ctx_;
        previous_ctx_ = other.previous_ctx_;
        engaged_ = other.engaged_;
        other.engaged_ = false;
      }
      return *this;
    }
    ~PrimaryContextGuard() {
      if (!engaged_) return;
      cuCtxSetCurrent(previous_ctx_);
      cuDevicePrimaryCtxRelease(device_);
    }

   private:
    CUdevice device_{0};
    CUcontext primary_ctx_{nullptr};
    CUcontext previous_ctx_{nullptr};
    bool engaged_{false};
  };
  PrimaryContextGuard cuda_ctx_guard;

  if (options.connect_now) {
    auto log_cuda_error = [](const char* call, CUresult status) {
      const char* name = nullptr;
      const char* description = nullptr;
      cuGetErrorName(status, &name);
      cuGetErrorString(status, &description);
      HOLOSCAN_LOG_ERROR("{} failed: {} ({})", call, name != nullptr ? name : "unknown",
                         description != nullptr ? description : "no description");
    };
    if (CUresult status = cuInit(0); status != CUDA_SUCCESS) {
      log_cuda_error("cuInit", status);
      return 1;
    }
    constexpr CUdevice kCudaDevice = 0;
    CUcontext previous_ctx = nullptr;
    // Query the thread's current context before we override it. On the first
    // driver call this is nullptr (nothing set yet); non-null means we were
    // invoked from a thread that already had a driver context and we must
    // restore it before returning.
    (void)cuCtxGetCurrent(&previous_ctx);
    CUcontext ctx = nullptr;
    if (CUresult status = cuDevicePrimaryCtxRetain(&ctx, kCudaDevice); status != CUDA_SUCCESS) {
      log_cuda_error("cuDevicePrimaryCtxRetain", status);
      return 1;
    }
    if (CUresult status = cuCtxSetCurrent(ctx); status != CUDA_SUCCESS) {
      log_cuda_error("cuCtxSetCurrent", status);
      cuDevicePrimaryCtxRelease(kCudaDevice);
      return 1;
    }
    cuda_ctx_guard = PrimaryContextGuard(kCudaDevice, ctx, previous_ctx);
  }

  isaac::os::ops::UcxxEndpoint endpoint{sim_hostname, sim_port, false};

  holoscan::Graph graph{"sim-so101-sinus-motion"};
  // Connect after startup so immediate receive completions can notify the graph.
  const auto connector =
      graph.op<isaac::os::ops::UcxxConnectOp>("ucx_connect", endpoint, options.connection);

  const auto joint_state_rx = graph.op<isaac::os::ops::UcxxReceiverOp<isaac::JointStateCpuT>>(
      "joint_state_receiver", endpoint, kJointStateTag, joint_state_buffer_size);
  const auto joint_command_tx = graph.op<isaac::os::ops::UcxxSenderOp<isaac::JointCommandCpuT>>(
      "joint_command_sender", endpoint, kJointCommandTag);
  const auto sinus = graph.op<SinusMotionOp>("sinus_motion", std::move(sinus_params));

  // Camera path: when enable_camera is set in the YAML, receive isaac.Image
  // frames over UCX and show them in a Holoviz window. The scene must have the
  // UCXCameraHelper OGN wired in (and matching camera_tensor_buffer_size) for
  // frames to flow. Set enable_camera: false to run headless with no camera path.
  std::optional<decltype(graph.op<isaac::os::ops::UcxxReceiverOp<holoscan::schema::ImageT>>(
      "", endpoint, 0ULL, 0U, 0U))>
      image_rx;
  std::optional<decltype(graph.op<holoscan::visualization::HolovizOp>(
      "", holoscan::visualization::LayerStack{},
      holoscan::visualization::SinkDesc{holoscan::visualization::WindowSinkDesc{}}))>
      image_viz;
  if (enable_camera) {
    image_rx.emplace(graph.op<isaac::os::ops::UcxxReceiverOp<holoscan::schema::ImageT>>(
        "image_receiver", endpoint, kImageTag, image_buffer_size, camera_buffer_size));
    holoscan::visualization::LayerStack layers;
    layers.add_image_layer("camera", {.format = holoscan::visualization::ImageFormat::kR8G8B8Unorm,
                                      .max_width = 1920,
                                      .max_height = 1080});
    image_viz.emplace(graph.op<holoscan::visualization::HolovizOp>(
        "holoviz", std::move(layers),
        holoscan::visualization::SinkDesc{holoscan::visualization::WindowSinkDesc{
            .title = "SO-101 Simulator Camera", .width = 640, .height = 480}}));
  }

  // Optional stop-after-N observer.
  std::optional<decltype(graph.op<TickCounter>("", 0U))> counter;
  if (options.message_count.has_value()) {
    counter.emplace(graph.op<TickCounter>("tick_counter", *options.message_count));
  }

  // Joint control path:  joint_state_rx -> [counter ->] sinus -> joint_command_tx
  // When --message-count is set, counter is spliced inline between the
  // receiver and sinus (single consumer per output, so PARTITION_COVER holds).
  if (counter) {
    graph.add_flow(joint_state_rx->out_message, (*counter)->joint_state_in);
    graph.add_flow((*counter)->joint_state_out, sinus->joint_state_in);
  } else {
    graph.add_flow(joint_state_rx->out_message, sinus->joint_state_in);
  }
  graph.add_flow(sinus->joint_command_out, joint_command_tx->in_message);

  // Camera path. Drop-oldest + depth 1 keeps the viewport fresh and prevents
  // backpressure on the UCX receiver when the display can't keep up.
  if (image_rx) {
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    graph.add_flow((*image_rx)->out_message, (*image_viz)->image_layer_input(0), latest);
  }

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // Partitions: each UCX receiver's NotificationSource lives alone.
  // Connection startup runs separately from control.
  graph.partition("connection").add(connector);
  graph.partition("joint_state_rx").add(joint_state_rx);
  auto control_partition = graph.partition("control_and_tx").add(sinus).add(joint_command_tx);
  if (counter) {
    control_partition.add(*counter);
  }
  if (image_rx) {
    // Keep display work off the image receiver's execution thread.
    graph.partition("image_transport").add(*image_rx);
    graph.partition("holoviz").add(*image_viz);
  }

  holoscan::CompileOptions compile_options;
  if (image_rx) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = "image_receiver",
                                              .output_port = "out_message",
                                              .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, compile_options);
  if (!plan.ok()) {
    std::cerr << "Failed to compile so101_sinus_motion_sim graph:\n" << plan.json() << '\n';
    return 1;
  }

  if (!options.connect_now) {
    HOLOSCAN_LOG_INFO(
        "so101_sinus_motion_sim graph compiled successfully (dry run — "
        "pass --connect / omit --no-connect to attach to a live UCX peer).");
    return 0;
  }

  HOLOSCAN_LOG_INFO("Running so101_sinus_motion_sim against {}:{}.", sim_hostname, sim_port);
  holoscan::RunSession session = holoscan::run_async(plan);
  session.wait();

  // Only two exit paths are treated as success:
  // - kRequestedCompleted: reached the message_count target and requested
  //   completion internally.
  // - kStopped:            terminated on external signal (Ctrl+C, SIGTERM).
  // Everything else (kFailed, any other value) is a real failure -- UCX
  // dropout, operator exception, graph error -- and must not look like a
  // clean run to CI or a supervisor.
  const holoscan::RunTermination termination = session.termination();
  if (termination == holoscan::RunTermination::kRequestedCompleted ||
      termination == holoscan::RunTermination::kStopped) {
    return 0;
  }
  HOLOSCAN_LOG_ERROR("so101_sinus_motion_sim did not terminate cleanly.");
  const auto vertices = plan.graph().vertices();
  for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
    std::string_view vertex_name{"unknown"};
    for (const auto& vertex : vertices) {
      if (vertex.id == diagnostic.source.vertex) {
        vertex_name = vertex.name;
        break;
      }
    }
    HOLOSCAN_LOG_ERROR("{} at {} (service {}).", diagnostic.token, vertex_name, diagnostic.service);
  }
  return 1;
}

}  // namespace isaac::os::apps::so101_sinus_motion_sim

int main(int argc, char** argv) {
  isaac::os::apps::so101_sinus_motion_sim::Options options;
  CLI::App app{"SO-101 sinusoidal joint motion demo (sim UCX)"};
  try {
    isaac::os::apps::so101_sinus_motion_sim::parse_options(app, options, argc, argv);
    return isaac::os::apps::so101_sinus_motion_sim::run(options);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  } catch (const std::exception& error) {
    std::cerr << "so101_sinus_motion_sim threw: " << error.what() << '\n';
    return 1;
  }
}
