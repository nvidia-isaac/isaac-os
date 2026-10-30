/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <CLI/CLI.hpp>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sipl_capture_op/sipl_capture_op.hpp>
#include <sipl_capture_op/sipl_capture_service.hpp>
#include <sipl_capture_op/sipl_frame_metadata_schema_traits.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/config/document.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "common/pose_tree/pose_tree.hpp"
#include "operators/camera_info/camera_info_op.h"
#include "operators/cuvslam/cuvslam_op.h"
#include "operators/image_conversion/nv12_to_rgb_op.h"

namespace camera = holoscan::camera;

namespace isaac::os {
namespace {

/// Visual odometry needs at least a pair to triangulate from.
constexpr std::size_t kMinimumCameras = 2U;

/// Redraw period of the window, at the 60 Hz a display is normally driven at.
constexpr auto kDisplayPeriod =
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>{1.0 / 60.0});
/// How long a captured frame stays eligible to be drawn. The cameras produce at 30 Hz against a
/// 60 Hz window, so a frame has to outlive more than one redraw or half of them would show
/// nothing.
constexpr auto kMaxDisplayedFrameAge = std::chrono::milliseconds{100};

/// The spread cuVSLAM specifies for an "almost synchronized" rig, and therefore the widest pairing
/// window its results are defined for. It is the default; anything wider is a diagnostic aid.
constexpr double kSupportedAlignmentWindowMs = 1.0;

/// Accepts and discards the per-frame sensor telemetry each camera publishes.
///
/// Visual odometry needs none of it: exposure, gain, white balance and the image signal processor
/// statistics exist to drive an auto-exposure loop the camera already runs for itself. The camera
/// publishes them on every frame regardless, and reports a warning whenever nothing is listening,
/// which at frame rate buries every other message in the log.
///
/// Accepting the report copies it, and in color mode it carries a large statistics block, so this
/// costs a few tens of megabytes per second. That is the price of a readable log until the camera
/// stops reporting an absent listener as a fault. Anything that starts using this telemetry, such
/// as measuring the capture instants of the two sensors against each other, should replace this
/// rather than run beside it.
class SensorTelemetrySinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(sensor_data, "in_sensor_data").queue_depth(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{sensor_data});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& /*context*/) override {
    auto sample = sensor_data.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    return {};
  }

  holoscan::Input<camera::SIPLFrameMetadataT> sensor_data;
};

struct Options {
  std::filesystem::path application_config_path;
  std::string camera_config;
  std::filesystem::path json_config;
  std::int32_t cuda_device{0};
  double alignment_window_ms{kSupportedAlignmentWindowMs};
  std::uint32_t window_width{1280};
  std::uint32_t window_height{540};
  bool headless{};
  bool validate_only{};
};

void parse_options(CLI::App& app, Options& options, int argc, char** argv) {
  app.add_option("--application-config-path", options.application_config_path,
                 "Calibration and tracker YAML configuration")
      ->required();
  app.add_option("--camera-config", options.camera_config,
                 "Camera configuration to use from the platform description. When omitted, the "
                 "first configuration that names a platform is selected.");
  app.add_option("--json-config", options.json_config,
                 "Platform description listing the camera modules")
      ->check(CLI::ExistingFile);
  app.add_option("--cuda-device", options.cuda_device,
                 "GPU that receives the captured frames. Only device 0 is supported")
      ->check(CLI::NonNegativeNumber);
  app.add_option("--alignment-window-ms", options.alignment_window_ms,
                 "Largest difference in capture time that still pairs two frames as one stereo "
                 "pair. Above 1 ms is diagnostic only")
      ->check(CLI::PositiveNumber);
  app.add_option("--window-width", options.window_width, "Width of the camera window")
      ->check(CLI::PositiveNumber);
  app.add_option("--window-height", options.window_height, "Height of the camera window")
      ->check(CLI::PositiveNumber);
  app.add_flag("--headless", options.headless,
               "Track without opening a window, for a machine with no display attached");
  app.add_flag("--validate", options.validate_only,
               "Check that the pipeline can be built and then exit without capturing");
  app.parse(argc, argv);

  if (!std::filesystem::is_regular_file(options.application_config_path)) {
    throw CLI::ValidationError(
        "--application-config-path",
        "Camera calibration file '" + options.application_config_path.string() +
            "' is missing or is not a regular file. Run 'sh applications/cuvslam_vb1940/"
            "generate_vb1940_config.sh' from the Isaac OS checkout to generate "
            "applications/cuvslam_vb1940/config/config.yaml.");
  }

  // The device reaches three places that have to agree: the tensor output bindings below, the
  // capture service's NvSci buffers, and whatever thread builds the tracker. The first two are
  // given it; the third is a thread this application does not own, and a device set on the main
  // thread does not carry to it. Rather than advertise a setting that would leave the capture path
  // and the tracker on different GPUs with no error, only the one value that is known to hold
  // everywhere is accepted. Lifting this means verifying all three on a machine with more than one
  // GPU, not deleting the check.
  if (options.cuda_device != 0) {
    throw CLI::ValidationError(
        "--cuda-device",
        "only device 0 is supported. The capture service and the tracker's own initialization have "
        "not been verified against another device.");
  }

  // Wider than the tracker's contract is allowed, because it is how an unsynchronized rig is
  // brought up and measured, but it is not a tuning knob: two frames that far apart were taken
  // from different places, and the tracker triangulates them as though they were not. The error
  // lands in the pose rather than in an exit code, which is why this says so out loud.
  if (options.alignment_window_ms > kSupportedAlignmentWindowMs) {
    HOLOSCAN_LOG_WARN(
        "Pairing frames captured up to {} ms apart. cuVSLAM specifies {} ms for an almost "
        "synchronized rig, so poses produced with this window are diagnostic and unvalidated. "
        "Synchronize the sensors in hardware rather than widening this further.",
        options.alignment_window_ms, kSupportedAlignmentWindowMs);
  }
}

/// Place one camera in the rig, from the calibration the configuration carries.
///
/// The tree is what relates the rig to its cameras from here on, so this is the only point where
/// those numbers are read. They do not change while the run lasts, which is why they are written
/// once here rather than republished with every frame.
void place_camera_in_rig(isaac::os::pose_tree::PoseTree& tree, const std::string& rig_frame,
                         const std::string& camera_frame, const std::vector<double>& translation,
                         const std::vector<double>& rotation_wxyz) {
  if (translation.size() != 3U || rotation_wxyz.size() != 4U) {
    throw std::runtime_error(
        "The calibration of camera '" + camera_frame +
        "' must give its position in the rig as three values and its orientation as four, "
        "ordered w, x, y, z.");
  }
  // A frame has to exist before it can be related to another: the tree looks names up, it does not
  // invent them, so naming a frame is a separate step from relating it.
  const auto rig = tree.find_or_create_frame(rig_frame);
  const auto camera = tree.find_or_create_frame(camera_frame);
  if (!rig || !camera) {
    throw std::runtime_error("Unable to name the '" + rig_frame + "' and '" + camera_frame +
                             "' frames.");
  }
  if (!tree.create_edges(rig.value(), camera.value())) {
    throw std::runtime_error("Unable to relate camera '" + camera_frame + "' to the rig.");
  }
  const isaac::os::pose_tree::Pose3d rig_t_camera{
      isaac::os::pose_tree::SO3d::from_quaternion(isaac::os::pose_tree::Quaternion<double>(
          rotation_wxyz[0], rotation_wxyz[1], rotation_wxyz[2], rotation_wxyz[3])),
      isaac::os::pose_tree::Vector3<double>(translation[0], translation[1], translation[2])};
  // Time zero: the placement is fixed, so it holds for every instant the run can ask about.
  if (!tree.set(rig.value(), camera.value(), 0.0, rig_t_camera)) {
    throw std::runtime_error("Unable to place camera '" + camera_frame + "' in the rig.");
  }
}

int run(const Options& options, holoscan::ManagedShutdownSignals& signals) {
  auto configuration = holoscan::config_io::load(options.application_config_path,
                                                 holoscan::config_io::Format::kYaml);

  // Bringing the cameras up here rather than at start-up is what makes the frame extents known
  // while the graph is still being described, and the frame buffers are sized from them.
  // The trailing arguments are defaults restated so the device, which is the last of them, can be
  // given at all. It selects the GPU the NvSci capture buffers are allocated on, and left out it
  // would silently be device 0 while the tensor outputs below followed --cuda-device.
  auto capture_service = std::make_shared<camera::SIPLCaptureService>(
      options.camera_config, options.json_config.string(), /*raw_output=*/false,
      /*capture_queue_depth=*/4U, /*nito_base_path=*/"/var/nvidia/nvcam/settings/sipl",
      /*timeout_us=*/1'000'000U, options.cuda_device);
  const auto& cameras = capture_service->get_camera_info();
  const std::size_t camera_count = cameras.size();
  if (camera_count < kMinimumCameras || camera_count > ops::CuvslamOp::kMaxCameras) {
    HOLOSCAN_LOG_ERROR(
        "The selected camera configuration describes {} camera(s). Visual odometry needs between "
        "{} and {}. Check the platform description at {}.",
        camera_count, kMinimumCameras, ops::CuvslamOp::kMaxCameras, options.json_config.string());
    return 1;
  }
  HOLOSCAN_LOG_INFO("Tracking a rig of {} cameras.", camera_count);

  const auto alignment_window = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double, std::milli>(options.alignment_window_ms));

  // One tree, owned here for the run. The graph copies the handle into every operator it builds,
  // so each holds a share of this one and it outlives all of them.
  const auto pose_tree = std::make_shared<isaac::os::pose_tree::PoseTree>();
  if (!pose_tree->init()) {
    HOLOSCAN_LOG_ERROR("Unable to prepare the tree of coordinate frames.");
    return 1;
  }

  const auto rig_frame = configuration.get<std::string>("/cuvslam/rig_frame");

  holoscan::Graph graph{"cuvslam-vb1940"};

  auto tracker = graph.op<ops::CuvslamOp>("cuvslam", alignment_window, pose_tree, camera_count);
  auto tracker_params = holoscan::params::decode<ops::CuvslamParams>(configuration, "/cuvslam");
  // Reported on every run rather than only where a default was taken, because the two are
  // indistinguishable from here: decode fills its defaults in silently, and each of them is valid,
  // so a tracker publishing into frames the rest of the graph never reads looks exactly like one
  // that is working. Naming the frames it settled on is what tells the two apart.
  HOLOSCAN_LOG_INFO("Visual odometry will track '{}' in '{}', in {} mode.",
                    tracker_params->rig_frame, tracker_params->odometry_frame,
                    tracker_params->odometry_mode);
  tracker.set_params(std::move(tracker_params));

  // Only the newest frame matters to a tracker that has fallen behind, so a full queue drops the
  // stale frame rather than stalling the camera.
  const holoscan::ConnectionOptions newest_frame{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };

  holoscan::CompileOptions compile_options;
  std::vector<holoscan::OperatorHandle<ops::Nv12ToRgbOp>> display_feeds;

  // Everything one camera needs between the sensor and the tracker. Each camera captures and
  // converts on its own thread so that one sensor cannot delay the other.
  const auto add_camera = [&](std::size_t index) {
    const auto& info = cameras[index];
    struct Stage {
      holoscan::OperatorHandle<camera::SIPLCaptureOp> capture;
      holoscan::OperatorHandle<ops::Nv12ToRgbOp> color;
      holoscan::OperatorHandle<ops::CameraInfoOp> calibration;
    };

    const std::string position = std::to_string(index);
    const std::string capture_name = "camera_" + position;
    const std::string color_name = "color_" + position;
    const std::string calibration_name = "camera_info_" + position;
    const std::string telemetry_name = "sensor_telemetry_" + position;

    place_camera_in_rig(
        *pose_tree, rig_frame, configuration.get<std::string>("/" + calibration_name + "/frame_id"),
        configuration.get<std::vector<double>>("/camera_placement_" + position + "/translation"),
        configuration.get<std::vector<double>>("/camera_placement_" + position + "/rotation_wxyz"));

    auto capture = graph.op<camera::SIPLCaptureOp>(capture_name, capture_service,
                                                   static_cast<std::uint32_t>(index),
                                                   holoscan::MemoryKind::kCudaDevice);
    auto color = graph.op<ops::Nv12ToRgbOp>(color_name, info.width, info.height);
    auto calibration = graph.op<ops::CameraInfoOp>(calibration_name);
    auto telemetry = graph.op<SensorTelemetrySinkOp>(telemetry_name);

    graph.add_flow(capture->frame, color->image, newest_frame);
    graph.add_flow(capture->frame, calibration->image, newest_frame);
    graph.add_flow(capture->sensor_data, telemetry->sensor_data, newest_frame);
    // Discarding the telemetry stays on this camera's thread so it cannot disturb tracking.
    auto partition = graph.partition(capture_name);
    partition.add(capture).add(color).add(calibration).add(telemetry);

    if (!options.headless) {
      display_feeds.push_back(color);
    }

    // Every output that carries a tensor has to name the GPU it lands on. A missing binding is
    // rejected rather than defaulting to device zero, and that applies to the capture operator as
    // much as to the conversion this application owns.
    compile_options.deployment
        .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
            .operator_path = capture_name,
            .output_port = "frame",
            .device = holoscan::DeviceId{options.cuda_device},
        })
        .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
            .operator_path = color_name,
            .output_port = "out_image",
            .device = holoscan::DeviceId{options.cuda_device},
        });
    auto calibration_params =
        holoscan::params::decode<ops::CameraInfoParams>(configuration, "/" + calibration_name);
    HOLOSCAN_LOG_INFO("Camera {} is calibrated as frame '{}', {} distortion.", index,
                      calibration_params->frame_id, calibration_params->distortion_model);
    calibration.set_params(std::move(calibration_params));
    return Stage{capture, color, calibration};
  };

  for (std::size_t index = 0; index < camera_count; ++index) {
    const auto stage = add_camera(index);
    graph.add_flow(stage.color->rgb_image, tracker->image(index), newest_frame);
    graph.add_flow(stage.calibration->camera_info, tracker->camera_info(index), newest_frame);
  }
  graph.partition("tracking").add(tracker);

  if (!display_feeds.empty()) {
    // One layer per camera, laid out side by side across the window. The visualizer reserves a
    // texture for the largest frame each layer will ever carry, so the layers are declared from
    // the extents the cameras reported.
    holoscan::visualization::LayerStack layers;
    const float share = 1.0F / static_cast<float>(camera_count);
    for (std::size_t index = 0; index < camera_count; ++index) {
      layers.add_image_layer("camera_" + std::to_string(index),
                             {
                                 .format = holoscan::visualization::ImageFormat::kR8G8B8Unorm,
                                 .max_width = cameras[index].width,
                                 .max_height = cameras[index].height,
                                 // Each camera takes an equal, adjacent share of the window.
                                 .view =
                                     holoscan::visualization::ViewTransform{
                                         .offset_x = static_cast<float>(index) * share,
                                         .scale_x = share,
                                     },
                             });
    }
    // The cameras produce at half the rate the window is redrawn at, so a frame has to stay
    // eligible for longer than one redraw or it would be missing from most of them.
    auto visualizer = graph.op<holoscan::visualization::HolovizOp>(
        "holoviz", std::move(layers),
        holoscan::visualization::SinkDesc{holoscan::visualization::WindowSinkDesc{
            .title = "cuVSLAM VB1940",
            .width = options.window_width,
            .height = options.window_height,
        }},
        holoscan::visualization::ClockTrigger{.period = kDisplayPeriod}, kMaxDisplayedFrameAge);
    for (std::size_t index = 0; index < display_feeds.size(); ++index) {
      graph.add_flow(display_feeds[index]->rgb_image, visualizer->image_layer_input(index),
                     newest_frame);
    }
    graph.partition("display").add(visualizer);
  }

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("The visual odometry pipeline cannot run as configured.");
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (options.validate_only) {
    HOLOSCAN_LOG_INFO("The visual odometry pipeline is valid. Exiting without capturing.");
    return 0;
  }

  HOLOSCAN_LOG_INFO("Tracking the camera rig. Press Ctrl-C to stop.");
  holoscan::RunSession session = holoscan::run_async(plan, signals);
  session.wait();

  const holoscan::RunTermination termination = session.termination();
  if (termination == holoscan::RunTermination::kFailed) {
    HOLOSCAN_LOG_ERROR("Visual odometry stopped because the pipeline failed.");
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      HOLOSCAN_LOG_ERROR("{}", holoscan::format_diagnostic(diagnostic, plan.graph()));
    }
    return 1;
  }
  if (termination == holoscan::RunTermination::kStopped) {
    HOLOSCAN_LOG_INFO("Visual odometry stopped on request. The cameras have been shut down.");
  }
  return 0;
}

}  // namespace
}  // namespace isaac::os

int main(int argc, char** argv) {
  // Take ownership of the shutdown signals before anything else runs. Bringing the cameras up while
  // the graph is described makes this process multithreaded, and the ownership this needs can only
  // be established while a single thread is live. Owning it is what turns Ctrl-C into a cooperative
  // stop that shuts the sensors down instead of leaving them streaming.
  auto created = holoscan::ManagedShutdownSignals::create();
  if (!created) {
    std::cerr << "Unable to take ownership of the shutdown signals: "
              << holoscan::runtime_error_message(created.error()) << '\n';
    return 1;
  }
  auto signals = std::move(created).value();

  // CLI11 keeps references to option values, so they must outlive the parser.
  isaac::os::Options options;
  CLI::App app{"Track a VB1940 stereo camera rig with cuVSLAM"};
  try {
    isaac::os::parse_options(app, options, argc, argv);
    return isaac::os::run(options, signals);
  } catch (const CLI::ParseError& error) {
    if (error.get_exit_code() != 0) {
      HOLOSCAN_LOG_ERROR("{}", error.what());
      return error.get_exit_code();
    }
    return app.exit(error);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
