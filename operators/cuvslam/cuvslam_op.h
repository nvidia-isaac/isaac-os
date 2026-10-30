/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_CUVSLAM_CUVSLAM_OP_H
#define ISAAC_OS_OPERATORS_CUVSLAM_CUVSLAM_OP_H

#include <cuvslam/cuvslam_gpu.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "camera_info_cpu_schema_traits.h"
#include "common/pose_tree/pose_tree.hpp"
#include "odometry_cpu_schema_traits.h"

namespace isaac::os::ops {

/// Configuration of the visual odometry tracker.
struct CuvslamParams {
  /// Frame the estimated poses are expressed in.
  std::string odometry_frame{"odom"};
  /// Frame the estimated poses describe, which is the origin of the camera rig.
  std::string rig_frame{"rig"};
  /// One of `mono`, `rgbd`, `inertial`, or `multicamera`.
  std::string odometry_mode{"multicamera"};
  /// Whether to build the tracker on a background thread. Building it takes long enough to stall
  /// the capture pipeline, so frames are skipped while it runs unless this is disabled.
  bool async_setup{true};
  /// Seconds of estimates to summarize in each progress report. Zero or less silences the report.
  double report_period_s{2.0};
};

HOLOSCAN_PARAM_SCHEMA(CuvslamParams, holoscan::param_schema("com.nvidia.isaac.CuvslamParams", 2),
                      holoscan::field<&CuvslamParams::odometry_frame>("odometry_frame"),
                      holoscan::field<&CuvslamParams::rig_frame>("rig_frame"),
                      holoscan::field<&CuvslamParams::odometry_mode>("odometry_mode"),
                      holoscan::field<&CuvslamParams::async_setup>("async_setup"),
                      holoscan::field<&CuvslamParams::report_period_s>("report_period_s"));

/// Tracks the pose of a stereo camera rig with cuVSLAM.
///
/// The two frames of a stereo pair have to describe the same instant for the tracker to triangulate
/// from them, so the operator declares that requirement rather than implementing it: the runtime
/// activates it only once both cameras have a frame within the alignment window, and skips the
/// frames it cannot pair. The two calibration inputs are read as whatever the cameras last
/// published, because a calibration does not change between frames.
///
/// The tracker is built from the first calibration pair to arrive, not at start-up, because its
/// construction needs the frame extents that only a running camera reports.
///
/// It also reports its own progress, periodically rather than per estimate, because a line per
/// frame at camera rate says less about whether the pipeline is keeping up than a summary does.
/// Reporting from here rather than from a downstream operator is what lets the report describe the
/// frames the tracker actually used: how far apart they were captured is known only at the point
/// they are paired, and is not recoverable from the estimate, which mirrors nav_msgs/Odometry and
/// has no field for it.
class CuvslamOp final : public holoscan::Operator<CuvslamParams> {
 public:
  /// Most cameras one rig can have here. The tracker itself accepts more; this is the number of
  /// input declarations written out below, and raising it means writing out more.
  static constexpr std::size_t kMaxCameras = 8;

  /// @param alignment_window Largest capture-time difference between the cameras that still counts
  ///        as one set. It bounds the runtime's frame pairing only: the tracker is given each
  ///        frame's own capture time and documents that an almost-synchronized rig delivers those
  ///        within 1 ms, so widening this admits sets it was not specified for rather than
  ///        raising what it tolerates.
  /// @param pose_tree The tree this operator reads the cameras' placement from and publishes each
  ///        estimate into. Shared with every other operator on the same rig, and with the
  ///        application that created it, which is what keeps it alive across a rebuild.
  /// @param camera_count Cameras in the rig, from two up to kMaxCameras. Every one of them
  ///        contributes a frame to each estimate, so all must deliver for the tracker to run.
  CuvslamOp(std::chrono::nanoseconds alignment_window,
            std::shared_ptr<isaac::os::pose_tree::PoseTree> pose_tree, std::size_t camera_count)
      : alignment_window_(alignment_window),
        pose_tree_(std::move(pose_tree)),
        camera_count_(camera_count) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;
  void stop() override;

  /// @brief Return the frame input for camera @p index.
  /// @param index Zero-based camera index, below kMaxCameras.
  [[nodiscard]] const holoscan::Input<holoscan::schema::ImageT>& image(std::size_t index) const;
  [[nodiscard]] holoscan::Input<holoscan::schema::ImageT>& image(std::size_t index);

  /// @brief Return the calibration input for camera @p index.
  /// @param index Zero-based camera index, below kMaxCameras.
  [[nodiscard]] const holoscan::Input<isaac::CameraInfoCpuT>& camera_info(std::size_t index) const;
  [[nodiscard]] holoscan::Input<isaac::CameraInfoCpuT>& camera_info(std::size_t index);

  holoscan::Output<isaac::OdometryCpuT> odometry;

 private:
  using CameraInfoSet = std::vector<isaac::CameraInfoCpuT>;
  using FrameSet = std::vector<holoscan::Sample<holoscan::schema::ImageT>>;

  // An input declaration can be neither copied nor moved, so the ports cannot live in an array and
  // are written out to the maximum. Only the first camera_count_ of each are declared in setup(),
  // and only those are wired; the rest are inert.
  holoscan::Input<holoscan::schema::ImageT> image_0_;
  holoscan::Input<holoscan::schema::ImageT> image_1_;
  holoscan::Input<holoscan::schema::ImageT> image_2_;
  holoscan::Input<holoscan::schema::ImageT> image_3_;
  holoscan::Input<holoscan::schema::ImageT> image_4_;
  holoscan::Input<holoscan::schema::ImageT> image_5_;
  holoscan::Input<holoscan::schema::ImageT> image_6_;
  holoscan::Input<holoscan::schema::ImageT> image_7_;
  holoscan::Input<isaac::CameraInfoCpuT> camera_info_0_;
  holoscan::Input<isaac::CameraInfoCpuT> camera_info_1_;
  holoscan::Input<isaac::CameraInfoCpuT> camera_info_2_;
  holoscan::Input<isaac::CameraInfoCpuT> camera_info_3_;
  holoscan::Input<isaac::CameraInfoCpuT> camera_info_4_;
  holoscan::Input<isaac::CameraInfoCpuT> camera_info_5_;
  holoscan::Input<isaac::CameraInfoCpuT> camera_info_6_;
  holoscan::Input<isaac::CameraInfoCpuT> camera_info_7_;

  std::size_t camera_count_;

  /// Builds the cuVSLAM tracker for the given rig. Expensive enough to run on a background thread,
  /// which is why it takes its calibration by value rather than reading operator state.
  bool build_tracker(CameraInfoSet camera_infos);

  /// Takes one frame from every camera in the rig.
  [[nodiscard]] holoscan::expected<FrameSet, holoscan::Error> receive_frames();

  /// Takes every camera's calibration, or nothing if one has yet to publish. Nothing is an
  /// ordinary answer: the cameras publish calibration alongside frames, so the first sets arrive
  /// before it does.
  [[nodiscard]] holoscan::expected<std::optional<CameraInfoSet>, holoscan::Error>
  receive_calibration();

  /// Brings the tracker up if it is not already, and says whether it can track this set. False
  /// covers waiting for a calibration and waiting for a build, which are the same to the caller.
  [[nodiscard]] holoscan::expected<bool, holoscan::Error> ensure_tracker();
  /// Starts the build, or performs it, once every calibration is in hand.
  [[nodiscard]] holoscan::expected<bool, holoscan::Error> start_tracker_build();

  /// Describes one frame to the tracker, rejecting it if it cannot be read.
  [[nodiscard]] static holoscan::expected<cuvslam::Image, holoscan::Error> describe_frame(
      const holoscan::schema::ImageT& frame, std::size_t index, std::int64_t set_timestamp_ns);
  /// Describes a whole set.
  [[nodiscard]] static holoscan::expected<std::vector<cuvslam::Image>, holoscan::Error>
  describe_frames(const FrameSet& frames, std::int64_t set_timestamp_ns);

  /// Finishes the work behind the frames, which the tracker reads on streams this graph does not
  /// own.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> finish_frames(
      holoscan::ExecutionContext& context) const;

  /// Hands one set to the tracker, or nothing if it lost track. Losing track is ordinary: the rig
  /// may have turned to face a blank wall, and the next set may well be tracked.
  [[nodiscard]] holoscan::expected<std::optional<cuvslam::PoseEstimate>, holoscan::Error> track(
      holoscan::ExecutionContext& context, const FrameSet& frames, std::int64_t set_timestamp_ns);

  /// Builds the message published for one estimate.
  [[nodiscard]] isaac::OdometryCpuT describe_estimate(const cuvslam::PoseEstimate& estimate) const;
  /// Records where the rig was, so anything in the process can ask without following the port.
  void record_in_pose_tree(const cuvslam::PoseEstimate& estimate);
  /// Publishes, tolerating an output nothing is connected to.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> publish(
      const isaac::OdometryCpuT& message, const holoscan::Sample<holoscan::schema::ImageT>& source);
  /// Records one estimate everywhere it belongs: the tree, the report, and the port.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> publish_estimate(
      const cuvslam::PoseEstimate& estimate,
      const holoscan::Sample<holoscan::schema::ImageT>& source);

  /// Report a milestone once. Each is otherwise silent, and silence cannot be told apart from a
  /// pipeline that never got that far.
  void report_ready();
  void report_waiting_for_calibration();

  /// Records one estimate against the current reporting period, and reports once the period is up.
  void report_progress(std::int64_t time_ns, const isaac::OdometryCpuT& estimate);
  /// Starts a new reporting period at the given estimate time.
  void begin_report_period(std::int64_t time_ns);

  std::chrono::nanoseconds alignment_window_;
  std::shared_ptr<isaac::os::pose_tree::PoseTree> pose_tree_;
  /// Odometry mode resolved once during start(). The background build reads it from a thread the
  /// runtime does not quiesce, where the parameter aggregate itself is not safe to read.
  cuvslam::Odometry::OdometryMode odometry_mode_{cuvslam::Odometry::OdometryMode::Multicamera};
  /// Read out of the parameters during start(), because the background build runs on a thread
  /// where the parameter aggregate is not safe to read.
  std::string odometry_frame_;
  std::string rig_frame_;
  /// Resolved once during start(). Every estimate is recorded against these, so looking the two
  /// names up again on each frame would be work with a known answer.
  isaac::os::pose_tree::PoseTree::frame_t odometry_frame_id_{};
  isaac::os::pose_tree::PoseTree::frame_t rig_frame_id_{};
  std::unique_ptr<cuvslam::Odometry> tracker_;
  /// Valid while a background tracker build is in flight and its result has not been collected.
  std::future<bool> tracker_build_;
  /// Report each milestone once. Startup is otherwise silent, and silence cannot be told apart
  /// from frames that never paired.
  bool reported_first_set_{};
  bool reported_waiting_for_calibration_{};
  bool reported_ready_{};
  bool reported_first_pose_{};
  bool reported_unconsumed_odometry_{};
  /// Progress reporting. The period is measured on the estimates' own timeline rather than on a
  /// wall clock, so a report describes a span of tracked motion rather than a span of real time in
  /// which the tracker may have produced nothing.
  bool period_started_{};
  std::int64_t period_start_ns_{};
  std::int64_t previous_estimate_ns_{};
  int estimate_count_{};
  std::int64_t shortest_estimate_gap_ns_{};
  std::int64_t longest_estimate_gap_ns_{};
  /// Widest spread within any one frame set handed to the tracker this period. Accumulated for
  /// every set, including sets the tracker went on to lose track on, because the spread is a fact
  /// about how the cameras are synchronized rather than about whether tracking succeeded.
  std::int64_t worst_capture_spread_ns_{};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_CUVSLAM_CUVSLAM_OP_H
