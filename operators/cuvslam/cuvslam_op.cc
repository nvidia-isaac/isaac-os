/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/cuvslam/cuvslam_op.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <cuda_runtime.h>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "common/pose_tree/math/pose3.hpp"
#include "common/pose_tree/math/so3.hpp"
#include "operators/cuvslam/calibration_mapping.h"

namespace isaac::os::ops {

namespace {

constexpr double kNanosecondsToSeconds = 1e-9;
constexpr double kNanosecondsToMilliseconds = 1e-6;

cuvslam::Distortion::Model to_cuvslam_model(const std::string& model) {
  if (model == "plumb_bob") {
    return cuvslam::Distortion::Model::Brown;
  }
  if (model == "rational_polynomial") {
    return cuvslam::Distortion::Model::Polynomial;
  }
  if (model == "equidistant") {
    return cuvslam::Distortion::Model::Fisheye;
  }
  // An empty name is how sensor_msgs/CameraInfo says a camera has no distortion. Anything else is
  // a model this tracker has no equivalent for, and undistorting with the wrong one is worse than
  // not undistorting at all, so both fall through to the pinhole camera.
  if (!model.empty()) {
    HOLOSCAN_LOG_WARN("Ignoring the unsupported distortion model '{}'.", model);
  }
  return cuvslam::Distortion::Model::Pinhole;
}

cuvslam::Odometry::OdometryMode to_cuvslam_mode(const std::string& mode) {
  if (mode == "mono") {
    return cuvslam::Odometry::OdometryMode::Mono;
  }
  if (mode == "rgbd") {
    return cuvslam::Odometry::OdometryMode::RGBD;
  }
  if (mode == "inertial") {
    return cuvslam::Odometry::OdometryMode::Inertial;
  }
  return cuvslam::Odometry::OdometryMode::Multicamera;
}

holoscan::Error failure(const char* message) {
  return holoscan::Error{holoscan::ErrorCode::kFailure, message};
}

}  // namespace

const holoscan::Input<holoscan::schema::ImageT>& CuvslamOp::image(std::size_t index) const {
  switch (index) {
    case 0:
      return image_0_;
    case 1:
      return image_1_;
    case 2:
      return image_2_;
    case 3:
      return image_3_;
    case 4:
      return image_4_;
    case 5:
      return image_5_;
    case 6:
      return image_6_;
    case 7:
      return image_7_;
    default:
      std::terminate();  // unreachable: callers check against kMaxCameras
  }
}

holoscan::Input<holoscan::schema::ImageT>& CuvslamOp::image(std::size_t index) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast) — safe, `this` is non-const here.
  return const_cast<holoscan::Input<holoscan::schema::ImageT>&>(
      static_cast<const CuvslamOp*>(this)->image(index));
}

const holoscan::Input<isaac::CameraInfoCpuT>& CuvslamOp::camera_info(std::size_t index) const {
  switch (index) {
    case 0:
      return camera_info_0_;
    case 1:
      return camera_info_1_;
    case 2:
      return camera_info_2_;
    case 3:
      return camera_info_3_;
    case 4:
      return camera_info_4_;
    case 5:
      return camera_info_5_;
    case 6:
      return camera_info_6_;
    case 7:
      return camera_info_7_;
    default:
      std::terminate();  // unreachable: callers check against kMaxCameras
  }
}

holoscan::Input<isaac::CameraInfoCpuT>& CuvslamOp::camera_info(std::size_t index) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast) — safe, `this` is non-const here.
  return const_cast<holoscan::Input<isaac::CameraInfoCpuT>&>(
      static_cast<const CuvslamOp*>(this)->camera_info(index));
}

void CuvslamOp::setup(holoscan::OperatorSpec& spec) {
  for (std::size_t index = 0; index < camera_count_; ++index) {
    const std::string suffix = std::to_string(index);
    spec.input(image(index), "in_image_" + suffix).queue_depth(1U);
    spec.input(camera_info(index), "in_camera_info_" + suffix).queue_depth(1U);
  }
  spec.output(odometry, "out_odometry");
}

holoscan::Contract CuvslamOp::contract() const {
  holoscan::Contract contract;
  // Pair the two cameras on capture time. This replaces the frame-by-frame caching and discarding
  // an earlier pipeline generation did by hand: unpairable frames never reach compute().
  std::vector<holoscan::PortRef> aligned;
  aligned.reserve(camera_count_);
  for (std::size_t index = 0; index < camera_count_; ++index) {
    aligned.emplace_back(image(index));
  }
  contract.trigger(holoscan::OnAligned{
      .ports = holoscan::PortList{std::move(aligned)},
      .window = alignment_window_,
  });
  for (std::size_t index = 0; index < camera_count_; ++index) {
    contract.input(image(index), holoscan::Exact{});
    // A calibration describes the camera, not the frame, so the newest one of any age is correct.
    // When none has arrived yet the operator takes its degraded path and skips the set.
    contract.input(camera_info(index),
                   holoscan::LatestWithin{.max_age = std::chrono::nanoseconds::max()});
    contract.on_missing(camera_info(index), holoscan::SafeDefault{});
  }
  return contract;
}

void CuvslamOp::start() {
  const auto& config = params();
  const std::string& mode = config.odometry_mode;
  if (mode != "mono" && mode != "rgbd" && mode != "inertial" && mode != "multicamera") {
    throw std::runtime_error("The requested odometry mode '" + mode +
                             "' is unsupported. Choose mono, rgbd, inertial, or multicamera.");
  }
  if (config.odometry_frame.empty() || config.rig_frame.empty()) {
    throw std::runtime_error("Visual odometry requires both an odometry frame and a rig frame.");
  }
  odometry_mode_ = to_cuvslam_mode(mode);
  odometry_frame_ = config.odometry_frame;
  rig_frame_ = config.rig_frame;
  // Both frames have to exist before they can be related: the tree looks names up rather than
  // inventing them. The rig may already have been named while the cameras were placed in it.
  if (!pose_tree_) {
    throw std::runtime_error(
        "Visual odometry was not given the tree of coordinate frames to publish into.");
  }
  isaac::os::pose_tree::PoseTree& tree = *pose_tree_;
  const auto odometry_frame = tree.find_or_create_frame(odometry_frame_);
  const auto rig_frame = tree.find_or_create_frame(rig_frame_);
  if (!odometry_frame || !rig_frame) {
    throw std::runtime_error("Unable to name the '" + odometry_frame_ + "' and '" + rig_frame_ +
                             "' frames.");
  }
  odometry_frame_id_ = odometry_frame.value();
  rig_frame_id_ = rig_frame.value();
  // Declare the edge this operator owns before any frame arrives, so a reader that asks early gets
  // an empty edge rather than a missing one.
  if (!tree.create_edges(odometry_frame_id_, rig_frame_id_)) {
    throw std::runtime_error("Unable to relate the '" + rig_frame_ + "' frame to '" +
                             odometry_frame_ + "'.");
  }
  cuvslam::WarmUpGPU();
}

bool CuvslamOp::build_tracker(CameraInfoSet camera_infos) {
  cuvslam::Rig rig;
  rig.cameras.reserve(camera_infos.size());
  for (const isaac::CameraInfoCpuT& info : camera_infos) {
    if (!info.header) {
      HOLOSCAN_LOG_ERROR("A camera published a calibration that does not name its frame.");
      return false;
    }
    const std::string& frame_id = info.header->frame_id;
    // The intrinsic matrix is a fixed 3x3 on the wire, so it cannot arrive short.
    const auto* const k = info.k.m();
    // Where this camera sits in the rig comes from the tree, which the application seeded from the
    // calibration before the run began.
    const auto rig_t_camera = pose_tree_->get(rig_frame_, frame_id);
    if (!rig_t_camera) {
      HOLOSCAN_LOG_ERROR("Unable to place camera '{}' in the '{}' frame.", frame_id, rig_frame_);
      return false;
    }
    // cuVSLAM orders a quaternion x, y, z, w.
    const auto rotation = rig_t_camera->rotation.quaternion().cast<float>();
    const auto translation = rig_t_camera->translation.cast<float>();
    cuvslam::Pose rig_from_camera{
        {rotation.x(), rotation.y(), rotation.z(), rotation.w()},
        {translation.x(), translation.y(), translation.z()},
    };

    cuvslam::Distortion distortion;
    distortion.model = to_cuvslam_model(info.distortion_model);
    if (distortion.model == cuvslam::Distortion::Model::Brown) {
      const auto brown = brown_from_plumb_bob(info.d);
      if (!brown) {
        HOLOSCAN_LOG_ERROR(
            "Camera '{}' declares plumb_bob distortion with {} coefficients. That model is "
            "defined with exactly five.",
            frame_id, info.d.size());
        return false;
      }
      distortion.parameters.assign(brown->begin(), brown->end());
    } else {
      distortion.parameters = narrow(info.d);
    }

    cuvslam::Camera camera{
        {static_cast<int32_t>(info.width), static_cast<int32_t>(info.height)},
        {static_cast<float>(k->Get(2)), static_cast<float>(k->Get(5))},
        {static_cast<float>(k->Get(0)), static_cast<float>(k->Get(4))},
        rig_from_camera,
        std::move(distortion),
        0,
        0,
        0,
        0,
    };
    rig.cameras.emplace_back(std::move(camera));
  }

  cuvslam::Odometry::Config config;
  config.odometry_mode = odometry_mode_;
  config.async_sba = false;
  // Left at its default. It bounds the gap between one set of frames and the next, above which the
  // tracker warns that it has been starved, and says nothing about how far apart the frames within
  // a set may be. The alignment window is that second quantity, so setting this from it would
  // silence a real warning about frame delivery while changing nothing about stereo tolerance.
  tracker_ = std::make_unique<cuvslam::Odometry>(rig, config);
  return true;
}

namespace {

/// The instant a set of frames is taken to describe, which is the first camera's capture time.
std::int64_t set_timestamp_of(
    const std::vector<holoscan::Sample<holoscan::schema::ImageT>>& frames) {
  return frames[0].data.header ? frames[0].data.header->capture_timestamp_ns : 0;
}

/// The widest gap between any two frames in a set.
///
/// Measured here because this is the set the tracker used, and nothing downstream can reconstruct
/// which frames those were.
std::int64_t capture_spread_of(
    const std::vector<holoscan::Sample<holoscan::schema::ImageT>>& frames,
    std::int64_t set_timestamp_ns) {
  std::int64_t earliest_ns = set_timestamp_ns;
  std::int64_t latest_ns = set_timestamp_ns;
  for (const auto& frame : frames) {
    const std::int64_t captured_ns =
        frame.data.header ? frame.data.header->capture_timestamp_ns : 0;
    earliest_ns = std::min(earliest_ns, captured_ns);
    latest_ns = std::max(latest_ns, captured_ns);
  }
  return latest_ns - earliest_ns;
}

}  // namespace

void CuvslamOp::report_ready() {
  if (!reported_ready_) {
    reported_ready_ = true;
    HOLOSCAN_LOG_INFO("Visual odometry is tracking the camera rig.");
  }
}

void CuvslamOp::report_waiting_for_calibration() {
  if (!reported_waiting_for_calibration_) {
    reported_waiting_for_calibration_ = true;
    HOLOSCAN_LOG_INFO(
        "Waiting for every camera to publish its calibration before starting visual odometry.");
  }
}

holoscan::expected<CuvslamOp::FrameSet, holoscan::Error> CuvslamOp::receive_frames() {
  FrameSet frames;
  frames.reserve(camera_count_);
  for (std::size_t index = 0; index < camera_count_; ++index) {
    auto frame = image(index).receive();
    if (!frame) {
      return holoscan::make_unexpected(std::move(frame).error());
    }
    frames.push_back(std::move(*frame));
  }
  if (!reported_first_set_) {
    reported_first_set_ = true;
    HOLOSCAN_LOG_INFO("All {} cameras delivered a frame within {} ms of each other.", camera_count_,
                      std::chrono::duration<double, std::milli>(alignment_window_).count());
  }
  return frames;
}

holoscan::expected<std::optional<CuvslamOp::CameraInfoSet>, holoscan::Error>
CuvslamOp::receive_calibration() {
  CameraInfoSet camera_infos;
  camera_infos.reserve(camera_count_);
  for (std::size_t index = 0; index < camera_count_; ++index) {
    if (camera_info(index).is_missing()) {
      report_waiting_for_calibration();
      return std::nullopt;
    }
    auto info = camera_info(index).receive_data();
    if (!info) {
      return holoscan::make_unexpected(std::move(info).error());
    }
    camera_infos.push_back(std::move(*info));
  }
  return camera_infos;
}

holoscan::expected<bool, holoscan::Error> CuvslamOp::ensure_tracker() {
  // A background build started on an earlier set may still be running. Keep skipping until it
  // finishes so the tracker is never used half-built.
  if (tracker_build_.valid()) {
    if (tracker_build_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      return false;
    }
    if (!tracker_build_.get()) {
      return holoscan::make_unexpected(
          failure("Unable to start visual odometry from the camera calibration."));
    }
    report_ready();
  }
  if (tracker_) {
    return true;
  }
  return start_tracker_build();
}

holoscan::expected<bool, holoscan::Error> CuvslamOp::start_tracker_build() {
  auto camera_infos = receive_calibration();
  if (!camera_infos) {
    return holoscan::make_unexpected(std::move(camera_infos).error());
  }
  if (!camera_infos->has_value()) {
    return false;
  }
  if (params().async_setup) {
    tracker_build_ =
        std::async(std::launch::async, [this, infos = std::move(**camera_infos)]() mutable {
          return build_tracker(std::move(infos));
        });
    return false;
  }
  if (!build_tracker(std::move(**camera_infos))) {
    return holoscan::make_unexpected(
        failure("Unable to start visual odometry from the camera calibration."));
  }
  report_ready();
  // The first frames after the cameras start streaming are not reliable, so they are skipped.
  return false;
}

holoscan::expected<cuvslam::Image, holoscan::Error> CuvslamOp::describe_frame(
    const holoscan::schema::ImageT& frame, const std::size_t index,
    const std::int64_t set_timestamp_ns) {
  const holoscan::Tensor* const pixels = frame.data.get();
  if (pixels == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "A camera frame reached visual odometry without pixel data."});
  }
  if (frame.encoding != holoscan::schema::ImageEncoding_RGB8 &&
      frame.encoding != holoscan::schema::ImageEncoding_MONO8) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "Visual odometry accepts only 8-bit color or grayscale camera frames."});
  }
  const DLDeviceType device = frame.data->device().device_type;
  const bool on_gpu = device == kDLCUDA || device == kDLCUDAManaged;
  if (!on_gpu && device != kDLCPU && device != kDLCUDAHost) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "A camera frame reached visual odometry in unusable memory."});
  }

  // The extents and the pixels arrive as separate facts: the descriptor says how large the frame
  // is, the tensor holds the bytes, and nothing has so far required the two to agree. The tracker
  // reads the tensor using the descriptor's extents, so a disagreement is not a wrong picture but
  // a read past the end of the buffer. They are reconciled here, along with the element type the
  // tracker is about to assume.
  const std::int64_t channels = frame.encoding == holoscan::schema::ImageEncoding_RGB8 ? 3 : 1;
  const DLDataType dtype = pixels->dtype();
  const std::vector<std::int64_t> shape = pixels->shape();
  // A mono frame may describe its single channel or leave it out; a color frame may not.
  const bool shaped_as_image =
      (shape.size() == 3U && shape[2] == channels) || (shape.size() == 2U && channels == 1);
  if (pixels->data() == nullptr || frame.width <= 0 || frame.height <= 0 || dtype.code != kDLUInt ||
      dtype.bits != 8U || dtype.lanes != 1U || !shaped_as_image || shape[0] != frame.height ||
      shape[1] != frame.width) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument,
        "A camera frame reached visual odometry as something other than the 8-bit image its "
        "description claims."});
  }

  // Rows may be padded -- the tracker is told the pitch rather than assuming none -- but the
  // padding has to be real: a pitch narrower than a row, or a buffer shorter than the rows it is
  // said to hold, would have the tracker read what it does not own. The pitch is also handed over
  // as a 32-bit quantity, so it has to survive the narrowing.
  const std::int64_t row_bytes = static_cast<std::int64_t>(frame.width) * channels;
  const std::int64_t pitch = pixels->strides().front();
  if (pitch < row_bytes || pitch > std::numeric_limits<std::int32_t>::max() ||
      pixels->nbytes() < (static_cast<std::int64_t>(frame.height) - 1) * pitch + row_bytes) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "A camera frame reached visual odometry with a row layout its pixel "
                        "buffer cannot hold."});
  }

  cuvslam::Image image{};
  image.pixels = pixels->data();
  image.width = frame.width;
  image.height = frame.height;
  image.pitch = static_cast<std::int32_t>(pitch);
  image.encoding = frame.encoding == holoscan::schema::ImageEncoding_RGB8
                       ? cuvslam::ImageData::Encoding::RGB
                       : cuvslam::ImageData::Encoding::MONO;
  image.data_type = cuvslam::ImageData::DataType::UINT8;
  image.is_gpu_mem = on_gpu;
  // Its own capture time, not the set's. The alignment window bounds how far apart these are, it
  // does not make them equal, and handing every image the first camera's timestamp would present
  // the set to the tracker as perfectly synchronized when it is not.
  image.timestamp_ns = frame.header ? frame.header->capture_timestamp_ns : set_timestamp_ns;
  image.camera_index = static_cast<std::uint32_t>(index);
  return image;
}

holoscan::expected<std::vector<cuvslam::Image>, holoscan::Error> CuvslamOp::describe_frames(
    const FrameSet& frames, const std::int64_t set_timestamp_ns) {
  std::vector<cuvslam::Image> images;
  images.reserve(frames.size());
  for (std::size_t index = 0; index < frames.size(); ++index) {
    auto image = describe_frame(frames[index].data, index, set_timestamp_ns);
    if (!image) {
      return holoscan::make_unexpected(std::move(image).error());
    }
    images.push_back(*image);
  }
  return images;
}

holoscan::expected<void, holoscan::Error> CuvslamOp::finish_frames(
    holoscan::ExecutionContext& context) const {
  // cuVSLAM schedules its own work on streams this graph does not own, so the frames have to be
  // finished before the pointers are handed over. This is the one blocking point in the pipeline
  // and it exists because the tracker is an external library, not because the dataflow needs it.
  const cudaError_t error = cudaStreamSynchronize(context.cuda_stream().get());
  if (error != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("Unable to finish the camera frames before tracking: {}.",
                       cudaGetErrorString(error));
    return holoscan::make_unexpected(
        failure("The camera frames could not be prepared for visual odometry."));
  }
  return {};
}

holoscan::expected<std::optional<cuvslam::PoseEstimate>, holoscan::Error> CuvslamOp::track(
    holoscan::ExecutionContext& context, const FrameSet& frames,
    const std::int64_t set_timestamp_ns) {
  auto images = describe_frames(frames, set_timestamp_ns);
  if (!images) {
    return holoscan::make_unexpected(std::move(images).error());
  }
  if (auto finished = finish_frames(context); !finished) {
    return holoscan::make_unexpected(std::move(finished).error());
  }
  cuvslam::PoseEstimate estimate = tracker_->Track(*images);
  if (!estimate.world_from_rig.has_value()) {
    HOLOSCAN_LOG_WARN("Visual odometry lost track of the camera rig on this frame.");
    return std::nullopt;
  }
  return estimate;
}

isaac::OdometryCpuT CuvslamOp::describe_estimate(const cuvslam::PoseEstimate& estimate) const {
  const cuvslam::Pose& pose = estimate.world_from_rig->pose;
  const auto& config = params();

  isaac::OdometryCpuT message;
  message.header = std::make_shared<holoscan::schema::HeaderT>();
  message.header->frame_id = config.odometry_frame;
  message.header->capture_timestamp_ns = estimate.timestamp_ns;
  message.child_frame_id = config.rig_frame;
  message.pose = std::make_shared<isaac::PoseCpuT>();
  message.pose->position =
      holoscan::schema::Vector3d(pose.translation[0], pose.translation[1], pose.translation[2]);
  // Both order a quaternion x, y, z, w, so this is a copy rather than a reordering.
  message.pose->orientation = holoscan::schema::Quaterniond(pose.rotation[0], pose.rotation[1],
                                                            pose.rotation[2], pose.rotation[3]);
  // cuVSLAM 17 and Odometry both use row-major (x, y, z, roll, pitch, yaw) covariance.
  const auto& covariance = estimate.world_from_rig->covariance_xyz_rpy;
  std::copy(covariance.begin(), covariance.end(), message.pose_covariance.data());
  // This tracker estimates a pose, not a velocity, and the table requires both. A zero twist
  // beside a zero covariance would read as a rig measured to be perfectly stationary, so the
  // covariance says the estimate is absent instead, the way sensor_msgs spells the same thing.
  message.twist = std::make_shared<isaac::TwistCpuT>();
  message.twist_covariance = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();
  message.twist_covariance(0, 0) = -1.0;
  return message;
}

void CuvslamOp::record_in_pose_tree(const cuvslam::PoseEstimate& estimate) {
  const cuvslam::Pose& pose = estimate.world_from_rig->pose;
  const double estimate_time_s =
      std::chrono::duration<double>(std::chrono::nanoseconds(estimate.timestamp_ns)).count();
  const isaac::os::pose_tree::Pose3d odometry_t_rig{
      isaac::os::pose_tree::SO3d::from_quaternion(isaac::os::pose_tree::Quaternion<double>(
          pose.rotation[3], pose.rotation[0], pose.rotation[1], pose.rotation[2])),
      isaac::os::pose_tree::Vector3<double>(pose.translation[0], pose.translation[1],
                                            pose.translation[2])};
  if (!pose_tree_->set(odometry_frame_id_, rig_frame_id_, estimate_time_s, odometry_t_rig)) {
    HOLOSCAN_LOG_WARN("Unable to record where the camera rig was at {:.6f} s.", estimate_time_s);
  }
}

holoscan::expected<void, holoscan::Error> CuvslamOp::publish(
    const isaac::OdometryCpuT& message, const holoscan::Sample<holoscan::schema::ImageT>& source) {
  auto published = odometry.emit(message, holoscan::sensor_io::derive_from(source));
  if (published || published.error().code != holoscan::ErrorCode::kNotReady) {
    return published;
  }
  // An output nothing is connected to never gets a runtime binding, and publishing into one
  // reports that rather than succeeding. It is not a failure here: an application that reads the
  // rig's pose out of the tree of coordinate frames, which is where this operator writes it, has
  // no reason to connect this port as well. Said once, because it describes how the graph was
  // wired and repeating it every frame would say nothing new.
  if (!reported_unconsumed_odometry_) {
    reported_unconsumed_odometry_ = true;
    HOLOSCAN_LOG_INFO(
        "Nothing is connected to the odometry output, so estimates reach the rest of the process "
        "through the '{}' frame of the pose tree only.",
        odometry_frame_);
  }
  return {};
}

holoscan::expected<void, holoscan::Error> CuvslamOp::publish_estimate(
    const cuvslam::PoseEstimate& estimate,
    const holoscan::Sample<holoscan::schema::ImageT>& source) {
  const isaac::OdometryCpuT message = describe_estimate(estimate);
  record_in_pose_tree(estimate);
  if (!reported_first_pose_) {
    reported_first_pose_ = true;
    HOLOSCAN_LOG_INFO("Visual odometry produced its first pose estimate.");
  }
  report_progress(estimate.timestamp_ns, message);
  return publish(message, source);
}

holoscan::expected<void, holoscan::Error> CuvslamOp::compute(holoscan::ExecutionContext& context) {
  auto frames = receive_frames();
  if (!frames) {
    return holoscan::make_unexpected(std::move(frames).error());
  }
  auto ready = ensure_tracker();
  if (!ready) {
    return holoscan::make_unexpected(std::move(ready).error());
  }

  // Recorded for every set handed to the tracker, including the ones skipped below, because how
  // far apart the cameras fire is a fact about the rig rather than about tracking.
  const std::int64_t timestamp_ns = set_timestamp_of(*frames);
  worst_capture_spread_ns_ =
      std::max(worst_capture_spread_ns_, capture_spread_of(*frames, timestamp_ns));
  if (!*ready) {
    return {};
  }

  auto estimate = track(context, *frames, timestamp_ns);
  if (!estimate) {
    return holoscan::make_unexpected(std::move(estimate).error());
  }
  if (!estimate->has_value()) {
    return {};
  }
  return publish_estimate(**estimate, (*frames)[0]);
}

void CuvslamOp::begin_report_period(std::int64_t time_ns) {
  period_start_ns_ = time_ns;
  previous_estimate_ns_ = time_ns;
  estimate_count_ = 0;
  shortest_estimate_gap_ns_ = std::numeric_limits<std::int64_t>::max();
  longest_estimate_gap_ns_ = 0;
  worst_capture_spread_ns_ = 0;
}

void CuvslamOp::report_progress(std::int64_t time_ns, const isaac::OdometryCpuT& estimate) {
  const double report_period_s = params().report_period_s;
  if (report_period_s <= 0.0) {
    return;
  }
  if (!period_started_) {
    begin_report_period(time_ns);
    period_started_ = true;
    return;
  }

  ++estimate_count_;
  const std::int64_t gap_ns = time_ns - previous_estimate_ns_;
  shortest_estimate_gap_ns_ = std::min(shortest_estimate_gap_ns_, gap_ns);
  longest_estimate_gap_ns_ = std::max(longest_estimate_gap_ns_, gap_ns);
  previous_estimate_ns_ = time_ns;

  const double elapsed_s = static_cast<double>(time_ns - period_start_ns_) * kNanosecondsToSeconds;
  if (elapsed_s <= report_period_s || elapsed_s <= 0.0) {
    return;
  }

  HOLOSCAN_LOG_INFO(
      "Visual odometry tracked {:.3f} frame sets per second over the last {:.1f} s. Frames within "
      "a set were captured up to {:.3f} ms apart, and the gap between sets ranged from {:.3f} ms "
      "to {:.3f} ms.",
      static_cast<double>(estimate_count_) / elapsed_s, elapsed_s,
      static_cast<double>(worst_capture_spread_ns_) * kNanosecondsToMilliseconds,
      static_cast<double>(shortest_estimate_gap_ns_) * kNanosecondsToMilliseconds,
      static_cast<double>(longest_estimate_gap_ns_) * kNanosecondsToMilliseconds);
  const auto& position = estimate.pose->position;
  const auto& orientation = estimate.pose->orientation;
  HOLOSCAN_LOG_INFO(
      "Camera rig '{}' is at x {:.5f} m, y {:.5f} m, z {:.5f} m in '{}', facing "
      "(w {:.5f}, x {:.5f}, y {:.5f}, z {:.5f}), as of {:.6f} s.",
      estimate.child_frame_id, position.x(), position.y(), position.z(), estimate.header->frame_id,
      orientation.w(), orientation.x(), orientation.y(), orientation.z(),
      static_cast<double>(time_ns) * kNanosecondsToSeconds);

  begin_report_period(time_ns);
}

void CuvslamOp::stop() {
  // A background build reads this operator's parameters and writes the tracker, so it has to
  // finish before anything it touches is torn down.
  if (tracker_build_.valid()) {
    try {
      tracker_build_.get();
    } catch (...) {
      // The build result no longer matters; only its completion does.
    }
  }
  tracker_.reset();
}

}  // namespace isaac::os::ops
