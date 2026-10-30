/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <holoscan/core/graph.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "circular_motion_op.h"
#include "common/image_schema/image_schema.h"
#include "common/mcap_logger/mcap_logger.h"
#include "image_counter_op.h"
#include "joint_state_wave_op.h"
#include "static_camera_info_op.h"

namespace isaac::os::mcap_logger_reference {

// Composes every generator op in this app and attaches each output to `logger` under a fixed,
// descriptive topic name -- a minimal worked example of HsdkMcapLogger usage covering both the
// plain-flatbuffer attach<T>() overload (joint states, poses, twists, transforms, camera info)
// and the tensor-schema attach<T>(..., contract) overload (the two image streams).
class ReferenceGraph final : public holoscan::Graph {
 public:
  explicit ReferenceGraph(isaac::os::mcap_logger::HsdkMcapLogger& logger)
      : Graph("mcap_logger_reference"), logger_(logger) {}

  // Whether compose() attached counter_image_encoded, which it does only when this host can
  // encode. Meaningful once the graph has been composed, i.e. after holoscan::compile().
  [[nodiscard]] bool records_encoded_topic() const { return records_encoded_topic_; }

 protected:
  void compose() override {
    // RealtimeClock's default Config has use_time_since_epoch = false: its timeline starts at 0
    // when the graph starts, not at the Unix epoch. That clock is what SampleMetadata::
    // publish_timestamp_ns (and therefore every logged message's MCAP log_time -- what Foxglove's
    // timeline/date display actually reads) is derived from, so leaving it at the default renders
    // every recording as starting "just after the Unix epoch" (1969/1970) in Foxglove, regardless
    // of when it was actually captured.
    set_default_clock(add_clock<holoscan::RealtimeClock>(
        "clock", holoscan::RealtimeClock::Config{.use_time_since_epoch = true}));

    auto sine = op<JointStateWaveOp>("sine_generator", Waveform::kSine);
    logger_.attach<isaac::JointStateCpuT>(*this, "sine_joint_state", sine->joint_state);

    auto cosine = op<JointStateWaveOp>("cosine_generator", Waveform::kCosine);
    logger_.attach<isaac::JointStateCpuT>(*this, "cosine_joint_state", cosine->joint_state);

    auto motion = op<CircularMotionOp>("circular_motion_generator");
    logger_.attach<isaac::PoseCpuT>(*this, "circular_pose", motion->out_pose);
    logger_.attach<isaac::TwistCpuT>(*this, "circular_twist", motion->out_twist);
    logger_.attach<isaac::TransformCpuT>(*this, "circular_transform", motion->out_transform);

    auto camera_info = op<StaticCameraInfoOp>("camera_info_generator");
    logger_.attach<isaac::CameraInfoCpuT>(*this, "camera_info", camera_info->out_camera_info);

    // Two image streams recorded side by side so one recording shows both paths the tensor-schema
    // attach<T>() overload supports, and what each costs.
    //
    // Raw: no VideoEncodeOptions, so every frame lands as uncompressed tensor bytes inside an
    // isaac.mcap_logger.TensorFrame envelope, which needs the isaac-to-foxglove-converter
    // extension to render.
    auto image = op<ImageCounterOp>("image_counter_generator", kImageWidth, kImageHeight,
                                    std::string("camera"));
    logger_.attach<holoscan::schema::ImageT>(
        *this, "counter_image", image->out_image,
        isaac::os::image_schema::ImageTensorFieldContract(kImageWidth, kImageHeight,
                                                          holoscan::schema::ImageEncoding_RGB8));

    // Encoded: VideoEncodeOptions{} (all defaults: h264, 10 Mbit/s, 30 fps, CUDA device 0) records
    // this topic as foxglove.CompressedVideo rather than raw tensor bytes, so an Image panel
    // renders it with no Foxglove extension installed. It also writes a
    // counter_image_encoded/metadata sidecar carrying the holoscan.schema.Image fields
    // CompressedVideo has no room for.
    //
    // The logger refuses a topic that asks for video on a GPU that cannot encode, and recording it
    // raw instead would push a 1080p stream through the logger to show nothing this topic is here
    // to show. So on a machine with no usable hardware encoder this topic is left out entirely --
    // no generator, no attach -- with a warning, and the recording has one image topic, not two.
    const isaac::os::mcap_logger::VideoEncodeOptions encode_options{};
    if (!isaac::os::mcap_logger::FrameEncoder::hardware_available(encode_options.cuda_device,
                                                                  encode_options.codec)) {
      HOLOSCAN_LOG_WARN(
          "GPU {} has no usable hardware video encoder, so counter_image_encoded is not recorded.",
          encode_options.cuda_device);
      return;
    }
    // The contract fixes the tensor's byte bound, so it has to be sized to this instance's own
    // resolution, not the raw one's.
    auto encoded_image = op<ImageCounterOp>("encoded_image_counter_generator", kEncodedImageWidth,
                                            kEncodedImageHeight, std::string("camera_encoded"));
    logger_.attach<holoscan::schema::ImageT>(
        *this, "counter_image_encoded", encoded_image->out_image,
        isaac::os::image_schema::ImageTensorFieldContract(kEncodedImageWidth, kEncodedImageHeight,
                                                          holoscan::schema::ImageEncoding_RGB8),
        isaac::os::mcap_logger::TopicLogOptions{.encode = encode_options});
    records_encoded_topic_ = true;
  }

 private:
  isaac::os::mcap_logger::HsdkMcapLogger& logger_;
  bool records_encoded_topic_ = false;
};

}  // namespace isaac::os::mcap_logger_reference
