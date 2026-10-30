/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlpack/dlpack.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "camera_info_cpu_schema_traits.h"
#include "common/image_schema/image_schema.h"
#include "common/mcap_replayer/compressed_video_builder.h"
#include "common/mcap_replayer/mcap_replayer.h"
#include "joint_state_cpu_schema_traits.h"
#include "pose_cpu_schema_traits.h"
#include "transform_cpu_schema_traits.h"
#include "twist_cpu_schema_traits.h"

namespace isaac::os::mcap_replay_reference {

// The topics of an mcap_logger_reference recording that this app replays. The replayer checks
// each name and its type against the file, so a mismatch is reported before the graph runs.
inline constexpr const char* kSineJointStateTopic = "sine_joint_state";
inline constexpr const char* kCosineJointStateTopic = "cosine_joint_state";
inline constexpr const char* kCircularPoseTopic = "circular_pose";
inline constexpr const char* kCircularTwistTopic = "circular_twist";
inline constexpr const char* kCircularTransformTopic = "circular_transform";
inline constexpr const char* kCameraInfoTopic = "camera_info";
inline constexpr const char* kCounterImageTopic = "counter_image";
// Recorded as hardware-encoded video only on a host with an encoder, so a recording may not hold
// it, and a host without a decoder cannot replay it. It is replayed when both are fine and
// otherwise left out; see encoded_topic_skip_reason().
inline constexpr const char* kEncodedCounterImageTopic = "counter_image_encoded";

// Every topic above, for code that must count only what this app replays. The encoded topic is
// listed too: a topic that is not being replayed is never reported as unsubscribed.
inline constexpr std::array<const char*, 8> kReplayedTopics = {
    kSineJointStateTopic,    kCosineJointStateTopic, kCircularPoseTopic, kCircularTwistTopic,
    kCircularTransformTopic, kCameraInfoTopic,       kCounterImageTopic, kEncodedCounterImageTopic};

// The frame geometry the display layer and window are sized to. Layer bounds must be declared in
// compose(), before any frame is read, so they are fixed to what mcap_logger_reference records.
// A differently sized recording still replays on every topic; Holoviz logs and drops frames that
// do not fit the layer rather than showing them misinterpreted.
inline constexpr std::uint32_t kDisplayWidth = 640;
inline constexpr std::uint32_t kDisplayHeight = 480;
// The encoded stream is 1080p. Its layer takes the full frame, and its window is half that size on
// each side, since Holoviz scales the image to the window.
inline constexpr std::uint32_t kEncodedImageWidth = 1920;
inline constexpr std::uint32_t kEncodedImageHeight = 1080;
inline constexpr std::uint32_t kEncodedWindowWidth = kEncodedImageWidth / 2;
inline constexpr std::uint32_t kEncodedWindowHeight = kEncodedImageHeight / 2;
// The largest image the replayed topic is imported to carry; a bigger frame is refused when it is
// published. Far above the 640x480 RGB8 frames mcap_logger_reference records.
inline constexpr std::uint64_t kMaxImageBytes = 8U * 1024U * 1024U;
// The same for the decoded 1080p RGB8 frames of the encoded topic (6.2 MB each).
inline constexpr std::uint64_t kMaxEncodedImageBytes = 16U * 1024U * 1024U;

// Consumes a replayed topic. Every replayed topic needs a consumer, and the run's counts come from
// the replayer's own stats rather than anything counted here.
template <typename T>
class DrainOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(16); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{in});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = in.receive();
    if (!received) {
      return holoscan::make_unexpected(received.error());
    }
    return {};
  }

  holoscan::Input<T> in;
};

namespace detail {

// A DLPack export of one tensor with a different shape over the same bytes. It owns the source
// export, which it releases once the view is released.
struct ReshapedExport {
  DLManagedTensorVersioned* source;
  std::array<std::int64_t, 3> shape;
  DLManagedTensorVersioned view;
};

// Frees a ReshapedExport and the source export it owns, once the view of it is released.
inline void release_reshaped_export(DLManagedTensorVersioned* view) {
  const std::unique_ptr<ReshapedExport> owner(static_cast<ReshapedExport*>(view->manager_ctx));
  if (owner->source->deleter != nullptr) {
    owner->source->deleter(owner->source);
  }
}

// Views a contiguous tensor as the given shape without copying it. The source keeps its read-only
// authority, which the DLPack export carries along.
inline holoscan::expected<holoscan::Tensor, holoscan::Error> reshaped_view(
    const holoscan::Tensor& source, const std::array<std::int64_t, 3>& shape) {
  auto exported = source.to_dlpack_versioned();
  if (!exported) {
    return holoscan::make_unexpected(exported.error());
  }
  auto reshaped = std::make_unique<ReshapedExport>(ReshapedExport{*exported, shape, **exported});
  reshaped->view.manager_ctx = reshaped.get();
  reshaped->view.deleter = release_reshaped_export;
  reshaped->view.dl_tensor.ndim = static_cast<std::int32_t>(shape.size());
  reshaped->view.dl_tensor.shape = reshaped->shape.data();
  reshaped->view.dl_tensor.strides = nullptr;
  auto view = holoscan::make_tensor_from_dlpack_versioned(&reshaped->view);
  if (!view) {
    // The import declined it, so nothing took ownership; releasing it frees the export.
    release_reshaped_export(&reshaped->view);
  }
  // The view's deleter now owns it: this was released inside the import on success.
  reshaped.release();
  return view;
}

}  // namespace detail

// Republishes a replayed image with its pixels shaped height x width x 3, which is what Holoviz's
// RGB8 layer reads. Recorded images arrive as one flat byte buffer, so this is only a new view of
// the same bytes: nothing is copied, and the view keeps the received buffer alive.
class ImageToHwcOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in").queue_depth(4);
    spec.output(out, "out");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{in});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = in.receive();
    if (!received) {
      return holoscan::make_unexpected(received.error());
    }
    holoscan::schema::ImageT image = received->data;
    // Only an RGB8 frame whose bytes are exactly width x height x 3 can be viewed this way; any
    // other would be read out of bounds or shown as garbage, so it is dropped and replay of every
    // topic carries on.
    const auto width = static_cast<std::int64_t>(image.width);
    const auto height = static_cast<std::int64_t>(image.height);
    // Unsigned, so a recording with enormous dimensions cannot overflow: two positive 32-bit sizes
    // times 3 always fit in 64 bits. Zero for a size that is not positive, which never matches.
    const std::uint64_t expected_bytes =
        width > 0 && height > 0
            ? static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) * kRgbChannels
            : 0U;
    if (image.data == nullptr || expected_bytes == 0 ||
        image.encoding != holoscan::schema::ImageEncoding_RGB8 ||
        static_cast<std::uint64_t>(image.data->nbytes()) != expected_bytes) {
      if (!warned_) {
        warned_ = true;
        HOLOSCAN_LOG_WARN(
            "A replayed frame is not tightly packed RGB8 and is not shown. Later frames like it "
            "are skipped silently.");
      }
      return {};
    }

    auto view = detail::reshaped_view(*image.data,
                                      {height, width, static_cast<std::int64_t>(kRgbChannels)});
    if (!view) {
      return holoscan::make_unexpected(view.error());
    }
    image.data = std::make_shared<holoscan::Tensor>(std::move(*view));

    // The display takes frames more slowly than they are replayed, and only ever shows the newest
    // one. A frame it has no room for is dropped here: returning the backpressure as an error
    // would stop the whole graph.
    auto sent = out.emit(image);
    if (!sent && sent.error().code == holoscan::ErrorCode::kBackpressured) {
      return {};
    }
    return sent;
  }

  holoscan::Input<holoscan::schema::ImageT> in;
  holoscan::Output<holoscan::schema::ImageT> out;

 private:
  static constexpr std::uint64_t kRgbChannels = 3;

  // Whether the first frame that could not be shown has been reported.
  bool warned_ = false;
};

// Why the encoded counter stream is not replayed from this recording on this host, or nothing if
// it is: the recording does not hold it, or the GPU that decodes video cannot decode it.
inline std::optional<std::string> encoded_topic_skip_reason(
    const std::filesystem::path& input_path,
    const isaac::os::mcap_replayer::ReplayOptions& options) {
  using isaac::os::mcap_replayer::McapReplayer;
  if (!McapReplayer::recording_has_topic(input_path, kEncodedCounterImageTopic)) {
    return std::string("The recording has no encoded video stream.");
  }
  const std::optional<std::string> codec =
      McapReplayer::recorded_video_codec(input_path, kEncodedCounterImageTopic);
  if (!codec.has_value()) {
    return std::string(
        "The recording's encoded video stream cannot be read; see the warning above.");
  }
  return isaac::os::mcap_replayer::video_decode_unavailable_reason(
      options.video_decoder_cuda_device, *codec);
}

// Builds the replayer for every topic of an mcap_logger_reference recording, the encoded counter
// stream too when with_encoded. Throws if the file cannot be opened or a topic is missing or
// recorded as a different type.
inline isaac::os::mcap_replayer::McapReplayer make_replayer(
    const std::filesystem::path& input_path, isaac::os::mcap_replayer::ReplayOptions options,
    bool with_encoded) {
  using isaac::os::mcap_replayer::McapReplayer;
  using isaac::os::mcap_replayer::Topic;
  const auto build = [&](auto... extra_topics) {
    return McapReplayer(
        input_path, options, Topic<isaac::JointStateCpuT>{kSineJointStateTopic},
        Topic<isaac::JointStateCpuT>{kCosineJointStateTopic},
        Topic<isaac::PoseCpuT>{kCircularPoseTopic}, Topic<isaac::TwistCpuT>{kCircularTwistTopic},
        Topic<isaac::TransformCpuT>{kCircularTransformTopic},
        Topic<isaac::CameraInfoCpuT>{kCameraInfoTopic},
        Topic<holoscan::schema::ImageT>{
            kCounterImageTopic, isaac::os::image_schema::ImageTensorFieldContract(kMaxImageBytes)},
        extra_topics...);
  };
  // The encoded topic is declared exactly like the raw one: the replayer sees from the recording
  // that it is video and decodes it.
  return with_encoded
             ? build(Topic<holoscan::schema::ImageT>{
                   kEncodedCounterImageTopic,
                   isaac::os::image_schema::ImageTensorFieldContract(kMaxEncodedImageBytes)})
             : build();
}

// Wires every topic of the replayer to a consumer: a drain for each, or a display for each image
// when visualizing. The replayer is owned by the caller, who starts and stops it.
class ReplayGraph final : public holoscan::Graph {
 public:
  ReplayGraph(isaac::os::mcap_replayer::McapReplayer& replayer, bool visualize, bool with_encoded)
      : Graph("mcap_replay_reference"),
        replayer_(replayer),
        visualize_(visualize),
        with_encoded_(with_encoded) {}

 private:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>(
        "clock", holoscan::RealtimeClock::Config{.use_time_since_epoch = true}));

    drain<isaac::JointStateCpuT>(kSineJointStateTopic, "sine_joint_state_sink");
    drain<isaac::JointStateCpuT>(kCosineJointStateTopic, "cosine_joint_state_sink");
    drain<isaac::PoseCpuT>(kCircularPoseTopic, "circular_pose_sink");
    drain<isaac::TwistCpuT>(kCircularTwistTopic, "circular_twist_sink");
    drain<isaac::TransformCpuT>(kCircularTransformTopic, "circular_transform_sink");
    drain<isaac::CameraInfoCpuT>(kCameraInfoTopic, "camera_info_sink");

    if (!visualize_) {
      drain<holoscan::schema::ImageT>(kCounterImageTopic, "counter_image_sink");
      if (with_encoded_) {
        drain<holoscan::schema::ImageT>(kEncodedCounterImageTopic, "counter_image_encoded_sink");
      }
      return;
    }

    display(kCounterImageTopic, "counter_image", "Replayed counter",
            {.image = {kDisplayWidth, kDisplayHeight}, .window = {kDisplayWidth, kDisplayHeight}});
    if (with_encoded_) {
      display(kEncodedCounterImageTopic, "counter_image_encoded", "Replayed encoded counter",
              {.image = {kEncodedImageWidth, kEncodedImageHeight},
               .window = {kEncodedWindowWidth, kEncodedWindowHeight}});
    }
  }

  // The sizes of one image display: the largest frame its layer takes and its window.
  struct DisplaySize {
    std::pair<std::uint32_t, std::uint32_t> image;
    std::pair<std::uint32_t, std::uint32_t> window;
  };

  // Shows a replayed image topic in a window of its own.
  void display(const char* topic, const char* name, const char* title, const DisplaySize& size) {
    holoscan::visualization::LayerStack layers;
    layers.add_image_layer(name, {.format = holoscan::visualization::ImageFormat::kR8G8B8Unorm,
                                  .max_width = size.image.first,
                                  .max_height = size.image.second});
    // Each replayed frame triggers one render. A clock-driven display would select frames by
    // capture time, which a replayed frame does not have.
    auto viz = op<holoscan::visualization::HolovizOp>(
        std::string(name) + "_display", std::move(layers),
        holoscan::visualization::SinkDesc{holoscan::visualization::WindowSinkDesc{
            .title = title, .width = size.window.first, .height = size.window.second}});
    auto to_hwc = op<ImageToHwcOp>(std::string(name) + "_hwc");
    replayer_.attach<holoscan::schema::ImageT>(*this, topic, to_hwc->in);
    // The display only ever shows the newest frame, and each frame the view passes on keeps the
    // received frame's buffer in use until Holoviz is done with it. A one-deep mailbox that drops
    // the oldest keeps that to a frame or two however much faster than the display frames arrive.
    constexpr holoscan::ConnectionOptions kNewestFrame{
        .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U};
    add_flow(to_hwc->out, viz->image_layer_input(0), kNewestFrame);
  }

  template <typename T>
  void drain(const char* topic, const char* name) {
    auto sink = op<DrainOp<T>>(name);
    replayer_.attach<T>(*this, topic, sink->in);
  }

  isaac::os::mcap_replayer::McapReplayer& replayer_;
  bool visualize_;
  bool with_encoded_;
};

}  // namespace isaac::os::mcap_replay_reference
