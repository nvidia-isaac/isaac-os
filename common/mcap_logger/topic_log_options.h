/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "holoscan/core/connection_options.hpp"
#include <fmt/format.h>

#include "common/mcap_logger/video_encode_options.h"

namespace isaac::os::mcap_logger {

// What one attached topic asks of the logger, beyond which output it records. Every field has a
// default, so `attach(...)` without one records the topic raw with a queue that suits most graphs.
//
// An aggregate: add new fields at the END, and build it with designated initializers, which
// survive appending and cannot misassign a field.
struct TopicLogOptions {
  // Record this topic as hardware-encoded video (foxglove.CompressedVideo) instead of raw tensors.
  // For tensor-schema topics whose type is holoscan::schema::ImageT only; anything else throws
  // from attach(). Unset records raw tensors, as every topic did before this option existed. The
  // logger's start() throws if this host has no usable hardware encoder on the device named, rather
  // than recording raw tensors in its place; FrameEncoder::hardware_available() lets a caller that
  // wants that fallback choose it openly.
  std::optional<VideoEncodeOptions> encode;

  // How many of this topic's samples the transport holds for the logger to write; when it is full
  // the oldest is dropped, with a warning. Must be at least 1. Per topic.
  std::uint32_t queue_depth = 16;
};

// The edge into a topic's exported output: the caller's options with the queue set from the topic.
// The export's own subscriber bounds are left unset because an export bound silently overrides the
// edge's depth.
[[nodiscard]] inline holoscan::ConnectionOptions EdgeOptionsFor(
    const TopicLogOptions& topic, holoscan::ConnectionOptions options) {
  options.queue_policy = holoscan::QueuePolicy::kDropOldest;
  options.queue_depth = topic.queue_depth;
  return options;
}

// Throws std::invalid_argument for options no topic could run with. Called from attach(), so a
// bad request fails at compose time and leaves no half-wired topic behind.
inline void ValidateTopicLogOptions(const TopicLogOptions& options, std::string_view topic) {
  if (options.queue_depth == 0) {
    throw std::invalid_argument(fmt::format(
        "The MCAP logger's topic '{}' cannot hold any samples because its queue depth is 0. Set "
        "it to at least 1.",
        topic));
  }
  if (options.encode) {
    ValidateVideoEncodeOptions(*options.encode);
  }
}

}  // namespace isaac::os::mcap_logger
