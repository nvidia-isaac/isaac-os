/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER

#include <atomic>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/logger/logger.hpp>

#include "applications/ros2_bridge_reference/image_generator_op.h"
#include "common/image_schema/image_schema.h"

namespace isaac::os::ros2_bridge_reference {

// What ImageCheckOp saw, readable by main.cc after the graph stops.
struct ImageCheckTally {
  std::atomic<std::uint64_t> received{0};
  std::atomic<std::uint64_t> bad_images{0};  // not in GPU memory, or the wrong size or pixels
};

[[nodiscard]] inline ImageCheckTally& image_check_tally() {
  static ImageCheckTally tally;
  return tally;
}

// Receives the images that made the round trip out through to_ros2 as CUDA buffers, through the
// external ROS2 relay node, and back in through from_ros2. Each image must arrive as a tensor in
// GPU memory, with the size and pixels ImageGeneratorOp produced; anything else is logged as an
// error and counted as wrong.
class ImageCheckOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in_image, "in_image"); }

  void stop() override {
    const ImageCheckTally& tally = image_check_tally();
    HOLOSCAN_LOG_INFO("ImageCheckOp received {} images ({} wrong).", tally.received.load(),
                      tally.bad_images.load());
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = in_image.receive();
    if (!received) {
      return holoscan::make_unexpected(received.error());
    }
    ImageCheckTally& tally = image_check_tally();
    tally.received.fetch_add(1);

    const holoscan::schema::ImageT& image = received->data;
    if (!image.data || !image.header) {
      tally.bad_images.fetch_add(1);
      return {};
    }
    const holoscan::Tensor& tensor = *image.data;
    if (tensor.device().device_type != kDLCUDA) {
      // The image went through ROS as a CUDA buffer, so a host tensor means something is wrong.
      HOLOSCAN_LOG_ERROR("ImageCheckOp received an image whose tensor is not in GPU memory.");
      tally.bad_images.fetch_add(1);
      return {};
    }
    // copy_to_host orders the read after the producer, whichever memory the tensor is in.
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(tensor.nbytes()));
    if (image.encoding != holoscan::schema::ImageEncoding_RGB8 || image.width != kImageWidth ||
        image.height != kImageHeight || pixels.size() != kImageBytes ||
        !tensor.copy_to_host(pixels.data(), pixels.size())) {
      tally.bad_images.fetch_add(1);
      return {};
    }
    const std::uint8_t expected = ImagePatternByte(image.header->capture_timestamp_ns);
    for (const std::uint8_t byte : pixels) {
      if (byte != expected) {
        tally.bad_images.fetch_add(1);
        break;
      }
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> in_image;
};

}  // namespace isaac::os::ros2_bridge_reference

#endif  // ISAAC_ROS2_BRIDGE_CUDA_BUFFER
