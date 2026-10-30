/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace isaac::os::mcap_logger_reference {

// Dimensions of the raw (unencoded) image stream. Also what StaticCameraInfoOp's intrinsics are
// built from, so the recorded camera_info pairs with that stream.
inline constexpr std::uint32_t kImageWidth = 640;
inline constexpr std::uint32_t kImageHeight = 480;

// Dimensions of the hardware-encoded image stream. Deliberately a realistic camera resolution and
// nearly seven times the pixel count of the raw stream (2,073,600 against 307,200), so comparing
// the two topics in one recording shows how much smaller the encoded path is despite carrying far
// more pixels.
inline constexpr std::uint32_t kEncodedImageWidth = 1920;
inline constexpr std::uint32_t kEncodedImageHeight = 1080;

// Emits a width x height RGB8 holoscan::schema::ImageT frame every tick: a black background with a
// white horizontal bar growing left-to-right as tick_count_ increases, resetting to empty every
// kResetPeriodTicks ticks (5s at the 20Hz tick rate below) -- a simple, human-visible "this stream
// is advancing" indicator when viewed in Foxglove's Image panel, without needing any text/glyph
// rendering (Holoscan SDK 5 does not provide that, and pulling in a font-rendering dependency for
// demo counter isn't worth it -- see this app's README for the fuller reasoning). Full image
// height, so the bar is unmistakable even at a small panel size.
//
// Dimensions and header.frame_id are constructor parameters so one graph can run several instances
// at different resolutions and still tell their frames apart in a viewer.
class ImageCounterOp final : public holoscan::Operator<> {
 public:
  static constexpr auto kPeriod =
      std::chrono::milliseconds(50);  // 20 Hz, matching the other streams.
  static constexpr std::uint32_t kResetPeriodTicks = 100;
  static constexpr std::size_t kBytesPerPixel = 3;  // RGB8.

  ImageCounterOp(std::uint32_t width, std::uint32_t height, std::string frame_id)
      : width_(width), height_(height), frame_id_(std::move(frame_id)) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(out_image, "out_image"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kPeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    // Scaled by this instance's own width, so the bar spans the same fraction of the frame at
    // every resolution rather than being sized for one of them.
    const std::uint32_t bar_width_px =
        (tick_count_ % kResetPeriodTicks) * width_ / kResetPeriodTicks;
    ++tick_count_;

    const std::size_t frame_bytes = static_cast<std::size_t>(width_) * height_ * kBytesPerPixel;

    // Same buffer-ownership pattern as test_mcap_logger_tensor_schema_integration.cc's
    // ImageGeneratorOp: a heap buffer this Tensor exclusively owns via a release function.
    // Held in a unique_ptr up to the make_tensor() call so the failure path frees it
    // automatically; released to a raw pointer only once make_tensor() has taken ownership.
    auto pixels = std::make_unique<std::uint8_t[]>(frame_bytes);
    std::memset(pixels.get(), 0, frame_bytes);  // Black background.
    for (std::uint32_t row = 0; row < height_; ++row) {
      std::uint8_t* row_start =
          pixels.get() + (static_cast<std::size_t>(row) * width_ * kBytesPerPixel);
      std::memset(row_start, 0xFF, static_cast<std::size_t>(bar_width_px) * kBytesPerPixel);
    }

    const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(frame_bytes)};
    auto tensor = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                                        holoscan::MemoryStorageType::kSystem, pixels.get(),
                                        [](void* ptr) -> holoscan::expected<void, holoscan::Error> {
                                          delete[] static_cast<std::uint8_t*>(ptr);
                                          return {};
                                        });
    if (!tensor) {
      return holoscan::make_unexpected(tensor.error());
    }
    pixels.release();  // make_tensor() took ownership; its release lambda above frees this now.

    holoscan::schema::ImageT value;
    value.header = std::make_shared<holoscan::schema::HeaderT>();
    value.header->frame_id = frame_id_;
    // Real wall-clock time, not a tick-derived value -- see CircularMotionOp's compute() comment
    // for why: Foxglove interprets this field as absolute Unix epoch nanoseconds. Left entirely
    // unset here previously, which defaults to 0 on decode -- same wrong "just after the Unix
    // epoch" symptom as a tick-derived value, just via a different root cause (a missing header
    // rather than a wrong one).
    value.header->capture_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::system_clock::now().time_since_epoch())
                                             .count();
    value.height = static_cast<std::int32_t>(height_);
    value.width = static_cast<std::int32_t>(width_);
    value.encoding = holoscan::schema::ImageEncoding_RGB8;
    value.data = std::make_unique<holoscan::Tensor>(std::move(*tensor));

    return out_image.emit(value);
  }

  holoscan::Output<holoscan::schema::ImageT> out_image;

 private:
  std::uint32_t width_;
  std::uint32_t height_;
  std::string frame_id_;
  std::uint32_t tick_count_ = 0;
};

}  // namespace isaac::os::mcap_logger_reference
