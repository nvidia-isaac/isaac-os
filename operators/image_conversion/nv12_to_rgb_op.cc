/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/image_conversion/nv12_to_rgb_op.h"

#include <nppi_color_conversion.h>

#include <array>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/provenance.hpp>

#include "operators/image_conversion/npp_context.h"
#include "operators/image_conversion/nv12_layout.h"

namespace isaac::os::ops {

namespace {

constexpr std::uint32_t kNv12OutputChannels = 3U;
constexpr DLDataType kNv12Dtype{static_cast<std::uint8_t>(kDLUInt), 8U, 1U};

holoscan::Error invalid_nv12_image(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, std::move(message)};
}

}  // namespace

Nv12ToRgbOp::Nv12ToRgbOp(std::uint32_t max_width, std::uint32_t max_height)
    : max_width_(max_width),
      max_height_(max_height),
      npp_context_(std::make_unique<detail::NppContext>()) {
  if (max_width == 0U || max_height == 0U ||
      max_width > std::numeric_limits<int>::max() / kNv12OutputChannels ||
      max_height > std::numeric_limits<std::int32_t>::max()) {
    throw std::invalid_argument("Image dimensions must be positive and fit NPP's integer limits.");
  }
}

Nv12ToRgbOp::~Nv12ToRgbOp() = default;

void Nv12ToRgbOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(image, "in_image")
      .queue_depth(1U)
      .expects_tensor(holoscan::TensorInputSpec{
          .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice,
                             .dtype = kNv12Dtype}});
  spec.output(rgb_image, "out_image")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice,
                             .dtype = kNv12Dtype,
                             .rank = 3U},
          .bounds = holoscan::tensor_bounds(static_cast<std::uint64_t>(max_width_) * max_height_ *
                                            kNv12OutputChannels),
          .storage = holoscan::TensorOutputStorage::kRuntimePool});
}

holoscan::Contract Nv12ToRgbOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{image});
  return result;
}

holoscan::expected<void, holoscan::Error> Nv12ToRgbOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = image.receive();
  if (!sample) return holoscan::make_unexpected(std::move(sample).error());

  const holoscan::schema::ImageT& source = sample->data;
  if (source.encoding != holoscan::schema::ImageEncoding_NV12) {
    return holoscan::make_unexpected(invalid_nv12_image("The image is not NV12."));
  }
  if (source.width <= 0 || source.height <= 0 || source.width % 2 != 0 || source.height % 2 != 0 ||
      source.width > static_cast<std::int32_t>(max_width_) ||
      source.height > static_cast<std::int32_t>(max_height_)) {
    return holoscan::make_unexpected(invalid_nv12_image(std::format(
        "The NV12 image is not an even, positive extent within the configured {}x{} limit.",
        max_width_, max_height_)));
  }
  if (source.plane_layouts.size() != 2U) {
    return holoscan::make_unexpected(
        invalid_nv12_image("The NV12 image must describe one Y and one UV plane."));
  }
  const holoscan::Tensor* const tensor = source.data.get();
  if (tensor == nullptr || tensor->data() == nullptr || tensor->nbytes() <= 0) {
    return holoscan::make_unexpected(invalid_nv12_image("The NV12 image carries no pixel data."));
  }

  const std::uint64_t y_offset = source.plane_layouts[0].offset_bytes();
  const std::uint64_t uv_offset = source.plane_layouts[1].offset_bytes();
  const std::uint64_t y_pitch = source.plane_layouts[0].row_stride_bytes();
  const std::uint64_t uv_pitch = source.plane_layouts[1].row_stride_bytes();
  if (const auto error = detail::validate_nv12_layout({
          .width = static_cast<std::uint32_t>(source.width),
          .height = static_cast<std::uint32_t>(source.height),
          .buffer_bytes = static_cast<std::uint64_t>(tensor->nbytes()),
          .y_offset = y_offset,
          .y_pitch = y_pitch,
          .uv_offset = uv_offset,
          .uv_pitch = uv_pitch,
      })) {
    return holoscan::make_unexpected(invalid_nv12_image(std::string{*error}));
  }

  const std::array<std::int64_t, 3> shape{source.height, source.width, kNv12OutputChannels};
  const std::array<std::int64_t, 3> strides{
      source.width * static_cast<std::int64_t>(kNv12OutputChannels), kNv12OutputChannels, 1};
  auto loan = rgb_image.allocate_tensor({.shape = shape,
                                         .dtype = kNv12Dtype,
                                         .byte_strides = holoscan::TensorByteStridesSpan{strides}});
  if (!loan) return holoscan::make_unexpected(std::move(loan).error());
  auto writer = loan->write(context.cuda_stream());
  if (!writer) return holoscan::make_unexpected(std::move(writer).error());
  auto npp_context = npp_context_->for_stream(context);
  if (!npp_context) return holoscan::make_unexpected(std::move(npp_context).error());

  const auto* const base = static_cast<const Npp8u*>(tensor->data());
  const std::array<const Npp8u*, 2> planes{base + y_offset, base + uv_offset};
  // SIPL requests REC709_ER (full range): use 709HDTV, not limited-range 709CSC.
  const NppStatus status = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
      planes.data(), static_cast<int>(y_pitch), static_cast<Npp8u*>(writer->data()),
      static_cast<int>(strides[0]), NppiSize{source.width, source.height}, *npp_context);
  if (status != NPP_SUCCESS) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "Unable to convert the NV12 image to RGB (NPP status " +
                                           std::to_string(static_cast<int>(status)) + ")."});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) return holoscan::make_unexpected(std::move(committed).error());

  holoscan::schema::ImageT descriptor = source;
  descriptor.data.reset();
  descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
  descriptor.significant_bits = 0U;
  descriptor.plane_layouts.clear();
  return rgb_image.emit_tensor(std::move(*loan), descriptor,
                               holoscan::sensor_io::derive_from(*sample));
}

}  // namespace isaac::os::ops
