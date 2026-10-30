/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/image_schema/image_schema.h"

#include <stdexcept>

#include <fmt/format.h>

namespace isaac::os::image_schema {

holoscan::expected<std::uint32_t, holoscan::Error> bytes_per_pixel(
    holoscan::schema::ImageEncoding encoding) {
  switch (encoding) {
    case holoscan::schema::ImageEncoding_RGB8:
    case holoscan::schema::ImageEncoding_BGR8:
      return 3U;
    case holoscan::schema::ImageEncoding_RGBA8:
    case holoscan::schema::ImageEncoding_BGRA8:
      return 4U;
    case holoscan::schema::ImageEncoding_MONO8:
      return 1U;
    default:
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          fmt::format("Unsupported holoscan.schema.Image encoding ({}): only RGB8, BGR8, RGBA8, "
                      "BGRA8, and MONO8 have a defined bytes-per-pixel.",
                      static_cast<int>(encoding))});
  }
}

holoscan::DescriptorTensorTopicContract<holoscan::schema::ImageT> ImageTensorFieldContract(
    std::uint64_t max_tensor_bytes, holoscan::MemoryKind memory_kind) {
  holoscan::DescriptorTensorTopicContract<holoscan::schema::ImageT> result;
  result.field(
      holoscan::TensorFieldSelector<holoscan::schema::ImageT>{.field_id = 4U, .tensor_index = 0U},
      holoscan::TensorBoundaryContract{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = memory_kind,
                  .dtype = DLDataType{kDLUInt, 8U, 1U},
                  .rank = 1U,
                  .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              },
          .bounds = holoscan::tensor_bounds(max_tensor_bytes),
          .transfer =
              holoscan::TensorTransferPolicy{
                  .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
              },
      });
  return result;
}

holoscan::DescriptorTensorTopicContract<holoscan::schema::ImageT> ImageTensorFieldContract(
    std::uint32_t width, std::uint32_t height, holoscan::schema::ImageEncoding encoding,
    holoscan::MemoryKind memory_kind) {
  auto bpp = bytes_per_pixel(encoding);
  if (!bpp) {
    throw std::invalid_argument(fmt::format("Unable to size a camera frame: {}",
                                            holoscan::runtime_error_message(bpp.error())));
  }
  std::uint64_t max_tensor_bytes = 0;
  if (__builtin_mul_overflow(static_cast<std::uint64_t>(width), static_cast<std::uint64_t>(height),
                             &max_tensor_bytes) ||
      __builtin_mul_overflow(max_tensor_bytes, static_cast<std::uint64_t>(*bpp),
                             &max_tensor_bytes)) {
    throw std::invalid_argument(
        fmt::format("Unable to size a camera frame: {}x{} at {} bytes/pixel overflows a 64-bit "
                    "byte count.",
                    width, height, *bpp));
  }
  return ImageTensorFieldContract(max_tensor_bytes, memory_kind);
}

}  // namespace isaac::os::image_schema
