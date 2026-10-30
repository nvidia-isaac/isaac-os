/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_IMAGE_CONVERSION_NV12_LAYOUT_H
#define ISAAC_OS_OPERATORS_IMAGE_CONVERSION_NV12_LAYOUT_H

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace isaac::os::ops::detail {

struct Nv12Layout {
  std::uint32_t width;
  std::uint32_t height;
  std::uint64_t buffer_bytes;
  std::uint64_t y_offset;
  std::uint64_t y_pitch;
  std::uint64_t uv_offset;
  std::uint64_t uv_pitch;
};

inline bool plane_fits(std::uint64_t offset, std::uint64_t pitch, std::uint64_t rows,
                       std::uint64_t row_bytes, std::uint64_t buffer_bytes) {
  if (rows == 0U || row_bytes == 0U || pitch == 0U || pitch < row_bytes || offset > buffer_bytes) {
    return false;
  }
  const std::uint64_t remaining = buffer_bytes - offset;
  if (rows - 1U > (std::numeric_limits<std::uint64_t>::max() - row_bytes) / pitch) return false;
  return (rows - 1U) * pitch + row_bytes <= remaining;
}

inline std::optional<std::string_view> validate_nv12_layout(const Nv12Layout& layout) {
  if (layout.y_pitch != layout.uv_pitch) {
    return "The Y and UV plane pitches differ, which the selected NPP conversion cannot accept.";
  }
  if (layout.y_pitch > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return "The NV12 plane pitch exceeds the range accepted by NPP.";
  }
  if (!plane_fits(layout.y_offset, layout.y_pitch, layout.height, layout.width,
                  layout.buffer_bytes)) {
    return "The Y plane layout extends beyond the camera buffer.";
  }
  if (!plane_fits(layout.uv_offset, layout.uv_pitch, layout.height / 2U, layout.width,
                  layout.buffer_bytes)) {
    return "The UV plane layout extends beyond the camera buffer.";
  }
  return std::nullopt;
}

}  // namespace isaac::os::ops::detail

#endif  // ISAAC_OS_OPERATORS_IMAGE_CONVERSION_NV12_LAYOUT_H
