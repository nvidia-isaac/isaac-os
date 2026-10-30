/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <string_view>

#include "operators/image_conversion/nv12_layout.h"

int main() {
  using isaac::os::ops::detail::Nv12Layout;
  using isaac::os::ops::detail::validate_nv12_layout;

  constexpr Nv12Layout valid{
      .width = 4U,
      .height = 4U,
      .buffer_bytes = 28U,
      .y_offset = 4U,
      .y_pitch = 4U,
      .uv_offset = 20U,
      .uv_pitch = 4U,
  };
  if (validate_nv12_layout(valid)) return EXIT_FAILURE;

  Nv12Layout unequal_pitch = valid;
  unequal_pitch.uv_pitch = 8U;
  if (validate_nv12_layout(unequal_pitch) !=
      std::string_view{"The Y and UV plane pitches differ, which the selected NPP conversion "
                       "cannot accept."}) {
    return EXIT_FAILURE;
  }

  Nv12Layout truncated_uv = valid;
  truncated_uv.buffer_bytes = 27U;
  if (validate_nv12_layout(truncated_uv) !=
      std::string_view{"The UV plane layout extends beyond the camera buffer."}) {
    return EXIT_FAILURE;
  }

  Nv12Layout zero_width = valid;
  zero_width.width = 0U;
  zero_width.y_pitch = 0U;
  zero_width.uv_pitch = 0U;
  if (!validate_nv12_layout(zero_width)) return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
