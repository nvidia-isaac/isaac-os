/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/core/external_topic.hpp"
#include "holoscan/sensor_io/image_metadata.hpp"
#include "holoscan/sensor_io/sensor_schema_package.hpp"

// holoscan.schema.Image building blocks shared by every caller that transports ImageT over a
// tensor-schema topic (currently mcap_logger and ros2_bridge). Deliberately not tied to either:
// neither MCAP nor ROS2 is a concept this header knows about.
namespace isaac::os::image_schema {

// Bytes per pixel for the 5 encodings sensor_msgs/msg/Image also names as plain lowercase
// strings (rgb8, rgba8, bgr8, bgra8, mono8). Every other ImageEncoding (multi-planar, float,
// Bayer, ...) has no single-plane, single-byte-per-component meaning this can compute, so it's
// reported as an error rather than a silently wrong zero.
[[nodiscard]] holoscan::expected<std::uint32_t, holoscan::Error> bytes_per_pixel(
    holoscan::schema::ImageEncoding encoding);

// Builds the holoscan.schema.Image `data` field's tensor contract: field_id=4 (image.fbs's
// `data: Tensor (id: 4)`), tensor_index=0, a contiguous uint8 byte buffer bounded by
// max_tensor_bytes, transferred by copy when the source Tensor isn't natively exportable.
// memory_kind constrains the exact physical storage the transport requires (defaults to
// ordinary host memory); pass e.g. MemoryKind::kCudaDevice for a producer whose Tensor is
// GPU-resident.
[[nodiscard]] holoscan::DescriptorTensorTopicContract<holoscan::schema::ImageT>
ImageTensorFieldContract(std::uint64_t max_tensor_bytes,
                         holoscan::MemoryKind memory_kind = holoscan::MemoryKind::kHost);

// Convenience overload for a caller that knows a frame's dimensions/encoding up front: computes
// max_tensor_bytes as width * height * bytes_per_pixel(encoding) and forwards to the overload
// above. Throws std::invalid_argument for an unsupported encoding, or if that computation
// overflows a 64-bit byte count. A caller that wants an explicit byte bound instead of this
// computed one should call the overload above directly.
[[nodiscard]] holoscan::DescriptorTensorTopicContract<holoscan::schema::ImageT>
ImageTensorFieldContract(std::uint32_t width, std::uint32_t height,
                         holoscan::schema::ImageEncoding encoding,
                         holoscan::MemoryKind memory_kind = holoscan::MemoryKind::kHost);

}  // namespace isaac::os::image_schema
