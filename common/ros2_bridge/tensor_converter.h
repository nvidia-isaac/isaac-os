/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sensor_msgs/msg/image.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/channel/tensor_schema_endpoints.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/core/external_topic.hpp"
#include "holoscan/memory/tensor_storage.hpp"
#include "holoscan/schema/descriptor_flatbuffer_payload.hpp"
#include "holoscan/sensor_io/image_metadata.hpp"
#include "holoscan/sensor_io/sensor_schema_package.hpp"
#include <fmt/format.h>

#include "common/image_schema/image_schema.h"
#include "common/ros2_bridge/image_encoding.h"
#include "common/ros2_bridge/memory_transfer.h"
#include "common/ros2_bridge/ros2_timestamp.h"
#include "common/ros2_bridge/ros2_topic_spec.h"
#include "common/ros2_bridge/ros_buffer_egress.h"
#include "common/ros2_bridge/ros_buffer_ingress.h"

namespace isaac::os::ros2_bridge {

// Undefined primary template: instantiating TensorConverter<Ros2T, HsdkT> for a pair with no
// specialization below is a compile error, not a silent runtime gap -- same discipline as
// Converter in converter.h.
template <typename Ros2T, typename HsdkT>
struct TensorConverter;

namespace detail {

// Keeps a ROS2 Image message alive for as long as any holoscan::Tensor built from its `data`
// buffer survives. rclcpp hands a subscription callback a shared_ptr specifically so it can be
// retained past the callback returning; retaining the shared_ptr itself here (not just a raw
// pointer into msg->data) is what stops ROS2 from freeing that buffer while a Tensor still
// references it, including after emit() returns -- the transport may hand the backing onward.
class Ros2ImageTensorBacking final : public holoscan::TensorMemoryReference {
 public:
  explicit Ros2ImageTensorBacking(std::shared_ptr<sensor_msgs::msg::Image> msg)
      : msg(std::move(msg)) {}

  std::shared_ptr<sensor_msgs::msg::Image> msg;
};

}  // namespace detail

// sensor_msgs/msg/Image <-> holoscan::schema::ImageT (the SDK's own descriptor-tensor-schema
// Image type, from holoscan::sensor_io -- not a project-local schema). height/width/encoding/
// data and header.stamp/frame_id round-trip; every other ImageT field (color_space, gain,
// plane_layouts, etc.) has no sensor_msgs/msg/Image equivalent and is left at its schema default.
template <>
struct TensorConverter<sensor_msgs::msg::Image, holoscan::schema::ImageT> {
  using Ros2Type = sensor_msgs::msg::Image;
  using HsdkType = holoscan::schema::ImageT;
  using Payload = holoscan::schema::DescriptorFlatBufferPayloadFor<holoscan::schema::Image>;

  // Not a per-message copy limit -- sizes the transport's preallocated backing capacity for the
  // `data` field. A frame exceeding it fails at emit() as a contract violation, not a silent
  // truncation; an oversized bound only costs preallocated headroom.
  static constexpr std::uint64_t kDefaultMaxTensorBytes = 8U * 1024U * 1024U;  // 8 MiB.

  // common/image_schema/image_schema.h defines the field selector/boundary contract for
  // Image's `data` field (field_id=4, tensor_index=0) -- this is shared with
  // common/mcap_logger/mcap_logger.h's attach<T>() callers, which build the same contract
  // via the same ImageTensorFieldContract(). Only the byte bound differs per caller: this bridge
  // uses the ROS2 topic's own configured (or default) max_tensor_bytes, since to_tensors()'s
  // Tensor wraps plain heap-backed storage -- not natively HoloMQ-exportable -- which is also why
  // ImageTensorFieldContract()'s transfer policy is kCopyIfNeeded rather than the default
  // kRequireExportable.
  [[nodiscard]] static holoscan::DescriptorTensorTopicContract<HsdkType> contract(
      const Ros2TopicSpec& topic) {
    const holoscan::MemoryKind kind = topic.memory_kind == Ros2MemoryKind::kCudaDevice
                                          ? holoscan::MemoryKind::kCudaDevice
                                          : holoscan::MemoryKind::kHost;
    return isaac::os::image_schema::ImageTensorFieldContract(
        topic.max_tensor_bytes.value_or(kDefaultMaxTensorBytes), kind);
  }

  // Takes the callback's own shared_ptr (not a dereferenced reference) so the Tensor built below
  // can keep the message alive -- required for correctness, not just style. In a CUDA build the
  // tensor RosBufferToTensor returns holds it, through its release function or device backing; in a
  // host-only build Ros2ImageTensorBacking above does (see that class's own comment for why).
  [[nodiscard]] static holoscan::expected<
      std::pair<std::vector<std::byte>, std::vector<holoscan::Tensor>>, holoscan::Error>
  to_tensors(const std::shared_ptr<sensor_msgs::msg::Image>& msg,
             [[maybe_unused]] const Ros2TopicSpec& spec) {
    auto encoding = detail::ros2_encoding_to_hsdk(msg->encoding);
    if (!encoding) {
      return holoscan::make_unexpected(std::move(encoding).error());
    }
    // ros2_encoding_to_hsdk() above and image_schema::bytes_per_pixel() are two independently
    // maintained tables of the same 5 supported encodings; checking this explicitly (rather than
    // assuming it can't fail here) means a future divergence between them surfaces as this exact
    // error instead of a confusing downstream size mismatch against a silently-wrong bound.
    auto bpp = isaac::os::image_schema::bytes_per_pixel(*encoding);
    if (!bpp) {
      return holoscan::make_unexpected(std::move(bpp).error());
    }

    // image.fbs requires Image's Tensor to address exactly one packed frame -- reject a buffer
    // that's too short (incomplete frame) or too long (trailing bytes would silently ride along,
    // then round-trip back out as a size mismatch in from_tensors_for_root()). height*width*bpp
    // is computed with overflow checks since height/width are attacker/producer-controlled
    // uint32s whose product, times bpp, can exceed a 64-bit byte count.
    std::size_t required_bytes = 0;
    if (__builtin_mul_overflow(static_cast<std::size_t>(msg->height),
                               static_cast<std::size_t>(msg->width), &required_bytes) ||
        __builtin_mul_overflow(required_bytes, static_cast<std::size_t>(*bpp), &required_bytes)) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          fmt::format("sensor_msgs/msg/Image {}x{} at {} bytes/pixel overflows a 64-bit byte "
                      "count.",
                      msg->height, msg->width, *bpp)});
    }
#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
    // Runs before the size check: a publisher on another GPU arrives as an empty CUDA buffer, and
    // the generic size error below would hide that cause.
    if (required_bytes != 0U && msg->data.size() == 0U && msg->data.get_backend_type() == "cuda") {
      if (spec.memory_kind == Ros2MemoryKind::kCudaDevice && !PoolDeviceBound()) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, detail::PoolUnboundMessage()});
      }
      return detail::EmptyCudaBufferError(spec.port_name);
    }
#endif
    if (msg->data.size() != required_bytes) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          fmt::format("sensor_msgs/msg/Image data size does not match its height/width/encoding: "
                      "got {} bytes, need exactly {} bytes for a {}x{} frame at '{}'.",
                      msg->data.size(), required_bytes, msg->height, msg->width, msg->encoding)});
    }

    // A set (nonzero) `step` must match a packed row -- image.fbs's Tensor has no per-row stride
    // field, so a wider (padded) step would silently ship row-shifted pixels.
    const std::uint32_t packed_step = static_cast<std::uint32_t>(msg->width) * (*bpp);
    if (msg->step != 0 && msg->step != packed_step) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          fmt::format("sensor_msgs/msg/Image step {} does not match a packed row for a {}-wide "
                      "frame at '{}' ({} bytes); padded rows are not supported by the ROS2 "
                      "bridge.",
                      msg->step, msg->width, msg->encoding, packed_step)});
    }

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
    // The helper wraps a buffer where it is when it already matches the topic's memory kind (no
    // copy, the tensor keeps the message alive) and copies it otherwise.
    auto tensor = RosBufferToTensor(msg, msg->data, spec);
#else
    auto backing = std::make_shared<detail::Ros2ImageTensorBacking>(msg);
    void* const data = msg->data.data();
    const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(msg->data.size())};
    auto tensor =
        holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                              holoscan::MemoryStorageType::kSystem, data, std::move(backing));
#endif
    if (!tensor) {
      return holoscan::make_unexpected(std::move(tensor).error());
    }

    // header.stamp maps onto capture_timestamp_ns (documented as the START of a sensor's
    // integration interval, which for a ROS2 producer is header.stamp); everything else in
    // ImageMetadataOptions is left at its schema default. Neither side is clock-tagged, so this
    // is a bit reinterpretation, not a clock translation: it assumes the ROS2-side clock matches
    // whatever the Holoscan side expects (not necessarily wall-clock). If that assumption doesn't
    // hold, timestamps round-trip bit-for-bit but are silently meaningless to compare.
    holoscan::schema::ImageMetadataOptions options;
    options.height = msg->height;
    options.width = msg->width;
    options.encoding = *encoding;
    options.capture_timestamp_ns =
        static_cast<std::int64_t>(msg->header.stamp.sec) * kNanosecondsPerSecond +
        static_cast<std::int64_t>(msg->header.stamp.nanosec);
    options.frame_id = msg->header.frame_id;
    auto metadata_bytes = holoscan::schema::build_image_metadata(*tensor, options);
    if (!metadata_bytes) {
      return holoscan::make_unexpected(std::move(metadata_bytes).error());
    }

    std::vector<holoscan::Tensor> tensors{std::move(*tensor)};
    return std::pair{std::move(*metadata_bytes), std::move(tensors)};
  }

  // The egress relay's entry point: reads root()/tensors off an already-resolved
  // ChannelResolvedTensorBundle, so no parse_image_metadata/image_tensor_metadata call is needed.
  [[nodiscard]] static holoscan::expected<sensor_msgs::msg::Image, holoscan::Error> from_tensors(
      const holoscan::ChannelResolvedTensorBundle<Payload>& bundle, const Ros2TopicSpec& spec) {
    return from_tensors_for_root(bundle.root(), bundle.tensors, spec);
  }

  // Split out from from_tensors() so tests can exercise this logic without a live
  // ChannelResolvedTensorBundle. `root` is nullable (empty metadata); returning expected<> lets
  // the poll-thread caller drop-and-log a bad sample instead of crashing on a null dereference.
  [[nodiscard]] static holoscan::expected<sensor_msgs::msg::Image, holoscan::Error>
  from_tensors_for_root(const holoscan::schema::Image* root,
                        std::span<const holoscan::Tensor> tensors,
                        [[maybe_unused]] const Ros2TopicSpec& spec = Ros2TopicSpec{}) {
    if (root == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "Received holoscan.schema.Image metadata with no root (empty metadata)."});
    }

    const std::string encoding = detail::hsdk_encoding_to_ros2(root->encoding());
    if (encoding.empty()) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          fmt::format("Unsupported holoscan.schema.Image encoding for the ROS2 bridge: {}.",
                      static_cast<int>(root->encoding()))});
    }
    // hsdk_encoding_to_ros2() above and image_schema::bytes_per_pixel() are two independently
    // maintained tables of the same 5 supported encodings; checking this explicitly (rather than
    // assuming it can't fail here) means a future divergence between them surfaces as this exact
    // error instead of a confusing downstream size mismatch against a silently-wrong bound.
    auto bpp = isaac::os::image_schema::bytes_per_pixel(root->encoding());
    if (!bpp) {
      return holoscan::make_unexpected(std::move(bpp).error());
    }

    if (tensors.empty()) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "Received holoscan.schema.Image with no tensors (expected exactly one, the `data` "
          "field)."});
    }

    const holoscan::Tensor& tensor = tensors[0];
    // Mirrors to_tensors()'s exact-size check (including the overflow guard) on the way back
    // out: this bundle may not have come from this bridge's own to_tensors, so a byte-count
    // mismatch must still be rejected here.
    std::size_t expected_bytes = 0;
    if (__builtin_mul_overflow(static_cast<std::size_t>(root->height()),
                               static_cast<std::size_t>(root->width()), &expected_bytes) ||
        __builtin_mul_overflow(expected_bytes, static_cast<std::size_t>(*bpp), &expected_bytes)) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          fmt::format("holoscan.schema.Image {}x{} at {} bytes/pixel overflows a 64-bit byte "
                      "count.",
                      root->height(), root->width(), *bpp)});
    }
    if (tensor.nbytes() != expected_bytes) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          fmt::format("holoscan.schema.Image tensor size does not match its height/width/"
                      "encoding: got {} bytes, need exactly {} bytes for a {}x{} frame.",
                      tensor.nbytes(), expected_bytes, root->height(), root->width())});
    }

    sensor_msgs::msg::Image out;
    // header is optional in image.fbs, so metadata built without one (e.g. by a non-ROS2
    // producer) leaves out.header zero-initialized rather than erroring.
    if (const holoscan::schema::Header* const header = root->header(); header != nullptr) {
      const auto stamp = to_ros2_timestamp(header->capture_timestamp_ns());
      if (!stamp) {
        return holoscan::make_unexpected(stamp.error());
      }
      out.header.stamp.sec = stamp->sec;
      out.header.stamp.nanosec = stamp->nanosec;
      if (const flatbuffers::String* const frame_id = header->frame_id(); frame_id != nullptr) {
        out.header.frame_id = frame_id->str();
      }
    }
    out.height = root->height();
    out.width = root->width();
    out.encoding = encoding;
    out.is_bigendian = 0U;
    out.step = out.width * (*bpp);
#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
    // The helper picks the copy for the topic's memory kind and where the tensor is: a host copy,
    // a copy down from the GPU, or a copy into a pooled CUDA buffer.
    if (auto copied = TensorToRosBuffer(tensor, spec, out.data); !copied) {
      return holoscan::make_unexpected(std::move(copied).error());
    }
    return out;
#else
    const auto* const bytes = tensor.data_as<std::uint8_t>();
    out.data.assign(bytes, bytes + tensor.nbytes());
    return out;
#endif
  }
};

}  // namespace isaac::os::ros2_bridge
