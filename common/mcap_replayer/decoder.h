/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "holoscan/core/domain/dlpack_utils.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/memory/tensor_storage.hpp"
#include "holoscan/schema/schema_package.hpp"
#include <flatbuffers/flatbuffers.h>

#include "common/mcap_logger/tensor_dtype.h"
#include "common/mcap_replayer/message.h"
#include "tensor_frame_generated.h"

namespace isaac::os::mcap_replayer {

// One topic's message builder: turns a recorded payload into the Message its port publishes. It
// is closed over the type it builds for and nothing else, so the prefetch thread calls it knowing
// nothing about types and it never touches a port; make_message_builder<T>() and
// make_tensor_message_builder<T>() make the two kinds.
//
// A failure (malformed bytes for the declared T) is not fatal: it returns why, in words fit for a
// terminal, rather than throwing. Reporting is the caller's, which knows the topic and how often it
// has already warned; a failure that happens once tends to recur at message rate.
//
// A topic recorded as encoded video arrives as two joined recorded messages, so a builder also
// receives the metadata half, empty for any other topic.
using MessageBuilder = std::function<holoscan::expected<std::unique_ptr<Message>, std::string>(
    std::span<const std::byte> payload, std::span<const std::byte> sidecar,
    std::uint64_t log_time_ns)>;

namespace detail {

// How many bytes a recorded shape and element type together describe, or no value if that product
// overflows: a wrapped-around, too-small count could otherwise appear to agree with the data
// actually carried. Each multiplication is guarded before it happens, in 64 bits regardless of
// std::size_t's width.
[[nodiscard]] inline std::optional<std::size_t> byte_count_of(
    const flatbuffers::Vector<std::uint32_t>& shape, DLDataType dtype) {
  std::uint64_t element_count = 1;
  for (const std::uint32_t dimension : shape) {
    if (dimension != 0 && element_count > std::numeric_limits<std::uint64_t>::max() /
                                              static_cast<std::uint64_t>(dimension)) {
      return std::nullopt;
    }
    element_count *= dimension;
  }
  const std::uint64_t element_bytes = holoscan::dlpack_dtype_item_size_bytes(dtype);
  if (element_bytes != 0 &&
      element_count > std::numeric_limits<std::uint64_t>::max() / element_bytes) {
    return std::nullopt;
  }
  const std::uint64_t total_bytes = element_count * element_bytes;
  if (total_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(total_bytes);
}

// Wraps a host buffer as a tensor that owns it, or says why it could not. The buffer is released
// to the tensor only once make_tensor() succeeds, so the failure path frees it.
[[nodiscard]] inline holoscan::expected<holoscan::Tensor, std::string_view> adopt_host_buffer(
    const std::vector<std::int64_t>& dimensions, DLDataType element_type,
    std::unique_ptr<std::byte[]> buffer) {
  auto tensor = holoscan::make_tensor(
      dimensions, element_type, std::nullopt, holoscan::MemoryStorageType::kSystem, buffer.get(),
      [](void* pointer) -> holoscan::expected<void, holoscan::Error> {
        delete[] static_cast<std::byte*>(pointer);
        return {};
      });
  if (!tensor) {
    return holoscan::make_unexpected(
        std::string_view("one of its tensors could not be rebuilt in memory"));
  }
  buffer.release();  // make_tensor() took ownership; its release function above frees this now.
  return std::move(*tensor);
}

// Rebuilds one recorded tensor in host memory, or says in plain language why it could not be, so
// a single log line tells the reader what is wrong with the recording.
[[nodiscard]] inline holoscan::expected<holoscan::Tensor, std::string_view> rebuild_tensor(
    const ::isaac::mcap_logger::TensorEntry* entry) {
  if (entry == nullptr) {
    return holoscan::make_unexpected(std::string_view("one of its tensors is missing"));
  }
  const std::optional<DLDataType> element_type = mcap_logger::FromTensorDType(entry->dtype());
  if (!element_type.has_value()) {
    return holoscan::make_unexpected(
        std::string_view("it was recorded with a kind of number this application cannot read"));
  }
  const auto* shape = entry->shape();
  const auto* data = entry->data();
  if (shape == nullptr || data == nullptr) {
    return holoscan::make_unexpected(
        std::string_view("one of its tensors has no size or no contents"));
  }

  // The recorded format does not tie the shape and element type to the data length, so a
  // malformed or hostile recording can disagree. Checked before allocating, since otherwise the
  // tensor's reported size could reach past its own allocation.
  const std::optional<std::size_t> expected_bytes = byte_count_of(*shape, *element_type);
  if (!expected_bytes.has_value()) {
    return holoscan::make_unexpected(
        std::string_view("one of its tensors claims more data than can be counted"));
  }
  if (*expected_bytes != data->size()) {
    return holoscan::make_unexpected(
        std::string_view("one of its tensors is a different size than it says it is"));
  }

  const std::vector<std::int64_t> dimensions(shape->begin(), shape->end());
  const std::size_t byte_count = data->size();
  // Left uninitialized because the copy below fills all of it; make_unique would zero it first.
  auto buffer = std::make_unique_for_overwrite<std::byte[]>(byte_count);
  std::memcpy(buffer.get(), data->data(), byte_count);
  return adopt_host_buffer(dimensions, *element_type, std::move(buffer));
}

}  // namespace detail

// Builds the message builder for one plain Topic<T>: the recorded flatbuffer, materialized as T.
template <typename T>
MessageBuilder make_message_builder() {
  return
      [](std::span<const std::byte> payload, std::span<const std::byte> /*sidecar*/,
         std::uint64_t log_time_ns) -> holoscan::expected<std::unique_ptr<Message>, std::string> {
        auto value = holoscan::schema_package<T>().materialize(payload);
        if (!value) {
          return holoscan::make_unexpected(holoscan::runtime_error_message(value.error()));
        }

        auto message = std::make_unique<FlatbufferMessage<T>>();
        message->value = std::move(*value);
        message->log_time_ns = log_time_ns;
        return message;
      };
}

// The builder for a topic recorded in mcap_logger's tensor envelope (the isaac.mcap_logger.
// TensorFrame wire schema). It rebuilds the tensors but passes the recorded metadata bytes through
// unchanged instead of materializing T, because the publisher (TypedTensorPort<T>::publisher)
// takes metadata bytes and a tensor list directly. Like make_message_builder<T>(), a malformed
// recording yields the reason rather than an entry, and never throws.
//
// T must be a tensor-schema type. Every tensor is rebuilt in host memory: the
// recording does not carry the memory kind or device it came from.
template <typename T>
  requires holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>
MessageBuilder make_tensor_message_builder() {
  return
      [](std::span<const std::byte> payload, std::span<const std::byte> /*sidecar*/,
         std::uint64_t log_time_ns) -> holoscan::expected<std::unique_ptr<Message>, std::string> {
        // The bytes come straight off disk unchecked, so verify them before walking: a truncated or
        // corrupt message would otherwise be read past the end of the buffer. This also rejects a
        // message that is not a tensor frame at all.
        flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t*>(payload.data()),
                                       payload.size());
        if (!::isaac::mcap_logger::VerifyTensorFrameBuffer(verifier)) {
          return holoscan::make_unexpected(
              std::string("it is damaged, or is not a recording of tensor data at all"));
        }
        const auto* frame = ::isaac::mcap_logger::GetTensorFrame(payload.data());
        if (frame == nullptr) {
          return holoscan::make_unexpected(
              std::string("it is damaged, or is not a recording of tensor data at all"));
        }

        const auto* metadata = frame->metadata();
        if (metadata == nullptr) {
          return holoscan::make_unexpected(
              std::string("it does not say what the tensor data it carries describes"));
        }

        const auto* entries = frame->tensors();
        const std::size_t recorded_count = entries != nullptr ? entries->size() : 0U;
        // Must equal T's tensor count exactly: fewer would leave a field empty yet report success,
        // and more would silently drop recorded data.
        constexpr auto kFieldCount =
            static_cast<std::size_t>(holoscan::DescriptorTensorSchemaNativeTraits<T>::tensor_count);
        if (recorded_count != kFieldCount) {
          return holoscan::make_unexpected(
              std::string("it carries a different number of tensors than it is being read as"));
        }

        std::vector<holoscan::Tensor> tensors;
        tensors.reserve(recorded_count);
        for (const auto* entry : *entries) {
          holoscan::expected<holoscan::Tensor, std::string_view> tensor =
              detail::rebuild_tensor(entry);
          if (!tensor) {
            return holoscan::make_unexpected(std::string(tensor.error()));
          }
          tensors.push_back(std::move(*tensor));
        }

        auto pending = std::make_unique<TensorMessage>();
        pending->metadata.assign(
            reinterpret_cast<const std::byte*>(metadata->data()),
            reinterpret_cast<const std::byte*>(metadata->data()) + metadata->size());
        pending->tensors = std::move(tensors);
        pending->log_time_ns = log_time_ns;
        return pending;
      };
}

}  // namespace isaac::os::mcap_replayer
