/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include <cuda_runtime.h>
#include <flatbuffers/flatbuffers.h>
#include <fmt/format.h>

#include "common/mcap_logger/tensor_dtype.h"
#include "tensor_frame_generated.h"

// Builds the TensorFrame FlatBuffers message a tensor-schema topic is recorded as: every tensor
// materialized to host bytes, plus the metadata and the schemas that let a reader decode it. Kept
// apart from the poller so that one can stay about polling, submitting and writing.
namespace isaac::os::mcap_logger {

// nullopt if any dimension doesn't fit uint32_t (tensor_frame.fbs's TensorEntry.shape is
// [uint32], not [int64] -- no tensor this component has ever handled comes near 2^32 in a single
// dimension, but this is what makes an impossible one a clean skip instead of silent truncation).
[[nodiscard]] inline std::optional<std::vector<std::uint32_t>> ToUint32Shape(
    const std::vector<std::int64_t>& shape) {
  std::vector<std::uint32_t> result;
  result.reserve(shape.size());
  for (const std::int64_t dim : shape) {
    if (dim < 0 || dim > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
      return std::nullopt;
    }
    result.push_back(static_cast<std::uint32_t>(dim));
  }
  return result;
}

// A failed build, with the reason the caller will report. The text is only ever made on a failure.
[[nodiscard]] inline holoscan::unexpected<holoscan::Error> TensorFailure(std::string reason) {
  return holoscan::make_unexpected(
      holoscan::Error{holoscan::ErrorCode::kInvalidArgument, std::move(reason)});
}

// One tensor as a TensorEntry, or the reason it cannot be one. The bytes are copied to the host
// whichever kind of memory the tensor is in.
[[nodiscard]] inline holoscan::expected<flatbuffers::Offset<::isaac::mcap_logger::TensorEntry>,
                                        holoscan::Error>
BuildTensorEntry(flatbuffers::FlatBufferBuilder& builder, const holoscan::Tensor& tensor) {
  const std::optional<::isaac::mcap_logger::TensorDType> dtype = ToTensorDType(tensor.dtype());
  if (!dtype) {
    return TensorFailure("has an unsupported dtype");
  }
  const std::optional<std::vector<std::uint32_t>> shape = ToUint32Shape(tensor.shape());
  if (!shape) {
    return TensorFailure("has a shape dimension too large for uint32_t");
  }

  const auto nbytes = static_cast<std::size_t>(tensor.nbytes());
  // A tensor that reports bytes but has nowhere to read them from cannot be copied, on the host or
  // the device, so it is refused before anything is added to the builder.
  if (nbytes != 0 && tensor.data() == nullptr) {
    return TensorFailure(fmt::format("reports {} bytes but has no data pointer", nbytes));
  }
  // Force alignment to the tensor's own element size (not just the default 1-byte
  // vector alignment) so a future reader can reinterpret this vector's bytes as
  // dtype's native C++ type directly, e.g. a float* over Float32 data, without that
  // cast being undefined behavior. itemsize() -- not a hand-rolled dtype table --
  // since it's already exactly bytes-per-element for the same DLDataType ToTensorDType
  // just validated above.
  builder.ForceVectorAlignment(nbytes, sizeof(std::uint8_t),
                               static_cast<std::size_t>(tensor.itemsize()));
  std::uint8_t* data = nullptr;
  const auto data_offset = builder.CreateUninitializedVector<std::uint8_t>(nbytes, &data);

  const auto device_type = tensor.device().device_type;
  if (device_type == kDLCUDA || device_type == kDLCUDAManaged) {
    const auto copy_type =
        device_type == kDLCUDAManaged ? cudaMemcpyDefault : cudaMemcpyDeviceToHost;
    const cudaError_t copied = cudaMemcpy(data, tensor.data(), nbytes, copy_type);
    if (copied != cudaSuccess) {
      return TensorFailure(cudaGetErrorString(copied));
    }
  } else if (device_type == kDLCPU || device_type == kDLCUDAHost) {
    std::memcpy(data, tensor.data(), nbytes);
  } else {
    return TensorFailure(
        fmt::format("its memory is on an unsupported device (DLPack device type {})",
                    static_cast<int>(device_type)));
  }

  return ::isaac::mcap_logger::CreateTensorEntry(builder, builder.CreateVector(*shape), *dtype,
                                                 data_offset);
}

// A byte span as the [ubyte] vector the schema declares.
//
// FlatBuffers treats any enum as a scalar, and std::byte is one, so CreateVector() takes the bytes
// exactly as they are and copies them the way it copies any scalar array -- no pointer
// reinterpretation. Only the resulting offset is retyped, from a vector of std::byte to the vector
// of uint8_t that the generated table asks for. The two are laid out identically: a length, then
// the bytes.
[[nodiscard]] inline flatbuffers::Offset<flatbuffers::Vector<std::uint8_t>> CreateByteVector(
    flatbuffers::FlatBufferBuilder& builder, std::span<const std::byte> bytes) {
  const auto vector = builder.CreateVector(bytes.data(), bytes.size());
  return flatbuffers::Offset<flatbuffers::Vector<std::uint8_t>>(vector.o);
}

// The whole frame into `builder`, finished and ready to write: every tensor of `bundle`, its
// metadata, and `schema_name` with `metadata_schema` (the metadata type's compiled reflection
// schema) so a reader can decode the metadata without knowing the type. Fails on the first tensor
// that cannot be materialized, leaving `builder` unfinished.
template <typename Bundle>
[[nodiscard]] holoscan::expected<void, holoscan::Error> BuildTensorFrame(
    flatbuffers::FlatBufferBuilder& builder, Bundle& bundle, std::string_view schema_name,
    std::span<const std::byte> metadata_schema) {
  std::vector<flatbuffers::Offset<::isaac::mcap_logger::TensorEntry>> entry_offsets;
  entry_offsets.reserve(bundle.tensors.size());
  for (const auto& tensor : bundle.tensors) {
    const auto entry = BuildTensorEntry(builder, tensor);
    if (!entry) {
      return holoscan::make_unexpected(entry.error());
    }
    entry_offsets.push_back(*entry);
  }

  const auto tensors_offset = builder.CreateVector(entry_offsets);
  const auto metadata_bytes = bundle.metadata_bytes();
  // metadata is itself a serialized FlatBuffers message (T's own metadata table, e.g.
  // holoscan.schema.Image) nested inside this one -- force it onto FlatBuffers' maximum
  // possible scalar alignment (sizeof(uint64_t)), a safe upper bound for embedding any
  // nested FlatBuffers message. (We don't have the builder that produced metadata_bytes
  // here, only its serialized bytes, so we can't query its actual GetBufferMinAlignment()
  // the way FlatBuffers' own CreateNestedFlatBuffer idiom does.)
  builder.ForceVectorAlignment(metadata_bytes.size(), sizeof(std::uint8_t), sizeof(std::uint64_t));
  const auto metadata_offset = CreateByteVector(builder, metadata_bytes);
  const auto schema_name_offset = builder.CreateString(schema_name.data(), schema_name.size());

  // metadata_binary_schema is T's compiled reflection schema (a serialized
  // reflection::Schema FlatBuffers message, per reflection.fbs) -- same nested-FlatBuffers
  // alignment reasoning as metadata above. Carrying this, not just metadata_schema_name, is
  // what lets a future reader that has never heard of T decode metadata generically via
  // FlatBuffers' own runtime reflection (flatbuffers/reflection.h), rather than needing T's
  // generated type linked in or (on the Foxglove/TypeScript side) hand-written,
  // type-specific decode code -- see decodeHoloscanSchemaImageMetadata()'s comment in
  // converters.ts for the one type this still hardcodes today. Consuming this field to
  // replace that hardcoding with a generic decoder is planned for a future MR.
  builder.ForceVectorAlignment(metadata_schema.size(), sizeof(std::uint8_t), sizeof(std::uint64_t));
  const auto binary_schema_offset = CreateByteVector(builder, metadata_schema);

  const auto frame_offset = ::isaac::mcap_logger::CreateTensorFrame(
      builder, tensors_offset, metadata_offset, schema_name_offset, binary_schema_offset);
  ::isaac::mcap_logger::FinishTensorFrameBuffer(builder, frame_offset);
  return {};
}

}  // namespace isaac::os::mcap_logger
