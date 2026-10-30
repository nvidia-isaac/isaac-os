/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <holoscan/logger/logger.hpp>

#include "operators/ucxx_ops/message_traits.h"

namespace isaac::os::ops {
namespace {

struct DecodeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

std::vector<float> decode_array(const holoscan::schema::Tensor* tensor, bool required) {
  if (!tensor) {
    if (required) throw std::runtime_error("The received message is missing a required tensor.");
    return {};
  }
  if (!tensor->data() || !tensor->dtype() || !tensor->device() || !tensor->shape() ||
      tensor->ndim() != 1U || tensor->shape()->size() != 1U || tensor->shape()->Get(0) < 0 ||
      static_cast<int>(tensor->dtype()->code()) != kDLFloat || tensor->dtype()->bits() != 32U ||
      tensor->dtype()->lanes() != 1U ||
      static_cast<int>(tensor->device()->device_type()) != kDLCPU ||
      (tensor->strides() && (tensor->strides()->size() != 1U || tensor->strides()->Get(0) != 1)) ||
      tensor->data()->size() % sizeof(float) != 0U ||
      static_cast<std::uint64_t>(tensor->shape()->Get(0)) !=
          tensor->data()->size() / sizeof(float)) {
    throw std::runtime_error(
        "The received tensor must be a contiguous CPU float32 vector with matching shape and "
        "bytes.");
  }
  std::vector<float> values(tensor->data()->size() / sizeof(float));
  if (!values.empty()) std::memcpy(values.data(), tensor->data()->data(), tensor->data()->size());
  return values;
}

flatbuffers::Offset<holoscan::schema::Tensor> encode_array(flatbuffers::FlatBufferBuilder& builder,
                                                           const std::vector<float>& values) {
  const auto data = builder.CreateVector(reinterpret_cast<const std::uint8_t*>(values.data()),
                                         values.size() * sizeof(float));
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(values.size())};
  const auto shape_offset = builder.CreateVector(shape.data(), shape.size());
  const holoscan::schema::DLDataType dtype(holoscan::schema::DLDataTypeCode_kDLFloat, 32, 1);
  const holoscan::schema::DLDevice device(holoscan::schema::DLDeviceType_kDLCPU, 0);
  return holoscan::schema::CreateTensor(builder, data, shape_offset, &dtype, &device, 1U);
}

template <typename Root>
std::int64_t header_timestamp(const Root& root) {
  if (!root.header()) {
    throw DecodeError("The received simulator message has no header.");
  }
  const auto* header = root.header();
  if (const auto capture = header->capture_timestamp_ns(); capture != 0) return capture;
  // Back-compat: `isaac.Header` legacy peers may populate only the deprecated
  // `stamp` field. `holoscan.schema.Header` has no `stamp` accessor, so the
  // guard limits the fallback to Roots whose Header carries one.
  if constexpr (requires { header->stamp(); }) {
    if (header->stamp()) return header->stamp()->time_ns();
  }
  throw DecodeError("The received simulator message has no timestamp.");
}

flatbuffers::Offset<isaac::Header> encode_header(flatbuffers::FlatBufferBuilder& builder,
                                                 std::int64_t timestamp_ns) {
  return isaac::CreateHeader(builder, {}, {}, timestamp_ns);
}

}  // namespace

isaac::JointStateCpuT UcxxMessageTraits<isaac::JointStateCpuT>::decode(const RootType& root) {
  NativeType result;
  result.position = decode_array(root.position(), true);
  result.velocity = decode_array(root.velocity(), true);
  result.effort = decode_array(root.effort(), false);
  if (result.position.size() != result.velocity.size() ||
      (!result.effort.empty() && result.effort.size() != result.position.size())) {
    throw std::runtime_error("The received joint-state arrays have different lengths.");
  }
  if (root.name()) {
    for (const auto* name : *root.name()) result.name.emplace_back(name->str());
  }
  return result;
}

std::int64_t UcxxMessageTraits<isaac::JointStateCpuT>::timestamp(const RootType& root) {
  return header_timestamp(root);
}

flatbuffers::Offset<isaac::JointCommand> UcxxMessageTraits<isaac::JointCommandCpuT>::encode(
    flatbuffers::FlatBufferBuilder& builder, const NativeType& message, std::int64_t timestamp_ns) {
  const auto position = encode_array(builder, message.position);
  const auto velocity = encode_array(builder, message.velocity);
  const auto effort = encode_array(builder, message.effort);
  const auto kp = encode_array(builder, message.kp);
  const auto kd = encode_array(builder, message.kd);
  const auto frame = builder.CreateString("sinus_motion");
  const auto header = isaac::CreateHeader(builder, {}, frame, timestamp_ns);
  return isaac::CreateJointCommand(builder, header, 0, position, velocity, effort, kp, kd);
}

flatbuffers::Offset<isaac::JointState> UcxxMessageTraits<isaac::JointStateCpuT>::encode(
    flatbuffers::FlatBufferBuilder& builder, const NativeType& message, std::int64_t timestamp_ns) {
  const auto position = encode_array(builder, message.position);
  const auto velocity = encode_array(builder, message.velocity);
  const auto effort = message.effort.empty() ? flatbuffers::Offset<holoscan::schema::Tensor>{}
                                             : encode_array(builder, message.effort);
  std::vector<flatbuffers::Offset<flatbuffers::String>> names;
  for (const auto& name : message.name) names.push_back(builder.CreateString(name));
  const auto name_vector = builder.CreateVector(names);
  const auto header = encode_header(builder, timestamp_ns);
  return isaac::CreateJointState(builder, header, name_vector, position, velocity, 0, effort);
}

isaac::JointCommandCpuT UcxxMessageTraits<isaac::JointCommandCpuT>::decode(const RootType& root) {
  NativeType result;
  result.position = decode_array(root.position(), true);
  result.velocity = decode_array(root.velocity(), true);
  result.effort = decode_array(root.effort(), true);
  result.kp = decode_array(root.kp(), true);
  result.kd = decode_array(root.kd(), true);
  return result;
}

std::int64_t UcxxMessageTraits<isaac::JointCommandCpuT>::timestamp(const RootType& root) {
  return header_timestamp(root);
}

holoscan::schema::ImuT UcxxMessageTraits<holoscan::schema::ImuT>::decode(const RootType& root) {
  // holoscan.schema.Imu declares orientation / angular_velocity /
  // linear_acceleration as inline FlatBuffers structs, but the vtable slot
  // can still mark them absent (the accessor returns nullptr). UnPackTo then
  // silently defaults an absent struct to zero, so a missing measurement
  // would decode as a {0,0,0} vector or a {0,0,0,0} quaternion whose norm
  // is zero -- downstream normalization would produce NaN and corrupt the
  // control path. Reject explicitly instead.
  if (!root.angular_velocity() || !root.linear_acceleration()) {
    throw std::runtime_error(
        "The received IMU message is missing angular_velocity or linear_acceleration.");
  }
  // Orientation validity is signaled via ImuFlags::ORIENTATION_VALID rather
  // than field presence, so absence is only an error when the flag asserts
  // the field is meaningful.
  if ((root.flags() & holoscan::schema::ImuFlags_ORIENTATION_VALID) != 0 && !root.orientation()) {
    throw std::runtime_error(
        "The received IMU message asserts ORIENTATION_VALID but has no orientation.");
  }
  NativeType result;
  root.UnPackTo(&result);
  return result;
}

std::int64_t UcxxMessageTraits<holoscan::schema::ImuT>::timestamp(const RootType& root) {
  return header_timestamp(root);
}

flatbuffers::Offset<holoscan::schema::Imu> UcxxMessageTraits<holoscan::schema::ImuT>::encode(
    flatbuffers::FlatBufferBuilder& builder, const NativeType& message, std::int64_t timestamp_ns) {
  // Fast path: the producer already carries a capture timestamp on `header`.
  if (message.header && message.header->capture_timestamp_ns != 0) {
    return RootType::Pack(builder, &message);
  }
  // Upstream producers (agile_wbc, mock IMU) leave `header` unset and carry
  // measurement time only in emit metadata. Copy the native message and
  // inject `timestamp_ns` so the packed buffer surfaces
  // Header.capture_timestamp_ns to the peer. Deep-copy the header if the
  // caller already owns one -- a plain `stamped = message` shares the same
  // shared_ptr<HeaderT> with the caller, so writing to capture_timestamp_ns
  // would silently mutate the caller's copy.
  NativeType stamped = message;
  if (!stamped.header) {
    stamped.header = std::make_shared<holoscan::schema::HeaderT>();
  } else {
    stamped.header = std::make_shared<holoscan::schema::HeaderT>(*stamped.header);
  }
  stamped.header->capture_timestamp_ns = timestamp_ns;
  return RootType::Pack(builder, &stamped);
}

holoscan::schema::ImageT UcxxMessageTraits<holoscan::schema::ImageT>::decode(const RootType& root) {
  NativeType result;
  root.UnPackTo(&result);
  return result;
}

std::int64_t UcxxMessageTraits<holoscan::schema::ImageT>::timestamp(const RootType& root) {
  return header_timestamp(root);
}

flatbuffers::Offset<holoscan::schema::Image> UcxxMessageTraits<holoscan::schema::ImageT>::encode(
    flatbuffers::FlatBufferBuilder& builder, const NativeType& message, std::int64_t timestamp_ns) {
  // Fast path: skip the deep copy (ImageT carries a device-backed tensor)
  // when the producer already put a capture timestamp on the header.
  if (message.header && message.header->capture_timestamp_ns != 0) {
    return RootType::Pack(builder, &message);
  }
  // Match the ImuT treatment: producers that carry time only in emit
  // metadata (not on message.header) would otherwise ship a buffer without
  // Header.capture_timestamp_ns, which the receive side now hard-requires.
  // Deep-copy the header so writing capture_timestamp_ns does not mutate
  // the caller's shared HeaderT (see ImuT::encode above).
  //
  // Log once so integration tests catch a producer that silently falls back
  // to this slow path when the fast path is expected. Pack does not embed
  // the pixel bytes (kHasSeparateTensor), so the cost is a small metadata
  // serialization -- acceptable, but worth surfacing.
  static std::atomic<bool> warned{false};
  if (!warned.exchange(true, std::memory_order_relaxed)) {
    HOLOSCAN_LOG_WARN(
        "ImageT encoder fell back to the slow path (Header.capture_timestamp_ns "
        "unset on the producer); consider setting it upstream.");
  }
  NativeType stamped = message;
  if (!stamped.header) {
    stamped.header = std::make_shared<holoscan::schema::HeaderT>();
  } else {
    stamped.header = std::make_shared<holoscan::schema::HeaderT>(*stamped.header);
  }
  stamped.header->capture_timestamp_ns = timestamp_ns;
  return RootType::Pack(builder, &stamped);
}

}  // namespace isaac::os::ops
