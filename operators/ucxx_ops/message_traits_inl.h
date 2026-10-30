/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>

// Newer SDKs expose built-in Time bindings separately from sensor schemas.
// Older SDKs register them through flatbuffer_schema_package.hpp instead.
#if __has_include(<holoscan/schema/sdk_schemas.hpp>)
#include <holoscan/schema/sdk_schemas.hpp>
#else
#include <holoscan/schema/flatbuffer_schema_package.hpp>
#endif
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "isaac_joint_command_generated.h"
#include "isaac_joint_state_generated.h"
#include "joint_command_cpu_schema_traits.h"
#include "joint_state_cpu_schema_traits.h"

namespace isaac::os::ops {

template <>
struct UcxxMessageTraits<isaac::JointStateCpuT> {
  using RootType = isaac::JointState;
  using NativeType = isaac::JointStateCpuT;
  static constexpr bool kHasSeparateTensor = false;
  static constexpr std::string_view kSchemaName = "isaac.JointState";
  static NativeType decode(const RootType& root);
  static std::int64_t timestamp(const RootType& root);
  static flatbuffers::Offset<RootType> encode(flatbuffers::FlatBufferBuilder& builder,
                                              const NativeType& message, std::int64_t timestamp_ns);
};

template <>
struct UcxxMessageTraits<isaac::JointCommandCpuT> {
  using RootType = isaac::JointCommand;
  using NativeType = isaac::JointCommandCpuT;
  static constexpr bool kHasSeparateTensor = false;
  static constexpr std::string_view kSchemaName = "isaac.JointCommand";
  // Sim's OgnUCXSubscribeJointCommand calls VerifyJointCommandBuffer which
  // rejects buffers without this identifier.
  static constexpr const char* kFileIdentifier = "IJCM";
  static NativeType decode(const RootType& root);
  static std::int64_t timestamp(const RootType& root);
  static flatbuffers::Offset<RootType> encode(flatbuffers::FlatBufferBuilder& builder,
                                              const NativeType& message, std::int64_t timestamp_ns);
};

template <>
struct UcxxMessageTraits<holoscan::schema::ImuT> {
  using RootType = holoscan::schema::Imu;
  using NativeType = holoscan::schema::ImuT;
  static constexpr bool kHasSeparateTensor = false;
  static constexpr std::string_view kSchemaName = "holoscan.schema.Imu";
  static NativeType decode(const RootType& root);
  static std::int64_t timestamp(const RootType& root);
  static flatbuffers::Offset<RootType> encode(flatbuffers::FlatBufferBuilder& builder,
                                              const NativeType& message, std::int64_t timestamp_ns);
};

template <>
struct UcxxMessageTraits<holoscan::schema::ImageT> {
  using RootType = holoscan::schema::Image;
  using NativeType = holoscan::schema::ImageT;
  static constexpr bool kHasSeparateTensor = true;
  static constexpr std::string_view kSchemaName = "isaac.Image";
  static NativeType decode(const RootType& root);
  static std::int64_t timestamp(const RootType& root);
  static flatbuffers::Offset<RootType> encode(flatbuffers::FlatBufferBuilder& builder,
                                              const NativeType& message, std::int64_t timestamp_ns);
};

template <>
struct UcxxMessageTraits<holoscan::schema::TimeT> {
  using RootType = holoscan::schema::Time;
  using NativeType = holoscan::schema::TimeT;
  static constexpr bool kHasSeparateTensor = false;
  static constexpr std::string_view kSchemaName = "isaac.Time";
  static NativeType decode(const RootType& root) {
    return NativeType{.time_ns = root.time_ns(), .acqtime_ns = root.acqtime_ns()};
  }
  static std::int64_t timestamp(const RootType& root) { return root.time_ns(); }
  static flatbuffers::Offset<RootType> encode(flatbuffers::FlatBufferBuilder& builder,
                                              const NativeType& message, std::int64_t) {
    return RootType::Pack(builder, &message);
  }
};

}  // namespace isaac::os::ops
