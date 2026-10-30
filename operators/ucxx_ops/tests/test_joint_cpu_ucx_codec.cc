/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>

#include <gtest/gtest.h>

#include "operators/ucxx_ops/message_traits.h"

namespace isaac::os::ops {

TEST(JointCpuUcxCodec, DecodesSimulatorStateWithAbsentOptionalFields) {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<float> position{0.1F, -0.2F, 0.3F, 0.4F, 0.0F, -0.1F};
  const auto bytes = builder.CreateVector(reinterpret_cast<const uint8_t*>(position.data()),
                                          position.size() * sizeof(float));
  const std::vector<int64_t> shape{6};
  const auto dims = builder.CreateVector(shape);
  const holoscan::schema::DLDataType dtype(holoscan::schema::DLDataTypeCode_kDLFloat, 32, 1);
  const holoscan::schema::DLDevice device(holoscan::schema::DLDeviceType_kDLCPU, 0);
  const auto tensor = holoscan::schema::CreateTensor(builder, bytes, dims, &dtype, &device, 1U);
  const auto header = isaac::CreateHeader(builder, {}, {}, 123456789);
  builder.Finish(isaac::CreateJointState(builder, header, 0, tensor, tensor));
  const auto* root = flatbuffers::GetRoot<isaac::JointState>(builder.GetBufferPointer());
  ASSERT_EQ(root->acceleration(), nullptr);
  ASSERT_EQ(root->temperature(), nullptr);
  const auto decoded = UcxxMessageTraits<isaac::JointStateCpuT>::decode(*root);
  EXPECT_EQ(decoded.position, position);
  EXPECT_EQ(decoded.velocity, position);
  EXPECT_TRUE(decoded.effort.empty());
  EXPECT_EQ(UcxxMessageTraits<isaac::JointStateCpuT>::timestamp(*root), 123456789);
}

TEST(JointCpuUcxCodec, FallsBackToStampWhenCaptureTimestampAbsent) {
  // Legacy peers only populate the deprecated `stamp` field. The reader must
  // still surface that timestamp until every producer has migrated.
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<float> position{0.0F};
  const std::byte* raw = reinterpret_cast<const std::byte*>(position.data());
  const auto bytes =
      builder.CreateVector(reinterpret_cast<const uint8_t*>(raw), position.size() * sizeof(float));
  const auto dims = builder.CreateVector(std::vector<int64_t>{1});
  const holoscan::schema::DLDataType dtype(holoscan::schema::DLDataTypeCode_kDLFloat, 32, 1);
  const holoscan::schema::DLDevice device(holoscan::schema::DLDeviceType_kDLCPU, 0);
  const auto tensor = holoscan::schema::CreateTensor(builder, bytes, dims, &dtype, &device, 1U);
  const auto stamp = isaac::CreateTime(builder, 42);
  const auto header = isaac::CreateHeader(builder, stamp);
  builder.Finish(isaac::CreateJointState(builder, header, 0, tensor, tensor));
  const auto* root = flatbuffers::GetRoot<isaac::JointState>(builder.GetBufferPointer());
  EXPECT_EQ(UcxxMessageTraits<isaac::JointStateCpuT>::timestamp(*root), 42);
}

TEST(JointCpuUcxCodec, RejectsMissingRequiredState) {
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(isaac::CreateJointState(builder));
  const auto* root = flatbuffers::GetRoot<isaac::JointState>(builder.GetBufferPointer());
  EXPECT_THROW(UcxxMessageTraits<isaac::JointStateCpuT>::decode(*root), std::runtime_error);
  EXPECT_THROW(UcxxMessageTraits<isaac::JointStateCpuT>::timestamp(*root), std::runtime_error);
}

TEST(JointCpuUcxCodec, RejectsMismatchedStateLengths) {
  isaac::JointStateCpuT state;
  state.position = {1.0F, 2.0F};
  state.velocity = {0.0F};
  flatbuffers::FlatBufferBuilder builder;
  using Codec = UcxxMessageTraits<isaac::JointStateCpuT>;
  builder.Finish(Codec::encode(builder, state, 100));
  const auto* root = flatbuffers::GetRoot<Codec::RootType>(builder.GetBufferPointer());
  EXPECT_THROW(Codec::decode(*root), std::runtime_error);
}

TEST(JointCpuUcxCodec, EncodesMaterializedCommandsAndSimulatorTimestamp) {
  isaac::JointCommandCpuT command;
  command.position = {0.1F, -0.2F};
  command.velocity = {0.0F, 0.0F};
  command.effort = {0.0F, 0.0F};
  command.kp = {30.0F, 30.0F};
  command.kd = {1.0F, 1.0F};
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(UcxxMessageTraits<isaac::JointCommandCpuT>::encode(builder, command, 987654321));
  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(verifier.VerifyBuffer<isaac::JointCommand>(nullptr));
  const auto* root = flatbuffers::GetRoot<isaac::JointCommand>(builder.GetBufferPointer());
  EXPECT_EQ(root->header()->capture_timestamp_ns(), 987654321);
  const auto decoded = UcxxMessageTraits<isaac::JointCommandCpuT>::decode(*root);
  EXPECT_EQ(decoded.position, command.position);
  EXPECT_EQ(decoded.velocity, command.velocity);
  EXPECT_EQ(decoded.effort, command.effort);
  EXPECT_EQ(decoded.kp, command.kp);
  EXPECT_EQ(decoded.kd, command.kd);
  for (const auto* tensor :
       {root->position(), root->velocity(), root->effort(), root->kp(), root->kd()}) {
    ASSERT_NE(tensor, nullptr);
    ASSERT_NE(tensor->data(), nullptr);
    EXPECT_EQ(tensor->data()->size(), 2U * sizeof(float));
    EXPECT_EQ(tensor->shape()->Get(0), 2);
  }
}

TEST(JointCpuUcxCodec, PreservesStateNamesEffortAndTimestamp) {
  isaac::JointStateCpuT state;
  state.position = {1.0F, 2.0F};
  state.velocity = {0.1F, 0.2F};
  state.effort = {3.0F, 4.0F};
  state.name = {"left", "right"};
  flatbuffers::FlatBufferBuilder builder;
  using Codec = UcxxMessageTraits<isaac::JointStateCpuT>;
  builder.Finish(Codec::encode(builder, state, 100));
  const auto* root = flatbuffers::GetRoot<Codec::RootType>(builder.GetBufferPointer());
  const auto decoded = Codec::decode(*root);
  EXPECT_EQ(decoded.position, state.position);
  EXPECT_EQ(decoded.velocity, state.velocity);
  EXPECT_EQ(decoded.effort, state.effort);
  EXPECT_EQ(decoded.name, state.name);
  EXPECT_EQ(Codec::timestamp(*root), 100);
  EXPECT_EQ(root->acceleration(), nullptr);
  EXPECT_EQ(root->temperature(), nullptr);
}

TEST(JointCpuUcxCodec, RejectsMalformedTensorDtype) {
  flatbuffers::FlatBufferBuilder builder;
  const auto data = builder.CreateVector(std::vector<uint8_t>{1, 2, 3, 4});
  const auto shape = builder.CreateVector(std::vector<int64_t>{1});
  const holoscan::schema::DLDataType dtype(holoscan::schema::DLDataTypeCode_kDLInt, 32, 1);
  const holoscan::schema::DLDevice device(holoscan::schema::DLDeviceType_kDLCPU, 0);
  const auto tensor = holoscan::schema::CreateTensor(builder, data, shape, &dtype, &device, 1U);
  builder.Finish(isaac::CreateJointState(builder, 0, 0, tensor, tensor));
  const auto* root = flatbuffers::GetRoot<isaac::JointState>(builder.GetBufferPointer());
  EXPECT_THROW(UcxxMessageTraits<isaac::JointStateCpuT>::decode(*root), std::runtime_error);
}

TEST(JointCpuUcxCodec, PreservesImuQuaternionVectorsAndHeader) {
  using Codec = UcxxMessageTraits<holoscan::schema::ImuT>;
  holoscan::schema::ImuT imu;
  imu.orientation = holoscan::schema::Quaternionf{0.1F, 0.2F, 0.3F, 0.5F};
  imu.angular_velocity = holoscan::schema::Vector3f{1.0F, 2.0F, 3.0F};
  imu.linear_acceleration = holoscan::schema::Vector3f{4.0F, 5.0F, 6.0F};
  imu.header = std::make_shared<holoscan::schema::HeaderT>();
  imu.header->frame_id = "imu";
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(Codec::encode(builder, imu, 1234));
  const auto* root = flatbuffers::GetRoot<Codec::RootType>(builder.GetBufferPointer());
  const auto decoded = Codec::decode(*root);
  EXPECT_FLOAT_EQ(decoded.orientation.w(), 0.5F);
  EXPECT_FLOAT_EQ(decoded.orientation.x(), 0.1F);
  EXPECT_FLOAT_EQ(decoded.orientation.y(), 0.2F);
  EXPECT_FLOAT_EQ(decoded.orientation.z(), 0.3F);
  EXPECT_FLOAT_EQ(decoded.angular_velocity.y(), 2.0F);
  EXPECT_FLOAT_EQ(decoded.linear_acceleration.z(), 6.0F);
  ASSERT_NE(decoded.header, nullptr);
  EXPECT_EQ(decoded.header->frame_id, "imu");
  EXPECT_EQ(decoded.header->capture_timestamp_ns, 1234);
}

TEST(JointCpuUcxCodec, ImuEncodePreservesPreStampedCaptureTimestamp) {
  // Fast path: when the producer already put a non-zero capture_timestamp_ns
  // on the header, encode() must not overwrite it with the emit-metadata arg.
  // Also confirms the caller's HeaderT is not mutated (shared_ptr aliasing).
  using Codec = UcxxMessageTraits<holoscan::schema::ImuT>;
  holoscan::schema::ImuT imu;
  imu.orientation = holoscan::schema::Quaternionf{0.1F, 0.2F, 0.3F, 0.5F};
  imu.angular_velocity = holoscan::schema::Vector3f{0.0F, 0.0F, 0.0F};
  imu.linear_acceleration = holoscan::schema::Vector3f{0.0F, 0.0F, 0.0F};
  imu.header = std::make_shared<holoscan::schema::HeaderT>();
  imu.header->frame_id = "imu";
  imu.header->capture_timestamp_ns = 5555;
  const auto original_header = imu.header;  // keep the shared_ptr for aliasing check
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(Codec::encode(builder, imu, /*timestamp_ns=*/9999));
  const auto* root = flatbuffers::GetRoot<Codec::RootType>(builder.GetBufferPointer());
  const auto decoded = Codec::decode(*root);
  ASSERT_NE(decoded.header, nullptr);
  EXPECT_EQ(decoded.header->capture_timestamp_ns, 5555);   // not overwritten by 9999
  EXPECT_EQ(original_header->capture_timestamp_ns, 5555);  // caller's copy untouched
}

TEST(JointCpuUcxCodec, PreservesTimeFields) {
  using Codec = UcxxMessageTraits<holoscan::schema::TimeT>;
  holoscan::schema::TimeT time;
  time.time_ns = 123;
  time.acqtime_ns = 456;
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(Codec::encode(builder, time, 0));
  const auto decoded =
      Codec::decode(*flatbuffers::GetRoot<Codec::RootType>(builder.GetBufferPointer()));
  EXPECT_EQ(decoded.time_ns, 123);
  EXPECT_EQ(decoded.acqtime_ns, 456);
}

}  // namespace isaac::os::ops
