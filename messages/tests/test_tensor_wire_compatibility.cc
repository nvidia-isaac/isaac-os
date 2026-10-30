/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>
#include <gtest/gtest.h>

#include "legacy_tensor_generated.h"
#include "tensor_generated.h"

namespace {

TEST(TensorWireCompatibility, LegacyIsaacTensorParsesAsHoloscanTensor) {
  flatbuffers::FlatBufferBuilder builder;
  const auto data = builder.CreateVector(std::vector<uint8_t>{0x00, 0x00, 0x80, 0x3f});
  const auto shape = builder.CreateVector(std::vector<int64_t>{1});
  const auto strides = builder.CreateVector(std::vector<int64_t>{1});
  const legacy::DLDataType dtype(legacy::DLDataTypeCode_kDLFloat, 32, 1);
  const legacy::DLDevice device(legacy::DLDeviceType_kDLCPU, 0);

  const auto tensor = legacy::CreateTensor(builder, data, shape, &dtype, &device, 1, strides);
  builder.Finish(tensor);

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(verifier.VerifyBuffer<holoscan::schema::Tensor>(nullptr));

  const auto* decoded = flatbuffers::GetRoot<holoscan::schema::Tensor>(builder.GetBufferPointer());
  ASSERT_NE(decoded->data(), nullptr);
  ASSERT_NE(decoded->shape(), nullptr);
  ASSERT_NE(decoded->dtype(), nullptr);
  ASSERT_NE(decoded->device(), nullptr);
  ASSERT_NE(decoded->strides(), nullptr);
  EXPECT_EQ(decoded->data()->size(), 4);
  EXPECT_EQ(decoded->data()->Get(2), 0x80);
  EXPECT_EQ(decoded->shape()->Get(0), 1);
  EXPECT_EQ(static_cast<uint8_t>(decoded->dtype()->code()), 2);
  EXPECT_EQ(decoded->dtype()->bits(), 32);
  EXPECT_EQ(decoded->dtype()->lanes(), 1);
  EXPECT_EQ(static_cast<int32_t>(decoded->device()->device_type()), 1);
  EXPECT_EQ(decoded->device()->device_id(), 0);
  EXPECT_EQ(decoded->ndim(), 1);
  EXPECT_EQ(decoded->strides()->Get(0), 1);
}

}  // namespace
