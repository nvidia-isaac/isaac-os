/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/reflection.h>

#include "common/mcap_logger/tensor_topic_poller.h"
#include "tensor_frame_bfbs_generated.h"
#include "tensor_frame_generated.h"

namespace isaac::os::mcap_logger {
namespace {

namespace wire = ::isaac::mcap_logger;

TEST(TensorFrameSchemaTest, RoundTripsOneTensorPlusMetadata) {
  flatbuffers::FlatBufferBuilder builder;

  constexpr std::size_t kDataSize = 6;
  const std::vector<std::uint32_t> shape{2, 3};
  const auto shape_offset = builder.CreateVector(shape);
  std::uint8_t* data_ptr = nullptr;
  const auto data_offset = builder.CreateUninitializedVector<std::uint8_t>(kDataSize, &data_ptr);
  for (std::uint8_t i = 0; i < kDataSize; ++i) {
    data_ptr[i] = i;
  }
  const auto entry_offset =
      wire::CreateTensorEntry(builder, shape_offset, wire::TensorDType_UInt8, data_offset);

  const std::vector<flatbuffers::Offset<wire::TensorEntry>> entries{entry_offset};
  const auto tensors_offset = builder.CreateVector(entries);
  const std::vector<std::uint8_t> metadata_bytes{0xAA, 0xBB};
  const auto metadata_offset = builder.CreateVector(metadata_bytes);
  const std::string metadata_schema_name = "holoscan.schema.Image";
  const auto schema_name_offset = builder.CreateString(metadata_schema_name);
  // Reuse TensorFrame's own embedded binary schema as a real, non-trivial byte blob for the
  // metadata_binary_schema fixture, rather than fabricating arbitrary bytes.
  const auto binary_schema_offset = builder.CreateVector(wire::TensorFrameBinarySchema::data(),
                                                         wire::TensorFrameBinarySchema::size());

  const auto frame_offset = wire::CreateTensorFrame(builder, tensors_offset, metadata_offset,
                                                    schema_name_offset, binary_schema_offset);
  wire::FinishTensorFrameBuffer(builder, frame_offset);

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  ASSERT_TRUE(wire::VerifyTensorFrameBuffer(verifier));

  const auto* frame = flatbuffers::GetRoot<wire::TensorFrame>(builder.GetBufferPointer());
  ASSERT_NE(frame->tensors(), nullptr);
  ASSERT_EQ(frame->tensors()->size(), entries.size());
  const auto* entry = frame->tensors()->Get(0);
  ASSERT_NE(entry->shape(), nullptr);
  ASSERT_EQ(entry->shape()->size(), shape.size());
  EXPECT_EQ(entry->shape()->Get(0), shape[0]);
  EXPECT_EQ(entry->shape()->Get(1), shape[1]);
  EXPECT_EQ(entry->dtype(), wire::TensorDType_UInt8);
  ASSERT_NE(entry->data(), nullptr);
  ASSERT_EQ(entry->data()->size(), kDataSize);
  for (std::uint8_t i = 0; i < kDataSize; ++i) {
    EXPECT_EQ(entry->data()->Get(i), i);
  }
  ASSERT_NE(frame->metadata(), nullptr);
  ASSERT_EQ(frame->metadata()->size(), metadata_bytes.size());
  EXPECT_EQ(frame->metadata()->Get(0), metadata_bytes[0]);
  EXPECT_EQ(frame->metadata()->Get(1), metadata_bytes[1]);
  ASSERT_NE(frame->metadata_schema_name(), nullptr);
  EXPECT_EQ(frame->metadata_schema_name()->str(), metadata_schema_name);
  ASSERT_NE(frame->metadata_binary_schema(), nullptr);
  ASSERT_EQ(frame->metadata_binary_schema()->size(), wire::TensorFrameBinarySchema::size());
  for (std::size_t i = 0; i < wire::TensorFrameBinarySchema::size(); ++i) {
    EXPECT_EQ(frame->metadata_binary_schema()->Get(static_cast<flatbuffers::uoffset_t>(i)),
              wire::TensorFrameBinarySchema::data()[i]);
  }
}

TEST(TensorFrameSchemaTest, BinarySchemaIsNonEmpty) {
  EXPECT_GT(wire::TensorFrameBinarySchema::size(), 0U);
  ASSERT_NE(wire::TensorFrameBinarySchema::data(), nullptr);
}

// kTensorFrameCanonicalName is a hand-written constant, not generated from tensor_frame.fbs (the
// schema has no traits binding -- see its CMakeLists.txt comment). This guards against it silently
// drifting from the BFBS's own root table name if tensor_frame.fbs's namespace or table name ever
// changes without updating the constant to match.
TEST(TensorFrameSchemaTest, EmbeddedBfbsRootNameMatchesCanonicalName) {
  const auto* reflected_schema = reflection::GetSchema(wire::TensorFrameBinarySchema::data());
  ASSERT_NE(reflected_schema, nullptr);
  ASSERT_NE(reflected_schema->root_table(), nullptr);
  ASSERT_NE(reflected_schema->root_table()->name(), nullptr);
  EXPECT_EQ(reflected_schema->root_table()->name()->str(), kTensorFrameCanonicalName);
}

}  // namespace
}  // namespace isaac::os::mcap_logger
