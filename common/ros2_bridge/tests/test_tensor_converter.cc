/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include <flatbuffers/flatbuffers.h>

#include "common/ros2_bridge/tensor_converter.h"

namespace isaac::os::ros2_bridge {
namespace {

using ImageConverter = TensorConverter<sensor_msgs::msg::Image, holoscan::schema::ImageT>;

TEST(ImageTensorConverter, ToTensorsThenFromTensorsRoundTripsMetadataAndPixels) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->height = 2;
  msg->width = 2;
  msg->encoding = "rgb8";
  msg->step = 6;
  msg->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  // Chosen so sec/nanosec split round-trips unambiguously through capture_timestamp_ns (an
  // arbitrary instant well clear of the epoch, with a sub-second remainder that would be lost by
  // a sec-only or ms-truncating conversion).
  msg->header.stamp.sec = 1'700'000'000;
  msg->header.stamp.nanosec = 123'456'789U;
  msg->header.frame_id = "camera_optical_frame";

  auto tensors = ImageConverter::to_tensors(msg, Ros2TopicSpec{});
  ASSERT_TRUE(tensors.has_value());
  auto& [metadata_bytes, tensor_list] = *tensors;
  ASSERT_EQ(tensor_list.size(), 1U);
  EXPECT_EQ(tensor_list[0].nbytes(), static_cast<std::int64_t>(msg->data.size()));

  flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t*>(metadata_bytes.data()),
                                 metadata_bytes.size());
  ASSERT_TRUE(verifier.VerifyBuffer<holoscan::schema::Image>(nullptr));
  const auto* root = flatbuffers::GetRoot<holoscan::schema::Image>(metadata_bytes.data());

  // Confirms to_tensors() actually wrote capture_timestamp_ns/frame_id into the built metadata,
  // not just that from_tensors_for_root() can read back whatever it's given (the two other tests
  // below already cover that half in isolation).
  ASSERT_NE(root->header(), nullptr);
  EXPECT_EQ(root->header()->capture_timestamp_ns(),
            1'700'000'000LL * 1'000'000'000LL + 123'456'789LL);
  ASSERT_NE(root->header()->frame_id(), nullptr);
  EXPECT_EQ(root->header()->frame_id()->str(), "camera_optical_frame");

  // from_tensors_for_root only needs a root pointer and a tensor span -- passing tensor_list
  // directly (rather than staging it through a ChannelResolvedTensorBundle, which is normally
  // only ever populated by the live transport) is equivalent and avoids constructing a bundle
  // this test doesn't otherwise exercise.
  const auto round_tripped = ImageConverter::from_tensors_for_root(root, tensor_list);
  ASSERT_TRUE(round_tripped.has_value());
  EXPECT_EQ(round_tripped->height, msg->height);
  EXPECT_EQ(round_tripped->width, msg->width);
  EXPECT_EQ(round_tripped->encoding, msg->encoding);
  ASSERT_EQ(round_tripped->data.size(), msg->data.size());
  EXPECT_EQ(round_tripped->data, msg->data);
  EXPECT_EQ(round_tripped->header.stamp.sec, msg->header.stamp.sec);
  EXPECT_EQ(round_tripped->header.stamp.nanosec, msg->header.stamp.nanosec);
  EXPECT_EQ(round_tripped->header.frame_id, msg->header.frame_id);
}

TEST(ImageTensorConverter, FromTensorsForRootLeavesHeaderZeroedWhenProducerNeverSetIt) {
  // A ROS2 sensor_msgs::msg::Image whose header was never touched has header.stamp/frame_id at
  // their own zero-initialized default (sec=0, nanosec=0, frame_id=""), same as a non-ROS2
  // Holoscan producer that never populates capture_timestamp_ns/frame_id. Either way,
  // from_tensors_for_root() must round-trip that back to a zeroed sensor_msgs::msg::Image::header
  // -- not crash, and not synthesize a nonzero value -- whether or not the SDK actually omits the
  // nested Header table for all-default field values (from_tensors_for_root()'s own root->header()
  // null check covers that case if it does).
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->height = 1;
  msg->width = 1;
  msg->encoding = "mono8";
  msg->data = {0U};

  auto tensors = ImageConverter::to_tensors(msg, Ros2TopicSpec{});
  ASSERT_TRUE(tensors.has_value());
  auto& [metadata_bytes, tensor_list] = *tensors;

  flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t*>(metadata_bytes.data()),
                                 metadata_bytes.size());
  ASSERT_TRUE(verifier.VerifyBuffer<holoscan::schema::Image>(nullptr));
  const auto* root = flatbuffers::GetRoot<holoscan::schema::Image>(metadata_bytes.data());

  const auto round_tripped = ImageConverter::from_tensors_for_root(root, tensor_list);
  ASSERT_TRUE(round_tripped.has_value());
  EXPECT_EQ(round_tripped->header.stamp.sec, 0);
  EXPECT_EQ(round_tripped->header.stamp.nanosec, 0U);
  EXPECT_EQ(round_tripped->header.frame_id, "");
}

// Builds minimal 1x1 mono8 metadata/tensor for a given raw capture_timestamp_ns, bypassing
// to_tensors()'s ROS2 sec/nanosec split -- needed to exercise capture_timestamp_ns values (a
// negative instant, or one beyond builtin_interfaces/Time's int32-seconds range) that a real
// sensor_msgs::msg::Image header could not itself produce, but that this int64 Holoscan-side
// field does not otherwise forbid.
std::pair<std::vector<std::byte>, std::vector<holoscan::Tensor>> BuildMinimalImageWithTimestamp(
    std::int64_t capture_timestamp_ns) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->data = {0U};
  auto backing = std::make_shared<detail::Ros2ImageTensorBacking>(msg);
  void* const data = msg->data.data();
  const std::array<std::int64_t, 1U> shape{1};
  auto tensor =
      holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                            holoscan::MemoryStorageType::kSystem, data, std::move(backing));
  std::vector<holoscan::Tensor> tensor_list{std::move(*tensor)};

  holoscan::schema::ImageMetadataOptions options;
  options.height = 1;
  options.width = 1;
  options.encoding = holoscan::schema::ImageEncoding_MONO8;
  options.capture_timestamp_ns = capture_timestamp_ns;
  auto metadata_bytes = holoscan::schema::build_image_metadata(tensor_list[0], options);
  return {std::move(*metadata_bytes), std::move(tensor_list)};
}

TEST(ImageTensorConverter, FromTensorsForRootNormalizesANegativeCaptureTimestamp) {
  // -1.5 s: truncating division/remainder (capture_timestamp_ns / 1e9, capture_timestamp_ns %
  // 1e9) would give sec=-1, nanosec=-500000000 -- a negative value that wraps to a huge one once
  // cast to uint32_t. Floor-normalized, this must instead read as sec=-2, nanosec=500000000
  // (builtin_interfaces/Time requires nanosec in [0, 1e9) regardless of sec's sign).
  const std::int64_t capture_timestamp_ns = -1'500'000'000LL;
  auto [metadata_bytes, tensor_list] = BuildMinimalImageWithTimestamp(capture_timestamp_ns);

  flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t*>(metadata_bytes.data()),
                                 metadata_bytes.size());
  ASSERT_TRUE(verifier.VerifyBuffer<holoscan::schema::Image>(nullptr));
  const auto* root = flatbuffers::GetRoot<holoscan::schema::Image>(metadata_bytes.data());

  const auto result = ImageConverter::from_tensors_for_root(root, tensor_list);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->header.stamp.sec, -2);
  EXPECT_EQ(result->header.stamp.nanosec, 500'000'000U);
}

TEST(ImageTensorConverter, FromTensorsForRootRejectsACaptureTimestampOutsideInt32Seconds) {
  // One nanosecond past the largest instant whose seconds component still fits int32: (INT32_MAX
  // + 1) seconds, converted to ns. builtin_interfaces/Time's sec field cannot represent this --
  // from_tensors_for_root() must reject it rather than silently truncate to a wrong-but-plausible
  // in-range value.
  const std::int64_t capture_timestamp_ns =
      (static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1) * 1'000'000'000LL;
  auto [metadata_bytes, tensor_list] = BuildMinimalImageWithTimestamp(capture_timestamp_ns);

  flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t*>(metadata_bytes.data()),
                                 metadata_bytes.size());
  ASSERT_TRUE(verifier.VerifyBuffer<holoscan::schema::Image>(nullptr));
  const auto* root = flatbuffers::GetRoot<holoscan::schema::Image>(metadata_bytes.data());

  const auto result = ImageConverter::from_tensors_for_root(root, tensor_list);
  EXPECT_FALSE(result.has_value());
}

TEST(ImageTensorConverter, FromTensorsForRootRejectsANullRootInsteadOfCrashing) {
  const std::vector<holoscan::Tensor> tensors;
  const auto result = ImageConverter::from_tensors_for_root(nullptr, tensors);
  EXPECT_FALSE(result.has_value());
}

TEST(ImageTensorConverter, FromTensorsForRootRejectsAnUnmappedEncoding) {
  // Built directly rather than through to_tensors(), which now rejects a tensor whose size
  // doesn't match its own message's height/width/encoding. This test needs a 2-byte tensor to
  // satisfy build_image_metadata()'s own validation when the metadata below is rebuilt as MONO16
  // (2 bytes/pixel) -- independent of what a real 1x1 mono8 message would actually require (1
  // byte) -- so it constructs the tensor the same way to_tensors() itself does, without going
  // through to_tensors()'s own size check.
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->data = {0U, 0U};
  auto backing = std::make_shared<detail::Ros2ImageTensorBacking>(msg);
  void* const data = msg->data.data();
  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(msg->data.size())};
  auto tensor =
      holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                            holoscan::MemoryStorageType::kSystem, data, std::move(backing));
  ASSERT_TRUE(tensor.has_value());
  const std::vector<holoscan::Tensor> tensor_list{std::move(*tensor)};

  // Rebuild the metadata with an encoding the ROS2 bridge does not map (MONO16 is defined by
  // image.fbs but has no sensor_msgs/msg/Image equivalent registered in hsdk_encoding_to_ros2) --
  // a uint8 Tensor is still a valid opaque byte-buffer carriage for a single-plane encoding that
  // names its own element type, per image.fbs's own comment on Image::data.
  holoscan::schema::ImageMetadataOptions options;
  options.height = 1;
  options.width = 1;
  options.encoding = holoscan::schema::ImageEncoding_MONO16;
  auto unmapped_metadata = holoscan::schema::build_image_metadata(tensor_list[0], options);
  ASSERT_TRUE(unmapped_metadata.has_value());

  flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t*>(unmapped_metadata->data()),
                                 unmapped_metadata->size());
  ASSERT_TRUE(verifier.VerifyBuffer<holoscan::schema::Image>(nullptr));
  const auto* root = flatbuffers::GetRoot<holoscan::schema::Image>(unmapped_metadata->data());

  const auto result = ImageConverter::from_tensors_for_root(root, tensor_list);
  EXPECT_FALSE(result.has_value());
}

// DescriptorTensorTopicContract exposes no accessor for the bound it was built with (its
// fields_ member is private, friended only to holoscan::Graph), so these two tests can only
// pin contract()'s observable behavior -- that it succeeds -- for an explicit vs. a defaulted
// bound, not the resulting numeric value. The default itself is exercised end-to-end by
// test_ros2_bridge_roundtrip.cc's Image test, which sends frames well under 8 MiB.
TEST(ImageTensorConverter, ContractBuildsSuccessfullyWithAnExplicitMaxTensorBytes) {
  Ros2TopicSpec topic;
  topic.max_tensor_bytes = 1024U;
  EXPECT_NO_THROW((void)ImageConverter::contract(topic));
}

TEST(ImageTensorConverter, ContractFallsBackToDefaultMaxTensorBytesWhenUnset) {
  Ros2TopicSpec topic;
  EXPECT_NO_THROW((void)ImageConverter::contract(topic));
}

TEST(ImageTensorConverter, ToTensorsRejectsAnUnsupportedEncoding) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->height = 1;
  msg->width = 1;
  msg->encoding = "not_a_real_encoding";
  msg->data = {0U};

  const auto result = ImageConverter::to_tensors(msg, Ros2TopicSpec{});
  EXPECT_FALSE(result.has_value());
}

TEST(ImageTensorConverter, ToTensorsRejectsDataShorterThanHeightWidthEncodingImply) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->height = 2;
  msg->width = 2;
  msg->encoding = "rgb8";
  msg->step = 6;
  // A full 2x2 rgb8 frame needs 2*2*3 = 12 bytes; this message only has 4.
  msg->data = {1, 2, 3, 4};

  const auto result = ImageConverter::to_tensors(msg, Ros2TopicSpec{});
  EXPECT_FALSE(result.has_value());
}

TEST(ImageTensorConverter, ToTensorsRejectsAStepInconsistentWithAPackedRow) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->height = 2;
  msg->width = 2;
  msg->encoding = "rgb8";
  // A packed 2-wide rgb8 row is 2*3 = 6 bytes; declare a padded row of 9 bytes instead. The
  // Tensor this would build addresses the padded buffer, but image.fbs's Image metadata has no
  // per-row stride field -- it always describes one packed frame -- so this would silently
  // corrupt a Holoscan consumer's row addressing if it were accepted.
  msg->step = 9;
  msg->data = {1, 2, 3, 0, 0, 0, 4, 5, 6, 0, 0, 0, 7, 8, 9, 0, 0, 0};

  const auto result = ImageConverter::to_tensors(msg, Ros2TopicSpec{});
  EXPECT_FALSE(result.has_value());
}

TEST(ImageTensorConverter, ToTensorsRejectsDataLongerThanHeightWidthEncodingImply) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->height = 2;
  msg->width = 2;
  msg->encoding = "rgb8";
  msg->step = 6;
  // A full 2x2 rgb8 frame needs exactly 2*2*3 = 12 bytes; this message has 13. Accepting the
  // extra trailing byte would carry it into the tensor unrejected -- to_tensors() must reject
  // any size other than the exact packed-frame size, not just a short one.
  msg->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

  const auto result = ImageConverter::to_tensors(msg, Ros2TopicSpec{});
  EXPECT_FALSE(result.has_value());
}

TEST(ImageTensorConverter, FromTensorsForRootRejectsATensorLongerThanHeightWidthEncodingImply) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->height = 2;
  msg->width = 2;
  msg->encoding = "rgb8";
  msg->step = 6;
  msg->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  auto tensors = ImageConverter::to_tensors(msg, Ros2TopicSpec{});
  ASSERT_TRUE(tensors.has_value());
  auto& [metadata_bytes, tensor_list] = *tensors;

  flatbuffers::Verifier verifier(reinterpret_cast<const std::uint8_t*>(metadata_bytes.data()),
                                 metadata_bytes.size());
  ASSERT_TRUE(verifier.VerifyBuffer<holoscan::schema::Image>(nullptr));
  const auto* root = flatbuffers::GetRoot<holoscan::schema::Image>(metadata_bytes.data());

  // A tensor with more bytes than height*width*encoding implies -- from_tensors_for_root must
  // reject this rather than copy the extra bytes into a sensor_msgs::msg::Image whose data.size()
  // then disagrees with its own height*step. Built the same way to_tensors() itself builds a
  // Tensor (Ros2ImageTensorBacking over a live buffer), since to_tensors() itself now rejects an
  // oversized buffer and so can no longer be used to produce one for this test.
  auto oversized_msg = std::make_shared<sensor_msgs::msg::Image>();
  oversized_msg->data = std::vector<std::uint8_t>(13U, 0U);
  auto backing = std::make_shared<detail::Ros2ImageTensorBacking>(oversized_msg);
  void* const oversized_data = oversized_msg->data.data();
  const std::array<std::int64_t, 1U> oversized_shape{13};
  auto oversized_tensor = holoscan::make_tensor(oversized_shape, DLDataType{kDLUInt, 8U, 1U},
                                                std::nullopt, holoscan::MemoryStorageType::kSystem,
                                                oversized_data, std::move(backing));
  ASSERT_TRUE(oversized_tensor.has_value());
  const std::vector<holoscan::Tensor> oversized_tensor_list{std::move(*oversized_tensor)};

  const auto result = ImageConverter::from_tensors_for_root(root, oversized_tensor_list);
  EXPECT_FALSE(result.has_value());
}

TEST(ImageTensorConverter, ToTensorsAcceptsAnUnsetStepOfZero) {
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->height = 2;
  msg->width = 2;
  msg->encoding = "rgb8";
  msg->step = 0;  // ROS2 does not require a producer to set `step`.
  msg->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  const auto result = ImageConverter::to_tensors(msg, Ros2TopicSpec{});
  EXPECT_TRUE(result.has_value());
}

}  // namespace
}  // namespace isaac::os::ros2_bridge
