/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>
#include <limits>

#include "gtest/gtest.h"

#include "common/ros2_bridge/converter.h"
#include "common/ros2_bridge/converter_concepts.h"
#include "common/ros2_bridge/default_ros2_converter.h"
#include "common/ros2_bridge/ros2_timestamp.h"

namespace isaac::os::ros2_bridge {
namespace {

using JointStateConverter = Converter<sensor_msgs::msg::JointState, isaac::JointStateCpuT>;

TEST(JointStateConverter, Ros2ToHsdkCastsAllFields) {
  sensor_msgs::msg::JointState msg;
  msg.name = {"joint_0", "joint_1"};
  msg.position = {0.1, 0.2};
  msg.velocity = {1.0, 2.0};
  msg.effort = {0.5, -0.5};

  const isaac::JointStateCpuT hsdk = JointStateConverter::to_hsdk(msg);
  ASSERT_EQ(hsdk.name.size(), 2U);
  EXPECT_EQ(hsdk.name[0], msg.name[0]);
  EXPECT_EQ(hsdk.name[1], msg.name[1]);
  ASSERT_EQ(hsdk.position.size(), 2U);
  EXPECT_FLOAT_EQ(hsdk.position[0], msg.position[0]);
  EXPECT_FLOAT_EQ(hsdk.position[1], msg.position[1]);
  ASSERT_EQ(hsdk.velocity.size(), 2U);
  EXPECT_FLOAT_EQ(hsdk.velocity[0], msg.velocity[0]);
  EXPECT_FLOAT_EQ(hsdk.velocity[1], msg.velocity[1]);
  ASSERT_EQ(hsdk.effort.size(), 2U);
  EXPECT_FLOAT_EQ(hsdk.effort[0], msg.effort[0]);
  EXPECT_FLOAT_EQ(hsdk.effort[1], msg.effort[1]);
}

TEST(JointStateConverter, HsdkToRos2CastsAllFields) {
  isaac::JointStateCpuT hsdk;
  hsdk.name = {"joint_0", "joint_1"};
  hsdk.position = {0.1F, 0.2F};
  hsdk.velocity = {1.0F, 2.0F};
  hsdk.effort = {0.5F, -0.5F};

  const sensor_msgs::msg::JointState ros2 = JointStateConverter::to_ros2(hsdk);
  ASSERT_EQ(ros2.name.size(), 2U);
  EXPECT_EQ(ros2.name[0], hsdk.name[0]);
  EXPECT_EQ(ros2.name[1], hsdk.name[1]);
  ASSERT_EQ(ros2.position.size(), 2U);
  EXPECT_NEAR(ros2.position[0], hsdk.position[0], 1e-5);
  EXPECT_NEAR(ros2.position[1], hsdk.position[1], 1e-5);
  ASSERT_EQ(ros2.velocity.size(), 2U);
  EXPECT_NEAR(ros2.velocity[0], hsdk.velocity[0], 1e-5);
  EXPECT_NEAR(ros2.velocity[1], hsdk.velocity[1], 1e-5);
  ASSERT_EQ(ros2.effort.size(), 2U);
  EXPECT_NEAR(ros2.effort[0], hsdk.effort[0], 1e-5);
  EXPECT_NEAR(ros2.effort[1], hsdk.effort[1], 1e-5);
}

using ImuConverter = Converter<sensor_msgs::msg::Imu, holoscan::schema::ImuT>;

TEST(ImuConverter, Ros2ToHsdkCastsAllFields) {
  sensor_msgs::msg::Imu msg;
  msg.header.stamp.sec = 5;
  msg.header.stamp.nanosec = 250'000'000U;
  msg.header.frame_id = "imu_link";
  msg.orientation.x = 0.1;
  msg.orientation.y = 0.2;
  msg.orientation.z = 0.3;
  msg.orientation.w = 0.9;
  msg.angular_velocity.x = 1.0;
  msg.angular_velocity.y = 2.0;
  msg.angular_velocity.z = 3.0;
  msg.linear_acceleration.x = 4.0;
  msg.linear_acceleration.y = 5.0;
  msg.linear_acceleration.z = 6.0;
  msg.orientation_covariance.fill(0.0);
  msg.orientation_covariance[0] = 0.01;
  msg.angular_velocity_covariance.fill(0.0);
  msg.angular_velocity_covariance[4] = 0.02;
  msg.linear_acceleration_covariance.fill(0.0);
  msg.linear_acceleration_covariance[8] = 0.03;

  const holoscan::schema::ImuT hsdk = ImuConverter::to_hsdk(msg);
  ASSERT_NE(hsdk.header, nullptr);
  EXPECT_EQ(hsdk.header->capture_timestamp_ns, 5'250'000'000LL);
  EXPECT_EQ(hsdk.header->frame_id, msg.header.frame_id);
  EXPECT_FLOAT_EQ(hsdk.orientation.x(), msg.orientation.x);
  EXPECT_FLOAT_EQ(hsdk.orientation.w(), msg.orientation.w);
  EXPECT_FLOAT_EQ(hsdk.angular_velocity.y(), msg.angular_velocity.y);
  EXPECT_FLOAT_EQ(hsdk.linear_acceleration.z(), msg.linear_acceleration.z);
  EXPECT_EQ(hsdk.flags, holoscan::schema::ImuFlags_ORIENTATION_VALID);
  ASSERT_NE(hsdk.orientation_covariance, nullptr);
  EXPECT_FLOAT_EQ(hsdk.orientation_covariance->m()->Get(0), msg.orientation_covariance[0]);
  ASSERT_NE(hsdk.angular_velocity_covariance, nullptr);
  EXPECT_FLOAT_EQ(hsdk.angular_velocity_covariance->m()->Get(4),
                  msg.angular_velocity_covariance[4]);
  ASSERT_NE(hsdk.linear_acceleration_covariance, nullptr);
  EXPECT_FLOAT_EQ(hsdk.linear_acceleration_covariance->m()->Get(8),
                  msg.linear_acceleration_covariance[8]);
}

TEST(ImuConverter, Ros2ToHsdkWithOrientationCovarianceSentinelClearsOrientationValid) {
  sensor_msgs::msg::Imu msg;
  msg.orientation.w = 1.0;
  msg.orientation_covariance.fill(0.0);
  msg.orientation_covariance[0] = -1.0;

  const holoscan::schema::ImuT hsdk = ImuConverter::to_hsdk(msg);
  EXPECT_EQ(hsdk.flags & holoscan::schema::ImuFlags_ORIENTATION_VALID, 0);
}

TEST(ImuConverter, HsdkToRos2CastsAllFields) {
  const sensor_msgs::msg::Imu original = [] {
    sensor_msgs::msg::Imu msg;
    msg.header.stamp.sec = 5;
    msg.header.stamp.nanosec = 250'000'000U;
    msg.header.frame_id = "imu_link";
    msg.orientation.w = 1.0;
    msg.angular_velocity.x = 1.0;
    msg.linear_acceleration.z = -9.8;
    msg.orientation_covariance.fill(0.0);
    msg.angular_velocity_covariance.fill(0.0);
    msg.linear_acceleration_covariance.fill(0.0);
    return msg;
  }();
  const holoscan::schema::ImuT hsdk = ImuConverter::to_hsdk(original);

  const auto ros2 = ImuConverter::to_ros2(hsdk);
  ASSERT_TRUE(ros2.has_value());
  EXPECT_EQ(ros2->header.stamp.sec, original.header.stamp.sec);
  EXPECT_EQ(ros2->header.stamp.nanosec, original.header.stamp.nanosec);
  EXPECT_EQ(ros2->header.frame_id, original.header.frame_id);
  EXPECT_NEAR(ros2->orientation.w, original.orientation.w, 1e-5);
  EXPECT_NEAR(ros2->angular_velocity.x, original.angular_velocity.x, 1e-5);
  EXPECT_NEAR(ros2->linear_acceleration.z, original.linear_acceleration.z, 1e-5);
}

TEST(ImuConverter, HsdkToRos2WithoutOrientationValidSetsSentinel) {
  holoscan::schema::ImuT hsdk;
  hsdk.orientation = holoscan::schema::Quaternionf(0.0F, 0.0F, 0.0F, 1.0F);
  // flags left at its default (0): ORIENTATION_VALID is not set, matching a raw
  // gyro/accel-only IMU that doesn't estimate orientation.

  const auto ros2 = ImuConverter::to_ros2(hsdk);
  ASSERT_TRUE(ros2.has_value());
  EXPECT_EQ(ros2->orientation_covariance[0], -1.0);
}

TEST(ImuConverter, HsdkToRos2WithNegativeNanosecondFloorsSecond) {
  holoscan::schema::ImuT hsdk;
  hsdk.orientation = holoscan::schema::Quaternionf(0.0F, 0.0F, 0.0F, 1.0F);
  hsdk.header = std::make_shared<holoscan::schema::HeaderT>();
  // -0.25s should floor-divide to sec=-1, nanosec=750'000'000, not truncate to sec=0.
  hsdk.header->capture_timestamp_ns = -250'000'000LL;
  hsdk.header->frame_id = "imu_link";

  const auto ros2 = ImuConverter::to_ros2(hsdk);
  ASSERT_TRUE(ros2.has_value());
  EXPECT_EQ(ros2->header.stamp.sec, -1);
  EXPECT_EQ(ros2->header.stamp.nanosec, 750'000'000U);
  EXPECT_EQ(ros2->header.frame_id, "imu_link");
}

TEST(ImuConverter, HsdkToRos2WithNoHeaderLeavesRos2HeaderDefaulted) {
  holoscan::schema::ImuT hsdk;
  hsdk.orientation = holoscan::schema::Quaternionf(0.0F, 0.0F, 0.0F, 1.0F);

  const auto ros2 = ImuConverter::to_ros2(hsdk);
  ASSERT_TRUE(ros2.has_value());
  EXPECT_EQ(ros2->header.stamp.sec, 0);
  EXPECT_EQ(ros2->header.stamp.nanosec, 0U);
  EXPECT_TRUE(ros2->header.frame_id.empty());
}

TEST(ImuConverter, HsdkToRos2RejectsACaptureTimestampOutsideInt32Seconds) {
  holoscan::schema::ImuT hsdk;
  hsdk.orientation = holoscan::schema::Quaternionf(0.0F, 0.0F, 0.0F, 1.0F);
  hsdk.header = std::make_shared<holoscan::schema::HeaderT>();
  // One nanosecond past the largest instant whose seconds component still fits int32.
  hsdk.header->capture_timestamp_ns =
      (static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1) * 1'000'000'000LL;

  const auto ros2 = ImuConverter::to_ros2(hsdk);
  EXPECT_FALSE(ros2.has_value());
}

using PoseConverter = Converter<geometry_msgs::msg::Pose, isaac::PoseCpuT>;
using PoseStampedConverter = Converter<geometry_msgs::msg::PoseStamped, isaac::PoseCpuT>;

TEST(PoseConverter, Ros2ToHsdkLeavesHeaderUnset) {
  geometry_msgs::msg::Pose msg;
  msg.position.x = 1.0;
  msg.position.y = 2.0;
  msg.position.z = 3.0;
  msg.orientation.w = 1.0;

  const isaac::PoseCpuT hsdk = PoseConverter::to_hsdk(msg);
  EXPECT_DOUBLE_EQ(hsdk.position.x(), msg.position.x);
  EXPECT_DOUBLE_EQ(hsdk.position.z(), msg.position.z);
  EXPECT_DOUBLE_EQ(hsdk.orientation.w(), msg.orientation.w);
  EXPECT_EQ(hsdk.header, nullptr);
}

TEST(PoseConverter, HsdkToRos2CastsAllFields) {
  isaac::PoseCpuT hsdk;
  hsdk.position = holoscan::schema::Vector3d(1.0, 2.0, 3.0);
  hsdk.orientation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);

  const geometry_msgs::msg::Pose ros2 = PoseConverter::to_ros2(hsdk);
  EXPECT_NEAR(ros2.position.x, hsdk.position.x(), 1e-5);
  EXPECT_NEAR(ros2.position.z, hsdk.position.z(), 1e-5);
  EXPECT_NEAR(ros2.orientation.w, hsdk.orientation.w(), 1e-5);
}

TEST(PoseStampedConverter, Ros2ToHsdkFillsHeader) {
  geometry_msgs::msg::PoseStamped msg;
  msg.header.stamp.sec = 5;
  msg.header.stamp.nanosec = 250'000'000U;
  msg.header.frame_id = "map";
  msg.pose.position.x = 1.0;
  msg.pose.orientation.w = 1.0;

  const isaac::PoseCpuT hsdk = PoseStampedConverter::to_hsdk(msg);
  EXPECT_DOUBLE_EQ(hsdk.position.x(), msg.pose.position.x);
  ASSERT_NE(hsdk.header, nullptr);
  EXPECT_EQ(hsdk.header->capture_timestamp_ns, 5'250'000'000LL);
  EXPECT_EQ(hsdk.header->frame_id, msg.header.frame_id);
}

TEST(PoseStampedConverter, HsdkToRos2CastsAllFields) {
  isaac::PoseCpuT hsdk;
  hsdk.position = holoscan::schema::Vector3d(1.0, 2.0, 3.0);
  hsdk.orientation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);
  hsdk.header = std::make_shared<holoscan::schema::HeaderT>();
  hsdk.header->capture_timestamp_ns = 5'250'000'000LL;
  hsdk.header->frame_id = "map";

  const auto ros2 = PoseStampedConverter::to_ros2(hsdk);
  ASSERT_TRUE(ros2.has_value());
  EXPECT_NEAR(ros2->pose.position.x, hsdk.position.x(), 1e-5);
  EXPECT_NEAR(ros2->pose.position.z, hsdk.position.z(), 1e-5);
  EXPECT_EQ(ros2->header.stamp.sec, 5);
  EXPECT_EQ(ros2->header.stamp.nanosec, 250'000'000U);
  EXPECT_EQ(ros2->header.frame_id, hsdk.header->frame_id);
}

TEST(PoseStampedConverter, HsdkToRos2WithNoHeaderLeavesRos2HeaderDefaulted) {
  isaac::PoseCpuT hsdk;
  hsdk.position = holoscan::schema::Vector3d(1.0, 0.0, 0.0);
  hsdk.orientation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);

  const auto ros2 = PoseStampedConverter::to_ros2(hsdk);
  ASSERT_TRUE(ros2.has_value());
  EXPECT_NEAR(ros2->pose.position.x, hsdk.position.x(), 1e-5);
  EXPECT_EQ(ros2->header.stamp.sec, 0);
  EXPECT_TRUE(ros2->header.frame_id.empty());
}

using TwistConverter = Converter<geometry_msgs::msg::Twist, isaac::TwistCpuT>;
using TwistStampedConverter = Converter<geometry_msgs::msg::TwistStamped, isaac::TwistCpuT>;

TEST(TwistConverter, Ros2ToHsdkLeavesHeaderUnset) {
  geometry_msgs::msg::Twist msg;
  msg.linear.x = 1.0;
  msg.angular.z = 0.5;

  const isaac::TwistCpuT hsdk = TwistConverter::to_hsdk(msg);
  EXPECT_DOUBLE_EQ(hsdk.linear.x(), msg.linear.x);
  EXPECT_DOUBLE_EQ(hsdk.angular.z(), msg.angular.z);
  EXPECT_EQ(hsdk.header, nullptr);
}

TEST(TwistConverter, HsdkToRos2CastsAllFields) {
  isaac::TwistCpuT hsdk;
  hsdk.linear = holoscan::schema::Vector3d(1.0, 2.0, 3.0);
  hsdk.angular = holoscan::schema::Vector3d(0.1, 0.2, 0.3);

  const geometry_msgs::msg::Twist ros2 = TwistConverter::to_ros2(hsdk);
  EXPECT_NEAR(ros2.linear.x, hsdk.linear.x(), 1e-5);
  EXPECT_NEAR(ros2.linear.y, hsdk.linear.y(), 1e-5);
  EXPECT_NEAR(ros2.linear.z, hsdk.linear.z(), 1e-5);
  EXPECT_NEAR(ros2.angular.x, hsdk.angular.x(), 1e-5);
  EXPECT_NEAR(ros2.angular.y, hsdk.angular.y(), 1e-5);
  EXPECT_NEAR(ros2.angular.z, hsdk.angular.z(), 1e-5);
}

TEST(TwistStampedConverter, Ros2ToHsdkFillsHeader) {
  geometry_msgs::msg::TwistStamped msg;
  msg.header.stamp.sec = 1;
  msg.header.frame_id = "base_link";
  msg.twist.linear.x = 2.0;

  const isaac::TwistCpuT hsdk = TwistStampedConverter::to_hsdk(msg);
  EXPECT_DOUBLE_EQ(hsdk.linear.x(), msg.twist.linear.x);
  ASSERT_NE(hsdk.header, nullptr);
  EXPECT_EQ(hsdk.header->capture_timestamp_ns, 1'000'000'000LL);
  EXPECT_EQ(hsdk.header->frame_id, msg.header.frame_id);
}

TEST(TwistStampedConverter, HsdkToRos2CastsAllFields) {
  isaac::TwistCpuT hsdk;
  hsdk.linear = holoscan::schema::Vector3d(1.0, 2.0, 3.0);
  hsdk.angular = holoscan::schema::Vector3d(0.1, 0.2, 0.3);
  hsdk.header = std::make_shared<holoscan::schema::HeaderT>();
  hsdk.header->capture_timestamp_ns = 5'250'000'000LL;
  hsdk.header->frame_id = "base_link";

  const auto ros2 = TwistStampedConverter::to_ros2(hsdk);
  ASSERT_TRUE(ros2.has_value());
  EXPECT_NEAR(ros2->twist.linear.x, hsdk.linear.x(), 1e-5);
  EXPECT_NEAR(ros2->twist.angular.z, hsdk.angular.z(), 1e-5);
  EXPECT_EQ(ros2->header.stamp.sec, 5);
  EXPECT_EQ(ros2->header.stamp.nanosec, 250'000'000U);
  EXPECT_EQ(ros2->header.frame_id, hsdk.header->frame_id);
}

TEST(TwistStampedConverter, HsdkToRos2RejectsACaptureTimestampOutsideInt32Seconds) {
  isaac::TwistCpuT hsdk;
  hsdk.header = std::make_shared<holoscan::schema::HeaderT>();
  hsdk.header->capture_timestamp_ns =
      (static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1) * 1'000'000'000LL;

  const auto ros2 = TwistStampedConverter::to_ros2(hsdk);
  EXPECT_FALSE(ros2.has_value());
}

using TransformConverter = Converter<geometry_msgs::msg::Transform, isaac::TransformCpuT>;
using TransformStampedConverter =
    Converter<geometry_msgs::msg::TransformStamped, isaac::TransformCpuT>;

TEST(TransformConverter, Ros2ToHsdkLeavesHeaderAndChildFrameIdUnset) {
  geometry_msgs::msg::Transform msg;
  msg.translation.x = 1.0;
  msg.rotation.w = 1.0;

  const isaac::TransformCpuT hsdk = TransformConverter::to_hsdk(msg);
  EXPECT_DOUBLE_EQ(hsdk.translation.x(), msg.translation.x);
  EXPECT_EQ(hsdk.header, nullptr);
  EXPECT_TRUE(hsdk.child_frame_id.empty());
}

TEST(TransformStampedConverter, Ros2ToHsdkFillsHeaderAndChildFrameId) {
  geometry_msgs::msg::TransformStamped msg;
  msg.header.stamp.sec = 2;
  msg.header.frame_id = "odom";
  msg.child_frame_id = "base_link";
  msg.transform.translation.z = 0.5;

  const isaac::TransformCpuT hsdk = TransformStampedConverter::to_hsdk(msg);
  EXPECT_DOUBLE_EQ(hsdk.translation.z(), msg.transform.translation.z);
  ASSERT_NE(hsdk.header, nullptr);
  EXPECT_EQ(hsdk.header->frame_id, msg.header.frame_id);
  EXPECT_EQ(hsdk.child_frame_id, msg.child_frame_id);
}

TEST(TransformStampedConverter, HsdkToRos2CastsChildFrameId) {
  isaac::TransformCpuT hsdk;
  hsdk.translation = holoscan::schema::Vector3d(0.0, 0.0, 0.0);
  hsdk.rotation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);
  hsdk.child_frame_id = "base_link";
  hsdk.header = std::make_shared<holoscan::schema::HeaderT>();
  hsdk.header->capture_timestamp_ns = 5'250'000'000LL;
  hsdk.header->frame_id = "odom";

  const auto ros2 = TransformStampedConverter::to_ros2(hsdk);
  ASSERT_TRUE(ros2.has_value());
  EXPECT_EQ(ros2->child_frame_id, hsdk.child_frame_id);
  EXPECT_EQ(ros2->header.stamp.sec, 5);
  EXPECT_EQ(ros2->header.stamp.nanosec, 250'000'000U);
  EXPECT_EQ(ros2->header.frame_id, hsdk.header->frame_id);
}

using OdometryConverter = Converter<nav_msgs::msg::Odometry, isaac::OdometryCpuT>;

TEST(OdometryConverter, Ros2ToHsdkCastsAllFields) {
  nav_msgs::msg::Odometry msg;
  msg.header.stamp.sec = 3;
  msg.header.frame_id = "odom";
  msg.child_frame_id = "base_link";
  msg.pose.pose.position.x = 1.0;
  msg.pose.pose.orientation.w = 1.0;
  msg.pose.covariance.fill(0.0);
  // Index 0 (-> (0, 0)) and n^2 - 1 (-> (n - 1, n - 1)) map to the same position under both
  // row-major and column-major indexing, so they can't distinguish correct row-major handling
  // from an accidentally-transposed one; (0, 1) and (1, 0) (indices 1 and 6) can, and get
  // deliberately different values so a row/column swap changes which one lands where.
  msg.pose.covariance[1] = 0.01;
  msg.pose.covariance[6] = 0.02;
  msg.twist.twist.linear.x = 0.5;
  msg.twist.covariance.fill(0.0);
  msg.twist.covariance[1] = 0.03;
  msg.twist.covariance[6] = 0.04;

  const isaac::OdometryCpuT hsdk = OdometryConverter::to_hsdk(msg);
  ASSERT_NE(hsdk.header, nullptr);
  EXPECT_EQ(hsdk.header->frame_id, msg.header.frame_id);
  EXPECT_EQ(hsdk.child_frame_id, msg.child_frame_id);
  ASSERT_NE(hsdk.pose, nullptr);
  EXPECT_DOUBLE_EQ(hsdk.pose->position.x(), msg.pose.pose.position.x);
  EXPECT_DOUBLE_EQ(hsdk.pose_covariance(0, 1), msg.pose.covariance[1]);
  EXPECT_DOUBLE_EQ(hsdk.pose_covariance(1, 0), msg.pose.covariance[6]);
  ASSERT_NE(hsdk.twist, nullptr);
  EXPECT_DOUBLE_EQ(hsdk.twist->linear.x(), msg.twist.twist.linear.x);
  EXPECT_DOUBLE_EQ(hsdk.twist_covariance(0, 1), msg.twist.covariance[1]);
  EXPECT_DOUBLE_EQ(hsdk.twist_covariance(1, 0), msg.twist.covariance[6]);
}

TEST(OdometryConverter, HsdkToRos2CastsAllFields) {
  isaac::OdometryCpuT hsdk;
  hsdk.header = std::make_shared<holoscan::schema::HeaderT>();
  hsdk.header->capture_timestamp_ns = 5'250'000'000LL;
  hsdk.header->frame_id = "odom";
  hsdk.child_frame_id = "base_link";
  hsdk.pose = std::make_shared<isaac::PoseCpuT>();
  hsdk.pose->position = holoscan::schema::Vector3d(1.0, 0.0, 0.0);
  hsdk.pose->orientation = holoscan::schema::Quaterniond(0.0, 0.0, 0.0, 1.0);
  hsdk.pose_covariance = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();
  // (0, 1) and (1, 0) get deliberately different values, unlike (0, 0)/(n - 1, n - 1), so a
  // row/column swap in to_ros2's mapping changes which ros2 index each one lands at.
  hsdk.pose_covariance(0, 1) = 0.01;
  hsdk.pose_covariance(1, 0) = 0.02;
  hsdk.twist = std::make_shared<isaac::TwistCpuT>();
  hsdk.twist->linear = holoscan::schema::Vector3d(0.5, 0.0, 0.0);
  hsdk.twist_covariance = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();

  const auto ros2 = OdometryConverter::to_ros2(hsdk);
  ASSERT_TRUE(ros2.has_value());
  EXPECT_EQ(ros2->header.stamp.sec, 5);
  EXPECT_EQ(ros2->header.stamp.nanosec, 250'000'000U);
  EXPECT_EQ(ros2->header.frame_id, hsdk.header->frame_id);
  EXPECT_EQ(ros2->child_frame_id, hsdk.child_frame_id);
  EXPECT_NEAR(ros2->pose.pose.position.x, hsdk.pose->position.x(), 1e-5);
  EXPECT_DOUBLE_EQ(ros2->pose.covariance[1], hsdk.pose_covariance(0, 1));
  EXPECT_DOUBLE_EQ(ros2->pose.covariance[6], hsdk.pose_covariance(1, 0));
  EXPECT_NEAR(ros2->twist.twist.linear.x, hsdk.twist->linear.x(), 1e-5);
}

TEST(OdometryConverter, HsdkToRos2WithMissingPoseAndTwistLeavesRos2Defaulted) {
  isaac::OdometryCpuT hsdk;
  // pose/twist (the shared_ptr fields) are the "missing" state under test; pose_covariance is
  // always-present (native_inline) and Eigen's default constructor leaves it uninitialized, so
  // it needs an explicit zero here to isolate that from what's actually being tested.
  hsdk.pose_covariance = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();

  const auto ros2 = OdometryConverter::to_ros2(hsdk);
  ASSERT_TRUE(ros2.has_value());
  EXPECT_NEAR(ros2->pose.pose.position.x, 0.0, 1e-5);
  EXPECT_NEAR(ros2->twist.twist.linear.x, 0.0, 1e-5);
  for (const double covariance_value : ros2->pose.covariance) {
    EXPECT_DOUBLE_EQ(covariance_value, 0.0);
  }
}

using CameraInfoConverter = Converter<sensor_msgs::msg::CameraInfo, isaac::CameraInfoCpuT>;

TEST(CameraInfoConverter, Ros2ToHsdkCastsAllFields) {
  sensor_msgs::msg::CameraInfo msg;
  msg.header.stamp.sec = 7;
  msg.header.frame_id = "camera";
  msg.height = 480U;
  msg.width = 640U;
  msg.distortion_model = "plumb_bob";
  msg.d = {0.1, 0.2};
  // k[2]/k[6] ((0, 2) and (2, 0)) and r[2]/r[6] get deliberately different values, unlike a
  // diagonal position (invariant under transpose) or the identity matrix (symmetric) previously
  // used for r -- either would pass identically whether the row-major mapping below is correct
  // or accidentally transposed.
  msg.k = {600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0};
  msg.r = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  msg.p = {600.0, 0.0, 320.0, 0.0, 0.0, 600.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  msg.binning_x = 1U;
  msg.roi.x_offset = 10U;
  msg.roi.do_rectify = true;

  const isaac::CameraInfoCpuT hsdk = CameraInfoConverter::to_hsdk(msg);
  ASSERT_NE(hsdk.header, nullptr);
  EXPECT_EQ(hsdk.header->frame_id, msg.header.frame_id);
  EXPECT_EQ(hsdk.height, msg.height);
  EXPECT_EQ(hsdk.distortion_model, msg.distortion_model);
  ASSERT_EQ(hsdk.d.size(), msg.d.size());
  EXPECT_DOUBLE_EQ(hsdk.d[1], msg.d[1]);
  // k/r are holoscan.schema.Matrix3d -- a plain flatc struct (not native_type/Eigen-backed), so
  // .m()->Get(flat_index) rather than (row, col); index 2/6 are (0, 2)/(2, 0) row-major.
  EXPECT_DOUBLE_EQ(hsdk.k.m()->Get(2), msg.k[2]);
  EXPECT_DOUBLE_EQ(hsdk.k.m()->Get(6), msg.k[6]);
  EXPECT_DOUBLE_EQ(hsdk.r.m()->Get(2), msg.r[2]);
  EXPECT_DOUBLE_EQ(hsdk.r.m()->Get(6), msg.r[6]);
  // p is non-square (3x4), so unlike k/r a row-major/column-major mixup can't land back on the
  // same position by coincidence -- this single check is already diagnostic on its own.
  EXPECT_DOUBLE_EQ(hsdk.p(2, 2), msg.p[10]);
  EXPECT_EQ(hsdk.roi_x_offset, msg.roi.x_offset);
  EXPECT_EQ(hsdk.roi_do_rectify, msg.roi.do_rectify);
}

TEST(CameraInfoConverter, HsdkToRos2CastsAllFields) {
  isaac::CameraInfoCpuT hsdk;
  hsdk.header = std::make_shared<holoscan::schema::HeaderT>();
  hsdk.header->capture_timestamp_ns = 7'500'000'000LL;
  hsdk.header->frame_id = "camera";
  hsdk.height = 480U;
  hsdk.width = 640U;
  hsdk.distortion_model = "plumb_bob";
  hsdk.d = {0.1, 0.2};
  // k/r are holoscan.schema.Matrix3d -- a plain flatc struct (not native_type/Eigen-backed),
  // constructed from a flat row-major std::array directly. Index 2/6 ((0, 2)/(2, 0)) get
  // deliberately different values so a row/column swap in to_ros2's mapping is visible below.
  hsdk.k = holoscan::schema::Matrix3d(
      std::array<double, 9>{600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0});
  hsdk.r = holoscan::schema::Matrix3d(
      std::array<double, 9>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
  // p is isaac.Matrix3x4d, Eigen::RowMajor -- Eigen's comma initializer fills row-major,
  // matching the wire layout regardless of Eigen's own (column-major, by default) storage order.
  hsdk.p << 600.0, 0.0, 320.0, 0.0, 0.0, 600.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  hsdk.roi_do_rectify = true;

  const auto ros2 = CameraInfoConverter::to_ros2(hsdk);
  ASSERT_TRUE(ros2.has_value());
  EXPECT_EQ(ros2->header.stamp.sec, 7);
  EXPECT_EQ(ros2->header.stamp.nanosec, 500'000'000U);
  EXPECT_EQ(ros2->header.frame_id, hsdk.header->frame_id);
  EXPECT_EQ(ros2->height, hsdk.height);
  EXPECT_EQ(ros2->distortion_model, hsdk.distortion_model);
  ASSERT_EQ(ros2->d.size(), hsdk.d.size());
  EXPECT_DOUBLE_EQ(ros2->d[1], hsdk.d[1]);
  EXPECT_DOUBLE_EQ(ros2->k[2], hsdk.k.m()->Get(2));
  EXPECT_DOUBLE_EQ(ros2->k[6], hsdk.k.m()->Get(6));
  EXPECT_DOUBLE_EQ(ros2->r[2], hsdk.r.m()->Get(2));
  EXPECT_DOUBLE_EQ(ros2->r[6], hsdk.r.m()->Get(6));
  EXPECT_DOUBLE_EQ(ros2->p[10], hsdk.p(2, 2));
  EXPECT_EQ(ros2->roi.do_rectify, hsdk.roi_do_rectify);
}

TEST(CameraInfoConverter, HsdkToRos2WithNoHeaderLeavesRos2HeaderDefaulted) {
  isaac::CameraInfoCpuT hsdk;
  hsdk.k = holoscan::schema::Matrix3d(std::array<double, 9>{});
  hsdk.r = holoscan::schema::Matrix3d(std::array<double, 9>{});
  hsdk.p = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>::Zero();

  const auto ros2 = CameraInfoConverter::to_ros2(hsdk);
  ASSERT_TRUE(ros2.has_value());
  EXPECT_EQ(ros2->header.stamp.sec, 0);
  EXPECT_TRUE(ros2->header.frame_id.empty());
}

TEST(FromRos2Timestamp, CombinesSecondsAndNanoseconds) {
  const auto ns = from_ros2_timestamp(3, 250'000'000U);
  ASSERT_TRUE(ns);
  EXPECT_EQ(*ns, 3'250'000'000);
}

TEST(FromRos2Timestamp, LargestValidStampDoesNotOverflow) {
  const auto ns = from_ros2_timestamp(std::numeric_limits<std::int32_t>::max(), 999'999'999U);
  ASSERT_TRUE(ns);
  EXPECT_EQ(*ns, std::int64_t{2'147'483'647} * 1'000'000'000 + 999'999'999);
}

TEST(FromRos2Timestamp, RejectsNanosecondsAtOrAboveOneSecond) {
  EXPECT_FALSE(from_ros2_timestamp(1, 1'000'000'000U));
  EXPECT_FALSE(from_ros2_timestamp(1, std::numeric_limits<std::uint32_t>::max()));
}

TEST(FromRos2Timestamp, RejectsNegativeSeconds) { EXPECT_FALSE(from_ros2_timestamp(-1, 0U)); }

}  // namespace
}  // namespace isaac::os::ros2_bridge

// Compile-time checks of the converter concepts. The test binary building is the test.
namespace isaac::os::ros2_bridge {
namespace {

// Every converter the bridge ships must satisfy the concept for its kind.
template <typename Pairs>
struct AllShippedConvertersValid;
template <typename... Cs>
struct AllShippedConvertersValid<std::tuple<Cs...>> {
  static constexpr bool value = ((PlainConverter<Cs> || TensorConverterFor<Cs>) && ...);
};
static_assert(AllShippedConvertersValid<DefaultRos2Converter::SupportedPairs>::value,
              "a shipped converter does not satisfy PlainConverter or TensorConverterFor");

// The two ways to_ros2 can return: the message, or a holoscan::expected of it.
static_assert(PlainConverter<Converter<sensor_msgs::msg::JointState, isaac::JointStateCpuT>>);
static_assert(PlainConverter<Converter<geometry_msgs::msg::PoseStamped, isaac::PoseCpuT>>);
static_assert(!TensorConverterFor<Converter<sensor_msgs::msg::JointState, isaac::JointStateCpuT>>);
static_assert(
    TensorConverterFor<TensorConverter<sensor_msgs::msg::Image, holoscan::schema::ImageT>>);
static_assert(!PlainConverter<TensorConverter<sensor_msgs::msg::Image, holoscan::schema::ImageT>>);

struct MissingToRos2 {
  using Ros2Type = sensor_msgs::msg::JointState;
  using HsdkType = isaac::JointStateCpuT;
  static HsdkType to_hsdk(const Ros2Type&) { return {}; }
};
static_assert(!PlainConverter<MissingToRos2>);

struct MissingFromTensors {
  using Ros2Type = sensor_msgs::msg::Image;
  using HsdkType = holoscan::schema::ImageT;
  using Payload = holoscan::schema::DescriptorFlatBufferPayloadFor<holoscan::schema::Image>;
  static holoscan::DescriptorTensorTopicContract<HsdkType> contract(const Ros2TopicSpec&) {
    return {};
  }
  static auto to_tensors(const std::shared_ptr<Ros2Type>&, const Ros2TopicSpec&) {
    return holoscan::expected<std::pair<std::vector<std::byte>, std::vector<holoscan::Tensor>>,
                              holoscan::Error>{};
  }
};
static_assert(!TensorConverterFor<MissingFromTensors>);

struct NoTypeAliases {
  static int to_hsdk(int) { return 0; }
};
static_assert(!PlainConverter<NoTypeAliases>);
static_assert(!TensorConverterFor<NoTypeAliases>);

}  // namespace
}  // namespace isaac::os::ros2_bridge
