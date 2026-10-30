/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <vector>

#include "Eigen/Dense"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "camera_info_cpu_schema_traits.h"
#include "common/ros2_bridge/ros2_timestamp.h"
#include "joint_state_cpu_schema_traits.h"
#include "odometry_cpu_schema_traits.h"
#include "pose_cpu_schema_traits.h"
#include "transform_cpu_schema_traits.h"
#include "twist_cpu_schema_traits.h"

namespace isaac::os::ros2_bridge {

// Undefined primary template: instantiating Converter<Ros2T, HsdkT> for a pair with no
// specialization below is a compile error, not a silent runtime gap.
template <typename Ros2T, typename HsdkT>
struct Converter;

// sensor_msgs/msg/JointState <-> isaac.JointStateCpu. Each additional registered pair gets
// its own Converter<Ros2T, HsdkT> specialization below this one, in the same one-pair-per-
// specialization style -- see DefaultRos2Converter::SupportedPairs (default_ros2_converter.h)
// for where new specializations get added to the registered set.
template <>
struct Converter<sensor_msgs::msg::JointState, isaac::JointStateCpuT> {
  using Ros2Type = sensor_msgs::msg::JointState;
  using HsdkType = isaac::JointStateCpuT;

  static isaac::JointStateCpuT to_hsdk(const sensor_msgs::msg::JointState& msg) {
    isaac::JointStateCpuT out;
    out.name = msg.name;
    out.position.assign(msg.position.begin(), msg.position.end());
    out.velocity.assign(msg.velocity.begin(), msg.velocity.end());
    out.effort.assign(msg.effort.begin(), msg.effort.end());
    return out;
  }

  static sensor_msgs::msg::JointState to_ros2(const isaac::JointStateCpuT& msg) {
    sensor_msgs::msg::JointState out;
    out.name = msg.name;
    out.position.assign(msg.position.begin(), msg.position.end());
    out.velocity.assign(msg.velocity.begin(), msg.velocity.end());
    out.effort.assign(msg.effort.begin(), msg.effort.end());
    return out;
  }
};

// sensor_msgs/msg/Imu <-> holoscan.schema.Imu (the SDK's own descriptor-free flatbuffer Imu
// type, from holoscan::sensor_io -- not a project-local schema). All fields round-trip except
// ImuFlags.ACCELEROMETER_SATURATED/GYROSCOPE_SATURATED/UNCALIBRATED, which sensor_msgs/Imu has
// no equivalent for and which to_hsdk() leaves unset.
template <>
struct Converter<sensor_msgs::msg::Imu, holoscan::schema::ImuT> {
  using Ros2Type = sensor_msgs::msg::Imu;
  using HsdkType = holoscan::schema::ImuT;

  static holoscan::schema::Matrix3f to_matrix3f(const std::array<double, 9>& covariance) {
    const std::array<float, 9> m{
        static_cast<float>(covariance[0]), static_cast<float>(covariance[1]),
        static_cast<float>(covariance[2]), static_cast<float>(covariance[3]),
        static_cast<float>(covariance[4]), static_cast<float>(covariance[5]),
        static_cast<float>(covariance[6]), static_cast<float>(covariance[7]),
        static_cast<float>(covariance[8])};
    return holoscan::schema::Matrix3f(m);
  }

  static std::array<double, 9> from_matrix3f(const holoscan::schema::Matrix3f& matrix) {
    std::array<double, 9> out{};
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = static_cast<double>(matrix.m()->Get(static_cast<flatbuffers::uoffset_t>(i)));
    }
    return out;
  }

  static holoscan::schema::ImuT to_hsdk(const sensor_msgs::msg::Imu& msg) {
    holoscan::schema::ImuT out;
    out.header = std::make_shared<holoscan::schema::HeaderT>();
    out.header->capture_timestamp_ns =
        static_cast<std::int64_t>(msg.header.stamp.sec) * kNanosecondsPerSecond +
        static_cast<std::int64_t>(msg.header.stamp.nanosec);
    out.header->frame_id = msg.header.frame_id;
    out.orientation = holoscan::schema::Quaternionf(
        static_cast<float>(msg.orientation.x), static_cast<float>(msg.orientation.y),
        static_cast<float>(msg.orientation.z), static_cast<float>(msg.orientation.w));
    out.angular_velocity = holoscan::schema::Vector3f(static_cast<float>(msg.angular_velocity.x),
                                                      static_cast<float>(msg.angular_velocity.y),
                                                      static_cast<float>(msg.angular_velocity.z));
    out.linear_acceleration =
        holoscan::schema::Vector3f(static_cast<float>(msg.linear_acceleration.x),
                                   static_cast<float>(msg.linear_acceleration.y),
                                   static_cast<float>(msg.linear_acceleration.z));
    // sensor_msgs/Imu's own convention: orientation_covariance[0] == -1 means this source
    // doesn't estimate orientation (a raw gyro/accel-only IMU, per ImuFlags_ORIENTATION_VALID's
    // own doc comment) and `orientation` should not be trusted -- so ORIENTATION_VALID only
    // gets set when that sentinel isn't present.
    if (msg.orientation_covariance[0] != -1.0) {
      out.flags = holoscan::schema::ImuFlags_ORIENTATION_VALID;
    }
    out.orientation_covariance =
        std::make_shared<holoscan::schema::Matrix3f>(to_matrix3f(msg.orientation_covariance));
    out.angular_velocity_covariance =
        std::make_shared<holoscan::schema::Matrix3f>(to_matrix3f(msg.angular_velocity_covariance));
    out.linear_acceleration_covariance = std::make_shared<holoscan::schema::Matrix3f>(
        to_matrix3f(msg.linear_acceleration_covariance));
    return out;
  }

  static holoscan::expected<sensor_msgs::msg::Imu, holoscan::Error> to_ros2(
      const holoscan::schema::ImuT& msg) {
    sensor_msgs::msg::Imu out;
    if (msg.header) {
      const auto stamp = to_ros2_timestamp(msg.header->capture_timestamp_ns);
      if (!stamp) {
        return holoscan::make_unexpected(stamp.error());
      }
      out.header.stamp.sec = stamp->sec;
      out.header.stamp.nanosec = stamp->nanosec;
      out.header.frame_id = msg.header->frame_id;
    }
    out.orientation.x = static_cast<double>(msg.orientation.x());
    out.orientation.y = static_cast<double>(msg.orientation.y());
    out.orientation.z = static_cast<double>(msg.orientation.z());
    out.orientation.w = static_cast<double>(msg.orientation.w());
    out.angular_velocity.x = static_cast<double>(msg.angular_velocity.x());
    out.angular_velocity.y = static_cast<double>(msg.angular_velocity.y());
    out.angular_velocity.z = static_cast<double>(msg.angular_velocity.z());
    out.linear_acceleration.x = static_cast<double>(msg.linear_acceleration.x());
    out.linear_acceleration.y = static_cast<double>(msg.linear_acceleration.y());
    out.linear_acceleration.z = static_cast<double>(msg.linear_acceleration.z());
    if (msg.orientation_covariance) {
      out.orientation_covariance = from_matrix3f(*msg.orientation_covariance);
    }
    // Mirrors to_hsdk()'s sentinel check in reverse: sensor_msgs/Imu's own convention is
    // orientation_covariance[0] == -1 means "this source doesn't estimate orientation, don't
    // trust `orientation`" -- so re-assert it in the outgoing message whenever
    // ORIENTATION_VALID isn't set, regardless of whatever covariance value was carried in.
    if ((msg.flags & holoscan::schema::ImuFlags_ORIENTATION_VALID) == 0) {
      out.orientation_covariance[0] = -1.0;
    }
    if (msg.angular_velocity_covariance) {
      out.angular_velocity_covariance = from_matrix3f(*msg.angular_velocity_covariance);
    }
    if (msg.linear_acceleration_covariance) {
      out.linear_acceleration_covariance = from_matrix3f(*msg.linear_acceleration_covariance);
    }
    return out;
  }
};

// geometry_msgs/msg/Pose <-> isaac.PoseCpu. Pose has no header, so to_hsdk() leaves
// PoseCpuT::header unset and to_ros2() ignores it.
template <>
struct Converter<geometry_msgs::msg::Pose, isaac::PoseCpuT> {
  using Ros2Type = geometry_msgs::msg::Pose;
  using HsdkType = isaac::PoseCpuT;

  static isaac::PoseCpuT to_hsdk(const geometry_msgs::msg::Pose& msg) {
    isaac::PoseCpuT out;
    out.position = holoscan::schema::Vector3d(msg.position.x, msg.position.y, msg.position.z);
    out.orientation = holoscan::schema::Quaterniond(msg.orientation.x, msg.orientation.y,
                                                    msg.orientation.z, msg.orientation.w);
    return out;
  }

  static geometry_msgs::msg::Pose to_ros2(const isaac::PoseCpuT& msg) {
    geometry_msgs::msg::Pose out;
    out.position.x = msg.position.x();
    out.position.y = msg.position.y();
    out.position.z = msg.position.z();
    out.orientation.x = msg.orientation.x();
    out.orientation.y = msg.orientation.y();
    out.orientation.z = msg.orientation.z();
    out.orientation.w = msg.orientation.w();
    return out;
  }
};

// geometry_msgs/msg/PoseStamped <-> isaac.PoseCpu. Shares isaac.PoseCpu with the unstamped Pose
// pair above; this pair additionally round-trips PoseCpuT::header from/to header.stamp/frame_id.
template <>
struct Converter<geometry_msgs::msg::PoseStamped, isaac::PoseCpuT> {
  using Ros2Type = geometry_msgs::msg::PoseStamped;
  using HsdkType = isaac::PoseCpuT;

  static isaac::PoseCpuT to_hsdk(const geometry_msgs::msg::PoseStamped& msg) {
    isaac::PoseCpuT out = Converter<geometry_msgs::msg::Pose, isaac::PoseCpuT>::to_hsdk(msg.pose);
    out.header = std::make_shared<holoscan::schema::HeaderT>();
    out.header->capture_timestamp_ns =
        static_cast<std::int64_t>(msg.header.stamp.sec) * kNanosecondsPerSecond +
        static_cast<std::int64_t>(msg.header.stamp.nanosec);
    out.header->frame_id = msg.header.frame_id;
    return out;
  }

  static holoscan::expected<geometry_msgs::msg::PoseStamped, holoscan::Error> to_ros2(
      const isaac::PoseCpuT& msg) {
    geometry_msgs::msg::PoseStamped out;
    out.pose = Converter<geometry_msgs::msg::Pose, isaac::PoseCpuT>::to_ros2(msg);
    if (msg.header) {
      const auto stamp = to_ros2_timestamp(msg.header->capture_timestamp_ns);
      if (!stamp) {
        return holoscan::make_unexpected(stamp.error());
      }
      out.header.stamp.sec = stamp->sec;
      out.header.stamp.nanosec = stamp->nanosec;
      out.header.frame_id = msg.header->frame_id;
    }
    return out;
  }
};

// geometry_msgs/msg/Twist <-> isaac.TwistCpu. Twist has no header, so to_hsdk() leaves
// TwistCpuT::header unset and to_ros2() ignores it.
template <>
struct Converter<geometry_msgs::msg::Twist, isaac::TwistCpuT> {
  using Ros2Type = geometry_msgs::msg::Twist;
  using HsdkType = isaac::TwistCpuT;

  static isaac::TwistCpuT to_hsdk(const geometry_msgs::msg::Twist& msg) {
    isaac::TwistCpuT out;
    out.linear = holoscan::schema::Vector3d(msg.linear.x, msg.linear.y, msg.linear.z);
    out.angular = holoscan::schema::Vector3d(msg.angular.x, msg.angular.y, msg.angular.z);
    return out;
  }

  static geometry_msgs::msg::Twist to_ros2(const isaac::TwistCpuT& msg) {
    geometry_msgs::msg::Twist out;
    out.linear.x = msg.linear.x();
    out.linear.y = msg.linear.y();
    out.linear.z = msg.linear.z();
    out.angular.x = msg.angular.x();
    out.angular.y = msg.angular.y();
    out.angular.z = msg.angular.z();
    return out;
  }
};

// geometry_msgs/msg/TwistStamped <-> isaac.TwistCpu. Shares isaac.TwistCpu with the unstamped
// Twist pair above; this pair additionally round-trips TwistCpuT::header from/to
// header.stamp/frame_id.
template <>
struct Converter<geometry_msgs::msg::TwistStamped, isaac::TwistCpuT> {
  using Ros2Type = geometry_msgs::msg::TwistStamped;
  using HsdkType = isaac::TwistCpuT;

  static isaac::TwistCpuT to_hsdk(const geometry_msgs::msg::TwistStamped& msg) {
    isaac::TwistCpuT out =
        Converter<geometry_msgs::msg::Twist, isaac::TwistCpuT>::to_hsdk(msg.twist);
    out.header = std::make_shared<holoscan::schema::HeaderT>();
    out.header->capture_timestamp_ns =
        static_cast<std::int64_t>(msg.header.stamp.sec) * kNanosecondsPerSecond +
        static_cast<std::int64_t>(msg.header.stamp.nanosec);
    out.header->frame_id = msg.header.frame_id;
    return out;
  }

  static holoscan::expected<geometry_msgs::msg::TwistStamped, holoscan::Error> to_ros2(
      const isaac::TwistCpuT& msg) {
    geometry_msgs::msg::TwistStamped out;
    out.twist = Converter<geometry_msgs::msg::Twist, isaac::TwistCpuT>::to_ros2(msg);
    if (msg.header) {
      const auto stamp = to_ros2_timestamp(msg.header->capture_timestamp_ns);
      if (!stamp) {
        return holoscan::make_unexpected(stamp.error());
      }
      out.header.stamp.sec = stamp->sec;
      out.header.stamp.nanosec = stamp->nanosec;
      out.header.frame_id = msg.header->frame_id;
    }
    return out;
  }
};

// geometry_msgs/msg/Transform <-> isaac.TransformCpu. Transform has no header or
// child_frame_id, so to_hsdk() leaves TransformCpuT::header/child_frame_id unset and to_ros2()
// ignores them.
template <>
struct Converter<geometry_msgs::msg::Transform, isaac::TransformCpuT> {
  using Ros2Type = geometry_msgs::msg::Transform;
  using HsdkType = isaac::TransformCpuT;

  static isaac::TransformCpuT to_hsdk(const geometry_msgs::msg::Transform& msg) {
    isaac::TransformCpuT out;
    out.translation =
        holoscan::schema::Vector3d(msg.translation.x, msg.translation.y, msg.translation.z);
    out.rotation = holoscan::schema::Quaterniond(msg.rotation.x, msg.rotation.y, msg.rotation.z,
                                                 msg.rotation.w);
    return out;
  }

  static geometry_msgs::msg::Transform to_ros2(const isaac::TransformCpuT& msg) {
    geometry_msgs::msg::Transform out;
    out.translation.x = msg.translation.x();
    out.translation.y = msg.translation.y();
    out.translation.z = msg.translation.z();
    out.rotation.x = msg.rotation.x();
    out.rotation.y = msg.rotation.y();
    out.rotation.z = msg.rotation.z();
    out.rotation.w = msg.rotation.w();
    return out;
  }
};

// geometry_msgs/msg/TransformStamped <-> isaac.TransformCpu. Shares isaac.TransformCpu with the
// unstamped Transform pair above; this pair additionally round-trips
// TransformCpuT::header/child_frame_id from/to header.stamp/frame_id/child_frame_id.
template <>
struct Converter<geometry_msgs::msg::TransformStamped, isaac::TransformCpuT> {
  using Ros2Type = geometry_msgs::msg::TransformStamped;
  using HsdkType = isaac::TransformCpuT;

  static isaac::TransformCpuT to_hsdk(const geometry_msgs::msg::TransformStamped& msg) {
    isaac::TransformCpuT out =
        Converter<geometry_msgs::msg::Transform, isaac::TransformCpuT>::to_hsdk(msg.transform);
    out.header = std::make_shared<holoscan::schema::HeaderT>();
    out.header->capture_timestamp_ns =
        static_cast<std::int64_t>(msg.header.stamp.sec) * kNanosecondsPerSecond +
        static_cast<std::int64_t>(msg.header.stamp.nanosec);
    out.header->frame_id = msg.header.frame_id;
    out.child_frame_id = msg.child_frame_id;
    return out;
  }

  static holoscan::expected<geometry_msgs::msg::TransformStamped, holoscan::Error> to_ros2(
      const isaac::TransformCpuT& msg) {
    geometry_msgs::msg::TransformStamped out;
    out.transform = Converter<geometry_msgs::msg::Transform, isaac::TransformCpuT>::to_ros2(msg);
    out.child_frame_id = msg.child_frame_id;
    if (msg.header) {
      const auto stamp = to_ros2_timestamp(msg.header->capture_timestamp_ns);
      if (!stamp) {
        return holoscan::make_unexpected(stamp.error());
      }
      out.header.stamp.sec = stamp->sec;
      out.header.stamp.nanosec = stamp->nanosec;
      out.header.frame_id = msg.header->frame_id;
    }
    return out;
  }
};

// nav_msgs/msg/Odometry <-> isaac.OdometryCpu. pose/twist convert through the same
// position/orientation and linear/angular field mapping as
// Converter<geometry_msgs::msg::Pose, isaac::PoseCpuT>/Converter<geometry_msgs::msg::Twist,
// isaac::TwistCpuT> above, duplicated here since OdometryCpu.pose/twist are unstamped (this
// table's own header is the single source of timestamp/frame for the whole message).
template <>
struct Converter<nav_msgs::msg::Odometry, isaac::OdometryCpuT> {
  using Ros2Type = nav_msgs::msg::Odometry;
  using HsdkType = isaac::OdometryCpuT;

  static isaac::OdometryCpuT to_hsdk(const nav_msgs::msg::Odometry& msg) {
    isaac::OdometryCpuT out;
    out.header = std::make_shared<holoscan::schema::HeaderT>();
    out.header->capture_timestamp_ns =
        static_cast<std::int64_t>(msg.header.stamp.sec) * kNanosecondsPerSecond +
        static_cast<std::int64_t>(msg.header.stamp.nanosec);
    out.header->frame_id = msg.header.frame_id;
    out.child_frame_id = msg.child_frame_id;

    out.pose = std::make_shared<isaac::PoseCpuT>();
    out.pose->position = holoscan::schema::Vector3d(
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z);
    out.pose->orientation =
        holoscan::schema::Quaterniond(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                      msg.pose.pose.orientation.z, msg.pose.pose.orientation.w);
    // isaac.Matrix6d is Eigen::RowMajor -- the same layout as ROS2's row-major covariance
    // array, so Eigen::Map reinterprets it directly instead of copying element-by-element.
    out.pose_covariance =
        Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(msg.pose.covariance.data());

    out.twist = std::make_shared<isaac::TwistCpuT>();
    out.twist->linear = holoscan::schema::Vector3d(
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z);
    out.twist->angular = holoscan::schema::Vector3d(
        msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z);
    out.twist_covariance =
        Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(msg.twist.covariance.data());
    return out;
  }

  // pose/twist/header are `required` in odometry_cpu.fbs, so flatc refuses to serialize an
  // OdometryCpuT missing them -- but this function also runs on plain in-memory structs that
  // never went through that serialization path, so the null checks below guard a state the
  // schema forbids on the wire but the C++ type itself doesn't.
  static holoscan::expected<nav_msgs::msg::Odometry, holoscan::Error> to_ros2(
      const isaac::OdometryCpuT& msg) {
    nav_msgs::msg::Odometry out;
    if (msg.header) {
      const auto stamp = to_ros2_timestamp(msg.header->capture_timestamp_ns);
      if (!stamp) {
        return holoscan::make_unexpected(stamp.error());
      }
      out.header.stamp.sec = stamp->sec;
      out.header.stamp.nanosec = stamp->nanosec;
      out.header.frame_id = msg.header->frame_id;
    }
    out.child_frame_id = msg.child_frame_id;

    if (msg.pose) {
      out.pose.pose.position.x = msg.pose->position.x();
      out.pose.pose.position.y = msg.pose->position.y();
      out.pose.pose.position.z = msg.pose->position.z();
      out.pose.pose.orientation.x = msg.pose->orientation.x();
      out.pose.pose.orientation.y = msg.pose->orientation.y();
      out.pose.pose.orientation.z = msg.pose->orientation.z();
      out.pose.pose.orientation.w = msg.pose->orientation.w();
    }
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(out.pose.covariance.data()) =
        msg.pose_covariance;

    if (msg.twist) {
      out.twist.twist.linear.x = msg.twist->linear.x();
      out.twist.twist.linear.y = msg.twist->linear.y();
      out.twist.twist.linear.z = msg.twist->linear.z();
      out.twist.twist.angular.x = msg.twist->angular.x();
      out.twist.twist.angular.y = msg.twist->angular.y();
      out.twist.twist.angular.z = msg.twist->angular.z();
    }
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(out.twist.covariance.data()) =
        msg.twist_covariance;
    return out;
  }
};

// sensor_msgs/msg/CameraInfo <-> isaac.CameraInfoCpu. Every field round-trips directly. k/r/p
// are fixed-size in both ROS2 (std::array<double, 9>/std::array<double, 9>/std::array<double,
// 12>) and isaac.CameraInfoCpu (holoscan.schema.Matrix3d/Matrix3d/isaac.Matrix3x4d) -- the
// size/shape mismatch this used to defend against with zero-pad/truncate logic is now a
// compile-time impossibility instead of a runtime-corrected edge case. k/r use
// holoscan.schema.Matrix3d (added upstream alongside Vector3d/Quaterniond) rather than a
// project-local type, since that shape is already covered there -- unlike Matrix3x4d (p), it's
// a plain flatc struct (no native_type/Eigen), so it round-trips through element-by-element
// .m()->Get(i) access instead of Eigen::Map, same as holoscan.schema.Matrix3f elsewhere in this
// file.
template <>
struct Converter<sensor_msgs::msg::CameraInfo, isaac::CameraInfoCpuT> {
  using Ros2Type = sensor_msgs::msg::CameraInfo;
  using HsdkType = isaac::CameraInfoCpuT;

  static std::array<double, 9> from_matrix3d(const holoscan::schema::Matrix3d& matrix) {
    std::array<double, 9> out{};
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = matrix.m()->Get(static_cast<flatbuffers::uoffset_t>(i));
    }
    return out;
  }

  static isaac::CameraInfoCpuT to_hsdk(const sensor_msgs::msg::CameraInfo& msg) {
    isaac::CameraInfoCpuT out;
    out.header = std::make_shared<holoscan::schema::HeaderT>();
    out.header->capture_timestamp_ns =
        static_cast<std::int64_t>(msg.header.stamp.sec) * kNanosecondsPerSecond +
        static_cast<std::int64_t>(msg.header.stamp.nanosec);
    out.header->frame_id = msg.header.frame_id;
    out.height = msg.height;
    out.width = msg.width;
    out.distortion_model = msg.distortion_model;
    out.d.assign(msg.d.begin(), msg.d.end());
    out.k = holoscan::schema::Matrix3d(msg.k);
    out.r = holoscan::schema::Matrix3d(msg.r);
    // isaac.Matrix3x4d is Eigen::RowMajor, matching ROS2's row-major p array, so Eigen::Map
    // reinterprets the buffer directly instead of copying element-by-element.
    out.p = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(msg.p.data());
    out.binning_x = msg.binning_x;
    out.binning_y = msg.binning_y;
    out.roi_x_offset = msg.roi.x_offset;
    out.roi_y_offset = msg.roi.y_offset;
    out.roi_height = msg.roi.height;
    out.roi_width = msg.roi.width;
    out.roi_do_rectify = msg.roi.do_rectify;
    return out;
  }

  static holoscan::expected<sensor_msgs::msg::CameraInfo, holoscan::Error> to_ros2(
      const isaac::CameraInfoCpuT& msg) {
    sensor_msgs::msg::CameraInfo out;
    if (msg.header) {
      const auto stamp = to_ros2_timestamp(msg.header->capture_timestamp_ns);
      if (!stamp) {
        return holoscan::make_unexpected(stamp.error());
      }
      out.header.stamp.sec = stamp->sec;
      out.header.stamp.nanosec = stamp->nanosec;
      out.header.frame_id = msg.header->frame_id;
    }
    out.height = msg.height;
    out.width = msg.width;
    out.distortion_model = msg.distortion_model;
    out.d.assign(msg.d.begin(), msg.d.end());
    out.k = from_matrix3d(msg.k);
    out.r = from_matrix3d(msg.r);
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(out.p.data()) = msg.p;
    out.binning_x = msg.binning_x;
    out.binning_y = msg.binning_y;
    out.roi.x_offset = msg.roi_x_offset;
    out.roi.y_offset = msg.roi_y_offset;
    out.roi.height = msg.roi_height;
    out.roi.width = msg.roi_width;
    out.roi.do_rectify = msg.roi_do_rectify;
    return out;
  }
};

}  // namespace isaac::os::ros2_bridge
