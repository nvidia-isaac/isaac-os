/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <tuple>
#include <type_traits>

#include "common/ros2_bridge/converter.h"
#include "common/ros2_bridge/tensor_converter.h"

namespace isaac::os::ros2_bridge {

// The built-in set of registered (ROS2 type, hsdk type) pairs. Downstream code adds custom
// types by subclassing this and extending SupportedPairs with tuple_cat (see README.md's
// "Adding a custom converter") -- never by editing this file.
struct DefaultRos2Converter {
  using SupportedPairs =
      std::tuple<Converter<sensor_msgs::msg::JointState, isaac::JointStateCpuT>,
                 Converter<sensor_msgs::msg::Imu, holoscan::schema::ImuT>,
                 Converter<geometry_msgs::msg::Pose, isaac::PoseCpuT>,
                 Converter<geometry_msgs::msg::PoseStamped, isaac::PoseCpuT>,
                 Converter<geometry_msgs::msg::Twist, isaac::TwistCpuT>,
                 Converter<geometry_msgs::msg::TwistStamped, isaac::TwistCpuT>,
                 Converter<geometry_msgs::msg::Transform, isaac::TransformCpuT>,
                 Converter<geometry_msgs::msg::TransformStamped, isaac::TransformCpuT>,
                 Converter<nav_msgs::msg::Odometry, isaac::OdometryCpuT>,
                 Converter<sensor_msgs::msg::CameraInfo, isaac::CameraInfoCpuT>,
                 TensorConverter<sensor_msgs::msg::Image, holoscan::schema::ImageT>>;
};

namespace detail {

// Counts how many of SupportedPairs's tuple-element types have both Ros2Type == Ros2T and
// HsdkType == HsdkT, entirely at compile time.
template <typename SupportedPairs, typename Ros2T, typename HsdkT>
struct MatchCount;

template <typename Ros2T, typename HsdkT, typename... Pairs>
struct MatchCount<std::tuple<Pairs...>, Ros2T, HsdkT> {
  static constexpr int value = (0 + ... +
                                ((std::is_same_v<typename Pairs::Ros2Type, Ros2T> &&
                                  std::is_same_v<typename Pairs::HsdkType, HsdkT>)
                                     ? 1
                                     : 0));
};

}  // namespace detail

// Resolves <Ros2T, HsdkT> against SupportedPairs and invokes fn(traits) with the matching
// Converter<Ros2T, HsdkT>/TensorConverter<Ros2T, HsdkT> instance. static_asserts exactly one
// pair matches -- "no matching converter" and "ambiguous registration" are compile errors here.
template <typename SupportedPairs, typename Ros2T, typename HsdkT, typename Fn>
void dispatch_by_type_pair(Fn&& fn) {
  constexpr int match_count = detail::MatchCount<SupportedPairs, Ros2T, HsdkT>::value;
  static_assert(match_count != 0,
                "No ROS2 converter registered for this (Ros2Type, HsdkType) pair -- add one to "
                "your ConverterT::SupportedPairs.");
  static_assert(match_count <= 1,
                "Ambiguous ROS2 converter registration: more than one converter in "
                "SupportedPairs matches this (Ros2Type, HsdkType) pair.");
  std::apply(
      [&](auto&&... traits) {
        auto try_one = [&](auto&& t) {
          using Traits = std::decay_t<decltype(t)>;
          if constexpr (std::is_same_v<typename Traits::Ros2Type, Ros2T> &&
                        std::is_same_v<typename Traits::HsdkType, HsdkT>) {
            fn(t);
          }
        };
        (try_one(traits), ...);
      },
      SupportedPairs{});
}

}  // namespace isaac::os::ros2_bridge
