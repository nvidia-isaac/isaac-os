/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <memory>

#include "holoscan/channel/tensor_schema_endpoints.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"

#include "common/ros2_bridge/ros2_topic_spec.h"

namespace isaac::os::ros2_bridge {

// The two aliases the registry needs to match a converter to a (Ros2Type, HsdkType) pair.
template <typename C>
concept ConverterTypes = requires {
  typename C::Ros2Type;
  typename C::HsdkType;
};

// A converter for a plain flatbuffer HsdkType: one value in, one value out. to_ros2 returns the
// message directly, or a holoscan::expected of it when the conversion can fail (Imu, PoseStamped,
// TwistStamped, TransformStamped, Odometry and CameraInfo do).
template <typename C>
concept PlainConverter = ConverterTypes<C> && requires(const typename C::Ros2Type& ros2_msg,
                                                       const typename C::HsdkType& hsdk_msg) {
  { C::to_hsdk(ros2_msg) } -> std::convertible_to<typename C::HsdkType>;
  requires std::convertible_to<decltype(C::to_ros2(hsdk_msg)), typename C::Ros2Type> ||
               std::same_as<decltype(C::to_ros2(hsdk_msg)),
                            holoscan::expected<typename C::Ros2Type, holoscan::Error>>;
};

// A converter for a descriptor-tensor-schema HsdkType: pixels travel as tensors.
template <typename C>
concept TensorConverterFor =
    ConverterTypes<C> && requires { typename C::Payload; } &&
    requires(const Ros2TopicSpec& spec, const std::shared_ptr<typename C::Ros2Type>& ros2_msg,
             const holoscan::ChannelResolvedTensorBundle<typename C::Payload>& bundle) {
      C::contract(spec);
      C::to_tensors(ros2_msg, spec);
      C::from_tensors(bundle, spec);
    };

}  // namespace isaac::os::ros2_bridge
