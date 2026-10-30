/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_TELEOP_TELEOP_PARAMS_CONFIG_CODEC_H
#define ISAAC_OS_OPERATORS_TELEOP_TELEOP_PARAMS_CONFIG_CODEC_H

#include <tuple>

#include <holoscan/config/codec.hpp>

#include "operators/teleop/teleop_twist_command_op.h"

namespace holoscan::config {

template <>
struct ConfigCodec<isaac::os::ops::TeleopAxis> : ObjectCodec<isaac::os::ops::TeleopAxis> {
  static ConfigResult<isaac::os::ops::TeleopAxis> decode(ConfigView view,
                                                         const DecodeOptions& options,
                                                         DecodeContext& context);

  static constexpr auto fields = std::tuple{
      object_field("code", &isaac::os::ops::TeleopAxis::code),
      defaulted_object_field("invert", &isaac::os::ops::TeleopAxis::invert),
  };
};

template <>
struct ConfigCodec<isaac::os::ops::TeleopDeadman> : ObjectCodec<isaac::os::ops::TeleopDeadman> {
  static ConfigResult<isaac::os::ops::TeleopDeadman> decode(ConfigView view,
                                                            const DecodeOptions& options,
                                                            DecodeContext& context);

  static constexpr auto fields = std::tuple{
      object_field("axis", &isaac::os::ops::TeleopDeadman::axis),
      defaulted_object_field("threshold", &isaac::os::ops::TeleopDeadman::threshold),
  };
};

template <>
struct ConfigCodec<isaac::os::ops::TeleopGamepadMapping>
    : ObjectCodec<isaac::os::ops::TeleopGamepadMapping> {
  static ConfigResult<isaac::os::ops::TeleopGamepadMapping> decode(ConfigView view,
                                                                   const DecodeOptions& options,
                                                                   DecodeContext& context);

  static constexpr auto fields = std::tuple{
      object_field("axes", &isaac::os::ops::TeleopGamepadMapping::axes),
      object_field("deadman", &isaac::os::ops::TeleopGamepadMapping::deadman),
      object_field("lower_body_enable_button",
                   &isaac::os::ops::TeleopGamepadMapping::lower_body_enable_button),
      defaulted_object_field("upper_body_enable_button",
                             &isaac::os::ops::TeleopGamepadMapping::upper_body_enable_button),
  };
};

}  // namespace holoscan::config

#endif  // ISAAC_OS_OPERATORS_TELEOP_TELEOP_PARAMS_CONFIG_CODEC_H
