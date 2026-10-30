/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_START_UP_START_UP_PARAMS_CONFIG_CODEC_H
#define ISAAC_OS_OPERATORS_START_UP_START_UP_PARAMS_CONFIG_CODEC_H

#include <tuple>

#include <holoscan/config/codec.hpp>

#include "operators/start_up/start_up_op.h"

template <>
struct holoscan::config::ConfigCodec<isaac::os::ops::StartUpWaypoint>
    : holoscan::config::ObjectCodec<isaac::os::ops::StartUpWaypoint> {
  static constexpr auto fields = std::tuple{
      object_field("name", &isaac::os::ops::StartUpWaypoint::name),
      defaulted_object_field("joint_positions", &isaac::os::ops::StartUpWaypoint::joint_positions),
      object_field("duration", &isaac::os::ops::StartUpWaypoint::duration),
  };
};

#endif  // ISAAC_OS_OPERATORS_START_UP_START_UP_PARAMS_CONFIG_CODEC_H
