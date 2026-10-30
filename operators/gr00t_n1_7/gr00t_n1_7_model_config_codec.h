/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_GR00T_N1_7_GR00T_N1_7_MODEL_CONFIG_CODEC_H
#define ISAAC_OS_OPERATORS_GR00T_N1_7_GR00T_N1_7_MODEL_CONFIG_CODEC_H

#include <tuple>

#include <holoscan/config/codec.hpp>
#include <holoscan/params/config.hpp>

#include "operators/gr00t_n1_7/gr00t_n1_7_fused_op.h"

// A model is only a nested YAML value, so define its three fields directly.
template <>
struct holoscan::config::ConfigCodec<isaac::os::ops::Gr00tN17Model>
    : holoscan::config::ObjectCodec<isaac::os::ops::Gr00tN17Model> {
  static constexpr auto fields = std::tuple{
      object_field("path", &isaac::os::ops::Gr00tN17Model::path),
      object_field("inputs", &isaac::os::ops::Gr00tN17Model::inputs),
      object_field("outputs", &isaac::os::ops::Gr00tN17Model::outputs),
  };
};

// Inference parameters have an HSDK parameter schema, so reuse it and require every field.
template <>
struct holoscan::config::ConfigCodec<isaac::os::ops::Gr00tN17InferenceParams>
    : holoscan::params::RequiredParamSchemaCodec<isaac::os::ops::Gr00tN17InferenceParams> {};

static_assert(holoscan::config::ConfigCodable<isaac::os::ops::Gr00tN17Model>);
static_assert(holoscan::config::ConfigCodable<isaac::os::ops::Gr00tN17InferenceParams>);

#endif  // ISAAC_OS_OPERATORS_GR00T_N1_7_GR00T_N1_7_MODEL_CONFIG_CODEC_H
