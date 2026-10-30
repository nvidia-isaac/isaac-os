/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_AGILE_WBC_HISTORY_TERM_CONFIG_CODEC_H
#define ISAAC_OS_OPERATORS_AGILE_WBC_HISTORY_TERM_CONFIG_CODEC_H

#include <tuple>

#include <holoscan/config/codec.hpp>

#include "operators/agile_wbc/wbc_params.h"

template <>
struct holoscan::config::ConfigCodec<isaac::os::ops::HistoryTerm>
    : holoscan::config::ObjectCodec<isaac::os::ops::HistoryTerm> {
  static constexpr auto fields = std::tuple{
      object_field("input_name", &isaac::os::ops::HistoryTerm::input_name),
      object_field("output_name", &isaac::os::ops::HistoryTerm::output_name),
      object_field("dim", &isaac::os::ops::HistoryTerm::dim),
      defaulted_object_field("length", &isaac::os::ops::HistoryTerm::length),
  };
};

#endif  // ISAAC_OS_OPERATORS_AGILE_WBC_HISTORY_TERM_CONFIG_CODEC_H
