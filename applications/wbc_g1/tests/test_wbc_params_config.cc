/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <holoscan/config/io.hpp>
#include <holoscan/params/config.hpp>

#include "operators/agile_wbc/history_term_config_codec.h"
#include "operators/agile_wbc/wbc_params.h"

namespace isaac::os {
namespace {

TEST(WbcParamsConfig, DecodesCanonicalConfig) {
  auto document =
      holoscan::config_io::load(ISAAC_WBC_CANONICAL_CONFIG, holoscan::config_io::Format::kYaml);

  auto params = holoscan::params::decode<ops::WbcParams>(document, "/gr00t_wbc");

  EXPECT_EQ(params->sdk_joint_order.size(), 43U);
  ASSERT_EQ(params->history_terms.size(), 7U);
  EXPECT_EQ(params->history_terms.front().input_name, "base_ang_vel_history");
}

}  // namespace
}  // namespace isaac::os
