/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <holoscan/config/io.hpp>
#include <holoscan/params/config.hpp>

#include "common/so101_driver/so101_driver_config.h"
#include "operators/so101/so101_delta_motion_control_op.h"
#include "operators/so101/so101_io_ops.h"
#include "operators/so101_reach/so101_reach_fused_op.h"

namespace isaac::os {
namespace {

TEST(So101ReachParamsConfig, DecodesCanonicalOperatorConfig) {
  auto config = holoscan::config_io::load(SO101_REACH_CONFIG, holoscan::config_io::Format::kYaml);

  (void)holoscan::params::decode<ops::So101StateSourceParams>(config, "/state_source");
  (void)holoscan::params::decode<ops::So101ReachParams>(config, "/policy");
  const auto motion_control =
      holoscan::params::decode<ops::So101DeltaMotionParams>(config, "/motion_control");
  ASSERT_EQ(motion_control->fixed_joint_positions.size(), 2U);
  EXPECT_DOUBLE_EQ(motion_control->fixed_joint_positions[0], 0.0);
  EXPECT_DOUBLE_EQ(motion_control->fixed_joint_positions[1], 0.0);
}

TEST(So101ReachParamsConfig, RejectsUnconfiguredRealRobot) {
  auto config = holoscan::config_io::load(SO101_REACH_CONFIG, holoscan::config_io::Format::kYaml);

  std::string error;
  try {
    (void)decode_so101_driver_options(config);
  } catch (const std::runtime_error& exception) {
    error = exception.what();
  }

  EXPECT_NE(error.find("calibration file"), std::string::npos) << error;
}

}  // namespace
}  // namespace isaac::os
