/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/config/io.hpp>
#include <holoscan/params/config.hpp>

#include "common/so101_driver/so101_driver_config.h"
#include "operators/simple_motion_planner/simple_motion_planner.h"
#include "operators/sinus_motion/sinus_motion_op.h"
#include "operators/so101/so101_io_ops.h"

namespace isaac::os {
namespace {

TEST(So101SinusParamsConfig, DecodesCanonicalOperatorConfig) {
  auto config = holoscan::config_io::load(SO101_SINUS_CONFIG, holoscan::config_io::Format::kYaml);

  (void)holoscan::params::decode<ops::So101StateSourceParams>(config, "/state_source");
  const auto sinus = holoscan::params::decode<ops::SinusMotionParams>(config, "/sinus_motion_op");
  (void)holoscan::params::decode<ops::SimpleMotionPlannerParams>(config, "/motion_planner_op");

  EXPECT_EQ(config.get<int>("/real_robot/position_p_gain"), 16);
  EXPECT_EQ(config.get<int>("/real_robot/position_i_gain"), 0);
  EXPECT_EQ(config.get<int>("/real_robot/position_d_gain"), 32);

  const auto home = config.get<std::vector<double>>("/real_robot/home_position");
  ASSERT_EQ(home.size(), sinus->default_pos.size());
  for (std::size_t index = 0; index < home.size(); ++index) {
    EXPECT_DOUBLE_EQ(home[index], sinus->default_pos[index]);
  }
}

TEST(So101SinusParamsConfig, RejectsUnconfiguredRealRobot) {
  auto config = holoscan::config_io::load(SO101_SINUS_CONFIG, holoscan::config_io::Format::kYaml);

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
