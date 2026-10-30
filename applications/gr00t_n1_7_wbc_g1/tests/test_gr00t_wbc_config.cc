/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include <holoscan/config/document.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/params/config.hpp>

#include "applications/gr00t_n1_7_wbc_g1/green_context.h"
#include "applications/gr00t_n1_7_wbc_g1/mock/mock_operators.h"
#include "operators/agile_wbc/history_term_config_codec.h"
#include "operators/agile_wbc/wbc_params.h"
#include "operators/gr00t_action_handler/gr00t_action_handler_op.h"
#include "operators/gr00t_n1_7/gr00t_n1_7_fused_op.h"
#include "operators/gr00t_n1_7/gr00t_n1_7_model_config_codec.h"
#include "operators/joint_command_combiner/joint_command_combiner_op.h"

namespace isaac::os {
namespace {

TEST(Gr00tWbcConfig, DecodesComposedCanonicalConfig) {
  auto application_config =
      holoscan::config_io::load(ISAAC_GR00T_APPLICATION_CONFIG, holoscan::config_io::Format::kYaml);
  EXPECT_EQ(load_wbc_green_context_sms(application_config), std::nullopt);
  auto model_config =
      holoscan::config_io::load(ISAAC_GR00T_MODEL_CONFIG, holoscan::config_io::Format::kYaml);
  auto composed_config = application_config.patched(holoscan::config::Edit::add(
      "/gr00t_n1_7/inference", model_config.at("/gr00t_n1_7").to_value()));

  auto wbc_params = holoscan::params::decode<ops::WbcParams>(composed_config, "/gr00t_wbc");
  auto gr00t_params = holoscan::params::decode<ops::Gr00tN17Params>(composed_config, "/gr00t_n1_7");
  (void)holoscan::params::decode<ops::Gr00tActionHandlerParams>(composed_config,
                                                                "/gr00t_action_handler");
  auto combiner_params = holoscan::params::decode<ops::JointCommandCombinerParams>(
      composed_config, "/joint_command_combiner");
  auto image_params =
      holoscan::params::decode<MockImageParams>(composed_config, "/mock_image_source");

  EXPECT_EQ(wbc_params->sdk_joint_order.size(), 43U);
  EXPECT_EQ(gr00t_params->inference.num_joints, 43);
  EXPECT_EQ(gr00t_params->inference.models.size(), 5U);
  EXPECT_FALSE(gr00t_params->align_observations);
  EXPECT_DOUBLE_EQ(gr00t_params->inference_period, 0.2);
  const auto sim_config = application_config.patched(holoscan::config::Edit::add(
      "/sim/gr00t_n1_7/inference", model_config.at("/gr00t_n1_7").to_value()));
  const auto sim = holoscan::params::decode<ops::Gr00tN17Params>(sim_config, "/sim/gr00t_n1_7");
  EXPECT_TRUE(sim->align_observations);
  EXPECT_DOUBLE_EQ(sim->inference_period, 0.7);
  EXPECT_EQ(combiner_params->joint_order.size(), 43U);
  EXPECT_DOUBLE_EQ(image_params->frame_rate, 30.0);
}

TEST(Gr00tWbcConfig, AcceptsExplicitGreenContextCountsIncludingDisabled) {
  for (const auto sm_count : {0U, 2U, 4U}) {
    const auto config =
        holoscan::config_io::parse("green_context: {wbc_sms: " + std::to_string(sm_count) + "}",
                                   holoscan::config_io::Format::kYaml);
    EXPECT_EQ(load_wbc_green_context_sms(config), sm_count);
    EXPECT_EQ(load_wbc_green_context_sms(config, 0U), 0U);
    EXPECT_EQ(load_wbc_green_context_sms(config, 8U), 8U);
  }
  holoscan::CompileOptions options;
  EXPECT_EQ(configure_wbc_green_context(options, 0U), 0U);
  EXPECT_TRUE(options.deployment.cuda_green_contexts().empty());
}

TEST(Gr00tWbcConfig, MissingGreenContextCountSelectsDeviceDefault) {
  for (const auto* yaml : {"{}", "green_context: {}"}) {
    const auto config = holoscan::config_io::parse(yaml, holoscan::config_io::Format::kYaml);
    EXPECT_EQ(load_wbc_green_context_sms(config), std::nullopt);
    EXPECT_EQ(load_wbc_green_context_sms(config, 0U), 0U);
  }
}

TEST(Gr00tWbcConfig, RejectsInvalidGreenContextCounts) {
  for (const auto* yaml : {"green_context: null", "green_context: 2", "green_context: []",
                           "green_context: {wbc_sms: null}", "green_context: {wbc_sms: -1}",
                           "green_context: {wbc_sms: 1.5}", "green_context: {wbc_sms: auto}",
                           "green_context: {wbc_sms: 4294967296}"}) {
    const auto config = holoscan::config_io::parse(yaml, holoscan::config_io::Format::kYaml);
    EXPECT_THROW(static_cast<void>(load_wbc_green_context_sms(config)), std::runtime_error) << yaml;
  }
}

}  // namespace
}  // namespace isaac::os
