/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <linux/input-event-codes.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>
#include <holoscan/config/codec.hpp>
#include <holoscan/config/document.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/params/config.hpp>

#include "applications/gr00t_n1_7_wbc_g1/green_context.h"
#include "operators/agile_wbc/history_term_config_codec.h"
#include "operators/g1_low_level/g1_params.h"
#include "operators/gr00t_action_handler/gr00t_action_handler_op.h"
#include "operators/gr00t_n1_7/gr00t_n1_7_fused_op.h"
#include "operators/gr00t_n1_7/gr00t_n1_7_model_config_codec.h"
#include "operators/start_up/start_up_params_config_codec.h"
#include "operators/teleop/teleop_params_config_codec.h"
#include "operators/v4l2_camera/v4l2_camera_op.h"

namespace isaac::os {
namespace {

constexpr std::size_t kG1JointCount = 43U;

holoscan::config::Document load_application() {
  const auto app =
      holoscan::config_io::load(ISAAC_GR00T_WBC_CONFIG, holoscan::config_io::Format::kYaml);
  const auto routine = holoscan::config_io::load(app.get<std::string>("/start_up/config_file"),
                                                 holoscan::config_io::Format::kYaml);
  const auto operator_config = app.patched(holoscan::config::Edit::remove("/start_up/config_file"));
  return operator_config.patched(
      holoscan::config::Edit::merge_object("/start_up", routine.root().to_value()));
}

class RealConfigTest : public testing::Test {
 protected:
  holoscan::PreparedParams<ops::TeleopParams> load(std::string_view mapping) {
    std::ifstream source{ISAAC_GR00T_WBC_CONFIG};
    std::string config{std::istreambuf_iterator<char>{source}, std::istreambuf_iterator<char>{}};
    constexpr std::string_view marker = "gamepad:\n";
    const auto position = config.find(marker);
    if (position == std::string::npos) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kNotFound,
                                   "The test configuration has no gamepad section.");
    }
    config.insert(position + marker.size(), mapping);
    return holoscan::params::decode<ops::TeleopParams>(
        holoscan::config_io::parse(config, holoscan::config_io::Format::kYaml), "/gamepad");
  }
};

constexpr std::string_view kCustomMapping = R"(  use_configured_gamepad_mapping: true
  gamepad_mapping:
    axes:
      - {code: ABS_Y, invert: false}
      - {code: ABS_X}
      - {code: ABS_RX}
    deadman: {axis: ABS_Z}
    lower_body_enable_button: BTN_TL
)";

TEST_F(RealConfigTest, LoadsTheNativeCustomLayout) {
  const auto params = load(std::string{kCustomMapping} + "    upper_body_enable_button: BTN_TR\n");
  EXPECT_TRUE(params->use_configured_gamepad_mapping);
  const auto& mapping = params->gamepad_mapping;
  ASSERT_EQ(mapping.axes.size(), 3U);
  EXPECT_EQ(mapping.axes[0].code, ABS_Y);
  EXPECT_FALSE(mapping.axes[0].invert);
  EXPECT_EQ(mapping.axes[1].code, ABS_X);
  EXPECT_TRUE(mapping.axes[1].invert);
  EXPECT_EQ(mapping.axes[2].code, ABS_RX);
  EXPECT_EQ(mapping.deadman.axis, ABS_Z);
  EXPECT_EQ(mapping.deadman.threshold, 200);
  EXPECT_EQ(mapping.lower_body_enable_button, BTN_TL);
  EXPECT_EQ(mapping.upper_body_enable_button, BTN_TR);
}

TEST_F(RealConfigTest, AllowsALayoutWithoutAnUpperBodyButton) {
  EXPECT_EQ(load(kCustomMapping)->gamepad_mapping.upper_body_enable_button, 0);
}

TEST_F(RealConfigTest, RejectsIncompleteOrUnknownInputMappings) {
  EXPECT_THROW(load("  gamepad_mapping: {lower_body_enable_button: BTN_TL}\n"), std::runtime_error);
  EXPECT_THROW(load(std::string{kCustomMapping} + "    upper_body_enable_button: NOT_A_BUTTON\n"),
               std::runtime_error);
}

TEST_F(RealConfigTest, AcceptsNumericInputCodes) {
  const auto params = load(R"(  use_configured_gamepad_mapping: true
  gamepad_mapping:
    axes:
      - {code: 1, invert: false}
      - {code: 0}
      - {code: 3}
    deadman: {axis: 2, threshold: 128}
    lower_body_enable_button: 310
    upper_body_enable_button: 0
)");
  ASSERT_EQ(params->gamepad_mapping.axes.size(), 3U);
  EXPECT_EQ(params->gamepad_mapping.axes[0].code, ABS_Y);
  EXPECT_EQ(params->gamepad_mapping.axes[1].code, ABS_X);
  EXPECT_EQ(params->gamepad_mapping.axes[2].code, ABS_RX);
  EXPECT_EQ(params->gamepad_mapping.deadman.axis, ABS_Z);
  EXPECT_EQ(params->gamepad_mapping.deadman.threshold, 128);
  EXPECT_EQ(params->gamepad_mapping.lower_body_enable_button, BTN_TL);
  EXPECT_EQ(params->gamepad_mapping.upper_body_enable_button, 0);
}

TEST_F(RealConfigTest, RejectsInvalidNumericAxisCodes) {
  for (const std::string_view code : {"-1", "11", "64", "4294967296"}) {
    SCOPED_TRACE(code);
    std::string mapping{kCustomMapping};
    mapping.replace(mapping.find("ABS_Y"), 5, code);
    EXPECT_THROW(load(mapping), std::runtime_error);
  }
}

TEST_F(RealConfigTest, RejectsInvalidNumericButtonCodes) {
  for (const std::string_view code : {"-1", "84", "768", "4294967606"}) {
    SCOPED_TRACE(code);
    EXPECT_THROW(load(std::string{kCustomMapping} +
                      "    upper_body_enable_button: " + std::string{code} + "\n"),
                 std::runtime_error);
  }
}

TEST_F(RealConfigTest, RejectsInputNamesOfTheWrongKindAndUnknownFields) {
  std::string mapping{kCustomMapping};
  mapping.replace(mapping.find("ABS_Y"), 5, "BTN_TL");
  EXPECT_THROW(load(mapping), std::runtime_error);
  EXPECT_THROW(load(std::string{kCustomMapping} + "    upper_body_enable_button: ABS_X\n"),
               std::runtime_error);
  EXPECT_THROW(load(std::string{kCustomMapping} + "    unexpected: 1\n"), std::runtime_error);
}

TEST(RealConfig, LoadsSharedWbcShoulderDefaults) {
  const auto wbc = holoscan::params::decode<ops::WbcParams>(load_application(), "/gr00t_wbc");
  EXPECT_DOUBLE_EQ(wbc->default_joint_positions.at("left_shoulder_roll_joint"), 0.25);
  EXPECT_DOUBLE_EQ(wbc->default_joint_positions.at("left_shoulder_yaw_joint"), 0.5);
  EXPECT_DOUBLE_EQ(wbc->default_joint_positions.at("right_shoulder_roll_joint"), -0.25);
  EXPECT_DOUBLE_EQ(wbc->default_joint_positions.at("right_shoulder_yaw_joint"), -0.5);
}

TEST(RealConfig, LoadsTheProductionRobotSetup) {
  const auto app = load_application();
  const auto dds = holoscan::params::decode<ops::G1DdsParams>(app, "/unitree_g1_robot");
  EXPECT_EQ(dds->g1_num_motors, static_cast<std::int64_t>(kG1JointCount));
  EXPECT_EQ(dds->dds_domain_id, 0);
  EXPECT_FALSE(dds->network_interface.empty());
  EXPECT_FALSE(load_wbc_green_context_sms(app).has_value());
}

TEST(RealConfig, LoadsTheCameraTheModelWasTrainedAgainst) {
  const auto camera =
      holoscan::params::decode<ops::V4l2CameraParams>(load_application(), "/camera");
  EXPECT_EQ(camera->width, 640);
  EXPECT_EQ(camera->height, 480);
  EXPECT_EQ(camera->frame_rate, 30);
  EXPECT_FALSE(camera->device.empty());
}

TEST(RealConfig, LoadsTheStartupRoutineWithNativeParameters) {
  const auto start_up =
      holoscan::params::decode<ops::StartUpParams>(load_application(), "/start_up");
  EXPECT_EQ(start_up->sdk_joint_order.size(), kG1JointCount);
  ASSERT_TRUE(start_up->kp.has_value());
  ASSERT_TRUE(start_up->kd.has_value());
  EXPECT_DOUBLE_EQ(start_up->kp->at(".*"), 100.0);
  EXPECT_DOUBLE_EQ(start_up->kp->at("left_hand_.*"), 1.0);
  EXPECT_DOUBLE_EQ(start_up->kd->at(".*"), 2.0);
  EXPECT_DOUBLE_EQ(start_up->control_dt, 0.02);
  EXPECT_DOUBLE_EQ(start_up->dwell_duration, 0.5);
  EXPECT_DOUBLE_EQ(start_up->auto_enable_delay, 60.0);
  ASSERT_EQ(start_up->waypoints.size(), 3U);
  EXPECT_EQ(start_up->waypoints.back().name, "start_up");
  EXPECT_DOUBLE_EQ(start_up->waypoints.back().duration, 2.0);
  EXPECT_DOUBLE_EQ(start_up->waypoints.back().joint_positions.at(".*_knee_joint"), 0.6235);
  EXPECT_EQ(start_up->upper_body_joints.size(), 31U);
}

TEST(RealConfig, PacesPlaybackAndTeleopAtTheControlRate) {
  const auto app = load_application();
  const auto teleop = holoscan::params::decode<ops::TeleopParams>(app, "/gamepad");
  const auto action =
      holoscan::params::decode<ops::Gr00tActionHandlerParams>(app, "/gr00t_action_handler");
  EXPECT_DOUBLE_EQ(teleop->emit_period, 0.004);
  EXPECT_DOUBLE_EQ(action->action_interval_s, 0.0333333);
  EXPECT_GT(action->handover_blend_steps, 0.0);
  EXPECT_TRUE(action->feedforward_torque);
}

TEST(SimConfig, UsesTheSimulationActionIntervalWithTheSharedActionHandler) {
  const auto action = holoscan::params::decode<ops::Gr00tActionHandlerParams>(
      load_application(), "/sim/gr00t_action_handler");
  EXPECT_DOUBLE_EQ(action->action_interval_s, 0.02);
  EXPECT_DOUBLE_EQ(action->handover_blend_steps, 3.0);
  EXPECT_TRUE(action->feedforward_torque);
}

TEST(RealConfig, LoadsWbcAndGr00tWithTheSharedModelSchemas) {
  const auto app = load_application();
  const auto model =
      holoscan::config_io::load(ISAAC_GR00T_MODEL_CONFIG, holoscan::config_io::Format::kYaml);
  const auto composed = app.patched(
      holoscan::config::Edit::add("/gr00t_n1_7/inference", model.at("/gr00t_n1_7").to_value()));
  const auto wbc = holoscan::params::decode<ops::WbcParams>(composed, "/gr00t_wbc");
  const auto gr00t = holoscan::params::decode<ops::Gr00tN17Params>(composed, "/gr00t_n1_7");
  EXPECT_EQ(wbc->sdk_joint_order, gr00t->sdk_joint_order);
  EXPECT_EQ(wbc->history_terms.size(), 7U);
  EXPECT_EQ(gr00t->inference.num_joints, static_cast<int>(kG1JointCount));
  EXPECT_EQ(gr00t->inference.models.size(), 5U);
  EXPECT_FALSE(gr00t->align_observations);
  EXPECT_DOUBLE_EQ(gr00t->inference_period, 0.2);
}

}  // namespace
}  // namespace isaac::os
