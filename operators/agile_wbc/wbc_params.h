/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_AGILE_WBC_WBC_PARAMS_H
#define ISAAC_OS_OPERATORS_AGILE_WBC_WBC_PARAMS_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>

namespace isaac::os::ops {

struct HistoryTerm {
  std::string input_name;
  std::string output_name;
  std::int64_t dim{};
  std::int64_t length{};
};

// Translation units that decode WbcParams from configuration must first include
// history_term_config_codec.h, which supplies the element codec for history_terms. The codec stays
// separate so this header does not require configuration support.
struct WbcParams {
  std::string onnx_path;
  std::vector<std::string> sdk_joint_order;
  std::vector<std::string> policy_joint_order;
  std::vector<std::string> action_joint_order;
  std::map<std::string, double> default_joint_positions;
  std::map<std::string, double> default_kp;
  std::map<std::string, double> default_kd;
  double ang_vel_scale{1.0};
  double velocity_scale{1.0};
  double action_clamp{10.0};
  std::string last_actions_input_name{"last_actions"};
  std::string quat_input_name{"root_link_quat_w"};
  std::string ang_vel_input_name{"root_ang_vel_b"};
  std::string velocity_commands_name{"velocity_commands"};
  std::string joint_pos_input_name{"joint_pos"};
  std::string joint_vel_input_name{"joint_vel"};
  std::string action_pos_output_name{"action_joint_pos"};
  std::string action_kp_output_name{"action_joint_pos_kp_gains"};
  std::string action_kd_output_name{"action_joint_pos_kd_gains"};
  double velocity_command_lin_x{};
  double velocity_command_lin_y{};
  double velocity_command_ang_z{};
  std::vector<HistoryTerm> history_terms;
};

HOLOSCAN_PARAM_SCHEMA(
    WbcParams, holoscan::param_schema("com.nvidia.isaac.WbcParams", 1),
    holoscan::field<&WbcParams::onnx_path>("onnx_path"),
    holoscan::field<&WbcParams::sdk_joint_order>("sdk_joint_order"),
    holoscan::field<&WbcParams::policy_joint_order>("policy_joint_order"),
    holoscan::field<&WbcParams::action_joint_order>("action_joint_order"),
    holoscan::field<&WbcParams::default_joint_positions>("default_joint_positions"),
    holoscan::field<&WbcParams::default_kp>("default_kp"),
    holoscan::field<&WbcParams::default_kd>("default_kd"),
    holoscan::field<&WbcParams::ang_vel_scale>("ang_vel_scale"),
    holoscan::field<&WbcParams::velocity_scale>("velocity_scale"),
    holoscan::field<&WbcParams::action_clamp>("action_clamp"),
    holoscan::field<&WbcParams::last_actions_input_name>("last_actions_input_name"),
    holoscan::field<&WbcParams::quat_input_name>("quat_input_name"),
    holoscan::field<&WbcParams::ang_vel_input_name>("ang_vel_input_name"),
    holoscan::field<&WbcParams::velocity_commands_name>("velocity_commands_name"),
    holoscan::field<&WbcParams::joint_pos_input_name>("joint_pos_input_name"),
    holoscan::field<&WbcParams::joint_vel_input_name>("joint_vel_input_name"),
    holoscan::field<&WbcParams::action_pos_output_name>("action_pos_output_name"),
    holoscan::field<&WbcParams::action_kp_output_name>("action_kp_output_name"),
    holoscan::field<&WbcParams::action_kd_output_name>("action_kd_output_name"),
    holoscan::field<&WbcParams::velocity_command_lin_x>("velocity_command_lin_x"),
    holoscan::field<&WbcParams::velocity_command_lin_y>("velocity_command_lin_y"),
    holoscan::field<&WbcParams::velocity_command_ang_z>("velocity_command_ang_z"),
    holoscan::field<&WbcParams::history_terms>("history_terms"));

/// @brief Validate cross-field relationships in an effective WBC parameter aggregate.
/// @throws std::invalid_argument naming the offending field.
void validate_wbc_params(const WbcParams& params);

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_AGILE_WBC_WBC_PARAMS_H
