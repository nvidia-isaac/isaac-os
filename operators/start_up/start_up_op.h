/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_START_UP_START_UP_OP_H
#define ISAAC_OS_OPERATORS_START_UP_START_UP_OP_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "joint_command_cpu_schema_traits.h"
#include "joint_state_cpu_schema_traits.h"
#include "operators/start_up/waypoint_sequencer.h"

namespace isaac::os::ops {

struct StartUpWaypoint {
  std::string name;
  std::map<std::string, double, std::less<>> joint_positions;
  double duration{};
};

struct StartUpParams {
  std::vector<std::string> sdk_joint_order;
  std::optional<std::map<std::string, double, std::less<>>> kp;
  std::optional<std::map<std::string, double, std::less<>>> kd;
  std::optional<std::map<std::string, double, std::less<>>> hold_kp;
  std::optional<std::map<std::string, double, std::less<>>> hold_kd;
  std::vector<std::string> upper_body_joints;
  /// Seconds to wait at the final pose before automatic enable. Ignored when enable inputs exist.
  double auto_enable_delay{};
  double control_dt{};
  double dwell_duration{};
  /// Maximum settled joint speed, in rad/s.
  double vel_settled_threshold{};
  std::vector<StartUpWaypoint> waypoints;
  /// Skip waypoints and blend once from measured pose into incoming commands.
  /// When enable inputs are connected, wait for both enables before starting the handover.
  /// Hold the measured upper-body pose until a non-degraded command and fresh state are available.
  /// Later upper-body enables toggle between that startup pose and incoming commands.
  bool handover_only{};
  /// Seconds for the one-time smoothstep handover.
  double handover_duration{2.0};
};

HOLOSCAN_PARAM_SCHEMA(
    StartUpParams, holoscan::param_schema("com.nvidia.isaac.StartUpParams", 1),
    holoscan::field<&StartUpParams::sdk_joint_order>("sdk_joint_order"),
    holoscan::field<&StartUpParams::kp>("kp"), holoscan::field<&StartUpParams::kd>("kd"),
    holoscan::field<&StartUpParams::hold_kp>("hold_kp"),
    holoscan::field<&StartUpParams::hold_kd>("hold_kd"),
    holoscan::field<&StartUpParams::upper_body_joints>("upper_body_joints"),
    holoscan::field<&StartUpParams::auto_enable_delay>("auto_enable_delay"),
    holoscan::field<&StartUpParams::control_dt>("control_dt"),
    holoscan::field<&StartUpParams::dwell_duration>("dwell_duration"),
    holoscan::field<&StartUpParams::vel_settled_threshold>("vel_settled_threshold"),
    holoscan::field<&StartUpParams::waypoints>("waypoints"),
    holoscan::field<&StartUpParams::handover_only>("handover_only"),
    holoscan::field<&StartUpParams::handover_duration>("handover_duration"));

/// Executes startup waypoints, then gates lower- and upper-body control.
class StartUpOp final : public holoscan::Operator<StartUpParams> {
 public:
  StartUpOp() = default;
  /// Registers the optional teleop enable inputs.
  explicit StartUpOp(bool enable_signal_connected)
      : enable_signal_connected_(enable_signal_connected) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::JointCommandCpuT> joint_command;
  holoscan::Input<isaac::JointStateCpuT> joint_state;
  holoscan::Input<bool> enable_lower;
  holoscan::Input<bool> enable_upper;
  holoscan::Output<isaac::JointCommandCpuT> command;

 private:
  enum class State : std::uint8_t { kStartingUp, kHandingOver, kRunning };
  // Limits position slew when toggling the upper-body hold.
  struct UpperBodyHold {
    static constexpr float kMaxVelocity = 0.5F;
    static constexpr float kMaxAcceleration = 1.0F;
    static constexpr float kArrived = 1.0e-4F;  ///< rad; within this, land and stop.

    std::vector<float> final_pose;  ///< Pose the upper body holds.
    std::vector<float> held_pose;   ///< Last emitted upper-body command; the slew state.
    bool has_any{};                 ///< Any upper-body joints are configured.
    bool released{};                ///< The upper-body enable toggled the hold off.
    bool active{true};              ///< False once released and settled back onto the controller.
    float speed{};                  ///< Slew speed of the fastest held joint, in rad/s.
    float weight{1.0F};             ///< 1 while fully held, 0 while fully following the controller.
    std::int64_t last_tick_ns{-1};
  };

  [[nodiscard]] std::size_t joint_count() const { return params().sdk_joint_order.size(); }
  [[nodiscard]] bool command_covers_every_joint(const isaac::JointCommandCpuT& value) const;
  holoscan::expected<void, holoscan::Error> compute_impl(holoscan::ExecutionContext& context);

  void initialize_upper_body();
  void advance_handover(const isaac::JointCommandCpuT& value, std::int64_t now_ns);
  bool prepare_handover(const isaac::JointCommandCpuT& value,
                        const holoscan::SampleMetadata& metadata);
  void cache_measured_state();
  void latch_enables();
  /// Apply upper-body control; return false when an invalid target is replaced with a held pose.
  bool update_upper_body(isaac::JointCommandCpuT& value, std::int64_t capture_ns,
                         std::int64_t now_ns);
  void advance_upper_hold(const isaac::JointCommandCpuT& value, std::int64_t now_ns);
  void hold_upper(isaac::JointCommandCpuT& value) const;
  void end_startup_if_enabled_or_delayed(std::int64_t now_ns);

  bool enable_signal_connected_{};

  std::unique_ptr<WaypointSequencer> sequencer_;
  UpperBodyHold upper_body_;
  std::vector<float> kp_;
  std::vector<float> kd_;
  std::vector<float> hold_kp_;
  std::vector<float> hold_kd_;
  std::vector<std::uint8_t> upper_body_mask_;
  std::int64_t auto_enable_delay_ns_{};

  State state_{State::kStartingUp};
  bool enable_latched_{};
  std::int64_t final_reached_ns_{-1};

  std::vector<float> measured_position_;
  std::vector<float> measured_velocity_;
  bool joint_state_seen_{};
  std::int64_t measured_time_ns_{-1};
  std::int64_t handover_start_ns_{};
  std::int64_t last_enable_lower_ns_{};
  std::int64_t last_enable_upper_ns_{};

  bool warned_command_size_{};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_START_UP_START_UP_OP_H
