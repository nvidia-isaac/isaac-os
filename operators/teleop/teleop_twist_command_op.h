/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_TELEOP_TELEOP_TWIST_COMMAND_OP_H
#define ISAAC_OS_OPERATORS_TELEOP_TELEOP_TWIST_COMMAND_OP_H

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "operators/teleop/evdev_device.h"
#include "twist_cpu_schema_traits.h"

namespace isaac::os::ops {

struct TeleopAxis {
  std::int64_t code{};
  bool invert{true};
};

struct TeleopDeadman {
  std::int64_t axis{};
  std::int64_t threshold{200};
};

struct TeleopGamepadMapping {
  /// Forward, sideways, and turn, in that order.
  std::vector<TeleopAxis> axes;
  TeleopDeadman deadman;
  std::int64_t lower_body_enable_button{};
  /// Zero disables the gamepad upper-body enable.
  std::int64_t upper_body_enable_button{};
};

struct TeleopParams {
  double lin_speed{0.8};     ///< m/s
  double ang_speed{0.5};     ///< rad/s
  double emit_period{0.02};  ///< seconds
  /// Device path or name; empty auto-detects.
  std::string gamepad_device;
  double gamepad_deadzone{0.10};  ///< Fraction of full stick deflection.
  /// Device path or name; empty auto-detects.
  std::string keyboard_device;
  bool use_configured_gamepad_mapping{};
  TeleopGamepadMapping gamepad_mapping;
};

HOLOSCAN_PARAM_SCHEMA(TeleopParams, holoscan::param_schema("com.nvidia.isaac.TeleopParams", 1),
                      holoscan::field<&TeleopParams::lin_speed>("lin_speed"),
                      holoscan::field<&TeleopParams::ang_speed>("ang_speed"),
                      holoscan::field<&TeleopParams::emit_period>("emit_period"),
                      holoscan::field<&TeleopParams::gamepad_device>("gamepad_device"),
                      holoscan::field<&TeleopParams::gamepad_deadzone>("gamepad_deadzone"),
                      holoscan::field<&TeleopParams::keyboard_device>("keyboard_device"),
                      holoscan::field<&TeleopParams::use_configured_gamepad_mapping>(
                          "use_configured_gamepad_mapping"),
                      holoscan::field<&TeleopParams::gamepad_mapping>("gamepad_mapping"));
[[nodiscard]] const TeleopGamepadMapping* default_gamepad_mapping(std::string_view driver_name);
[[nodiscard]] std::array<float, 3> keyboard_velocity(bool forward, bool back, bool left, bool right,
                                                     bool turn_left, bool turn_right, bool stop,
                                                     float lin_speed, float ang_speed);
/// Toggles on enable presses and disarms on disconnect.
class GamepadArmingLatch {
 public:
  bool update(bool enable_pressed, bool connected) {
    const bool was_armed = armed_;
    if (!connected) {
      armed_ = false;  // Fail safe.
    } else if (enable_pressed && !enable_was_pressed_) {
      armed_ = !armed_;
    }
    enable_was_pressed_ = enable_pressed;
    return armed_ != was_armed;
  }

  [[nodiscard]] bool armed() const { return armed_; }

 private:
  bool armed_{};
  bool enable_was_pressed_{};
};

/// An armed gamepad overrides the keyboard; its motion gate must be held to move.
class TeleopTwistCommandOp final : public holoscan::Operator<TeleopParams> {
 public:
  /// Sim applications arm locally without emitting real-robot startup gates.
  explicit TeleopTwistCommandOp(bool emit_enable_signals = true)
      : emit_enable_signals_(emit_enable_signals) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Output<isaac::TwistCpuT> velocity_commands;
  holoscan::Output<bool> enable_lower_body;
  holoscan::Output<bool> enable_upper_body;

 private:
  bool emit_enable_signals_{true};

  [[nodiscard]] std::array<float, 3> gamepad_command(const EvdevDevice::Reading& reading) const;
  [[nodiscard]] std::array<float, 3> keyboard_command() const;
  [[nodiscard]] bool lower_body_enable_pressed(const EvdevDevice::Reading& reading);
  [[nodiscard]] bool upper_body_enable_pressed();

  EvdevDevice gamepad_;
  EvdevDevice keyboard_;
  GamepadArmingLatch arming_latch_;
  TeleopGamepadMapping mapping_;

  bool upper_gamepad_was_pressed_{};
  bool upper_keyboard_was_pressed_{};
  bool lower_keyboard_was_pressed_{};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_TELEOP_TELEOP_TWIST_COMMAND_OP_H
