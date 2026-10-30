/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/teleop/teleop_twist_command_op.h"

#include <linux/input-event-codes.h>

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {
namespace {

constexpr std::size_t kAxisCount = 3U;
const TeleopGamepadMapping kHidGenericGamepad{
    .axes = {TeleopAxis{ABS_Y, true}, TeleopAxis{ABS_X, true}, TeleopAxis{ABS_Z, true}},
    .deadman = TeleopDeadman{ABS_RX, 200},
    .lower_body_enable_button = BTN_WEST,
    .upper_body_enable_button = BTN_Z,
};
const TeleopGamepadMapping kHidPlaystationGamepad{
    .axes = {TeleopAxis{ABS_Y, true}, TeleopAxis{ABS_X, true}, TeleopAxis{ABS_RX, true}},
    .deadman = TeleopDeadman{ABS_Z, 200},
    .lower_body_enable_button = BTN_TL,
    .upper_body_enable_button = BTN_TR,
};

std::array<VelocityAxis, kAxisCount> to_device_axes(const TeleopGamepadMapping& mapping) {
  std::array<VelocityAxis, kAxisCount> axes{};
  for (std::size_t index = 0; index < kAxisCount && index < mapping.axes.size(); ++index) {
    axes[index] =
        VelocityAxis{static_cast<int>(mapping.axes[index].code), mapping.axes[index].invert};
  }
  return axes;
}

}  // namespace

const TeleopGamepadMapping* default_gamepad_mapping(std::string_view driver_name) {
  if (driver_name == "playstation") {
    return &kHidPlaystationGamepad;
  }
  if (driver_name == "hid-generic") {
    return &kHidGenericGamepad;
  }
  return nullptr;
}

std::array<float, 3> keyboard_velocity(bool forward, bool back, bool left, bool right,
                                       bool turn_left, bool turn_right, bool stop, float lin_speed,
                                       float ang_speed) {
  if (stop) {
    return {0.0F, 0.0F, 0.0F};
  }
  return {
      forward == back ? 0.0F : (forward ? lin_speed : -lin_speed),
      left == right ? 0.0F : (left ? lin_speed : -lin_speed),
      turn_left == turn_right ? 0.0F : (turn_left ? ang_speed : -ang_speed),
  };
}

void TeleopTwistCommandOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(velocity_commands, "out_velocity_commands");
  spec.output(enable_lower_body, "out_enable_lower_body");
  spec.output(enable_upper_body, "out_enable_upper_body");
}

holoscan::Contract TeleopTwistCommandOp::contract() const {
  const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(params().emit_period));
  holoscan::Contract contract;
  contract.trigger(holoscan::OnClock{.period = period});
  return contract;
}

void TeleopTwistCommandOp::start() {
  const TeleopParams& config = params();
  if (config.emit_period <= 0.0) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "The teleop command interval must be a positive duration.");
  }
  const bool device_configured = !config.gamepad_device.empty();
  const std::string device = resolve_device_path(config.gamepad_device, DeviceKind::kGamepad);
  if (device_configured && device.empty()) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kNotFound,
                                 "No gamepad matches the configured device \"" +
                                     config.gamepad_device +
                                     "\". Check the name or path, for example with evtest or "
                                     "/proc/bus/input/devices, and that "
                                     "your user is in the 'input' group.");
  }

  const bool choose_from_driver = !config.use_configured_gamepad_mapping && !device.empty();
  const std::string driver = choose_from_driver ? hid_driver_of(device) : std::string{};
  const TeleopGamepadMapping* driver_default =
      choose_from_driver ? default_gamepad_mapping(driver) : nullptr;

  mapping_ = {};
  if (config.use_configured_gamepad_mapping) {
    if (config.gamepad_mapping.axes.size() != kAxisCount) {
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kInvalidArgument,
          "A configured gamepad layout must name an axis for forward, sideways, and turn.");
    }
    mapping_ = config.gamepad_mapping;
  } else if (driver_default != nullptr) {
    mapping_ = *driver_default;
  }
  // Re-resolve the configured name on reconnect.
  const EvdevDevice::Config gamepad_config{
      .device = config.gamepad_device,
      .kind = DeviceKind::kGamepad,
      .axes = to_device_axes(mapping_),
      .deadzone = config.gamepad_deadzone,
  };
  const bool opened = gamepad_.open_or_defer(gamepad_config);

  if (choose_from_driver) {
    std::string problem;
    if (!opened) {
      problem =
          "The gamepad at " + device +
          " could not be opened to identify it. Check the name or path, for example with evtest or "
          "/proc/bus/input/devices, and that your user is in the 'input' group";
    } else if (!is_ps5_dualsense(gamepad_.vendor_id(), gamepad_.product_id())) {
      problem = "The controller at " + device + " (\"" + gamepad_.device_name() +
                "\") is not a recognised model, so a built-in button layout would be wrong. "
                "Configure the button layout for this controller";
    } else if (driver_default == nullptr) {
      const std::string driver_name = driver.empty() ? "unknown" : driver;
      problem = "The controller at " + device + " is claimed by the '" + driver_name +
                "' kernel driver, which has no built-in button layout. Configure the button layout "
                "for this controller";
    }
    if (!problem.empty()) {
      if (device_configured) {
        throw std::runtime_error(problem + ".");
      }
      HOLOSCAN_LOG_WARN("{}. The detected gamepad is ignored.", problem);
      mapping_ = {};
    }
  }

  const EvdevDevice::Config keyboard_config{
      .device = config.keyboard_device,
      .kind = DeviceKind::kKeyboard,
  };
  keyboard_.open_or_defer(keyboard_config);

  arming_latch_ = {};
  upper_gamepad_was_pressed_ = false;
  upper_keyboard_was_pressed_ = false;
  lower_keyboard_was_pressed_ = false;

  HOLOSCAN_LOG_INFO(
      "Teleop is ready. On the gamepad, press the lower-body enable to arm, then hold the motion "
      "gate to drive. When the gamepad is disarmed, W, S, A, D, Q, and E drive, and Space "
      "zeros keyboard velocity. P enables the lower body; O toggles the upper-body hold.");
}

void TeleopTwistCommandOp::stop() {
  gamepad_.close();
  keyboard_.close();
}

std::array<float, 3> TeleopTwistCommandOp::gamepad_command(
    const EvdevDevice::Reading& reading) const {
  const auto lin_speed = static_cast<float>(params().lin_speed);
  const auto ang_speed = static_cast<float>(params().ang_speed);
  return {reading.axes[0] * lin_speed, reading.axes[1] * lin_speed, reading.axes[2] * ang_speed};
}

std::array<float, 3> TeleopTwistCommandOp::keyboard_command() const {
  return keyboard_velocity(keyboard_.button_held(KEY_W), keyboard_.button_held(KEY_S),
                           keyboard_.button_held(KEY_A), keyboard_.button_held(KEY_D),
                           keyboard_.button_held(KEY_Q), keyboard_.button_held(KEY_E),
                           keyboard_.button_held(KEY_SPACE), static_cast<float>(params().lin_speed),
                           static_cast<float>(params().ang_speed));
}

bool TeleopTwistCommandOp::lower_body_enable_pressed(const EvdevDevice::Reading& reading) {
  bool pressed = false;

  const bool gamepad_pressed =
      gamepad_.button_held(static_cast<int>(mapping_.lower_body_enable_button));
  if (arming_latch_.update(gamepad_pressed, reading.connected)) {
    if (arming_latch_.armed()) {
      HOLOSCAN_LOG_INFO("The gamepad is armed and the lower body is enabled.");
      pressed = true;
    } else {
      HOLOSCAN_LOG_INFO(
          "The gamepad is disarmed because {}.",
          reading.connected ? "the enable was pressed again" : "the gamepad was disconnected");
    }
  }

  const bool keyboard_pressed = keyboard_.button_held(KEY_P);
  if (keyboard_pressed && !lower_keyboard_was_pressed_) {
    HOLOSCAN_LOG_INFO("The lower body is enabled.");
    pressed = true;
  }
  lower_keyboard_was_pressed_ = keyboard_pressed;
  return pressed;
}

bool TeleopTwistCommandOp::upper_body_enable_pressed() {
  bool pressed = false;

  const bool gamepad_pressed =
      gamepad_.button_held(static_cast<int>(mapping_.upper_body_enable_button));
  if (gamepad_pressed && !upper_gamepad_was_pressed_) {
    pressed = true;
  }
  upper_gamepad_was_pressed_ = gamepad_pressed;

  const bool keyboard_pressed = keyboard_.button_held(KEY_O);
  if (keyboard_pressed && !upper_keyboard_was_pressed_) {
    pressed = true;
  }
  upper_keyboard_was_pressed_ = keyboard_pressed;
  return pressed;
}

holoscan::expected<void, holoscan::Error> TeleopTwistCommandOp::compute(
    holoscan::ExecutionContext& context) {
  auto now = context.clock().now();
  if (!now) {
    return holoscan::make_unexpected(std::move(now).error());
  }
  const holoscan::EmitOptions options{.capture_time = *now};

  const EvdevDevice::Reading reading = gamepad_.poll();
  keyboard_.poll();  // Drain events so the held-key state is current.
  // Update the arming latch before selecting a velocity source.
  if (lower_body_enable_pressed(reading) && emit_enable_signals_) {
    auto emitted = enable_lower_body.emit(true, options);
    if (!emitted) {
      return emitted;
    }
  }
  if (upper_body_enable_pressed() && emit_enable_signals_) {
    auto emitted = enable_upper_body.emit(true, options);
    if (!emitted) {
      return emitted;
    }
  }
  const bool gate_held = gamepad_.axis_raw(static_cast<int>(mapping_.deadman.axis)) >
                         static_cast<int>(mapping_.deadman.threshold);
  const bool armed = reading.connected && arming_latch_.armed();
  // An armed gamepad with the deadman released commands zero velocity.
  std::array<float, 3> velocity{};
  if (!armed) {
    velocity = keyboard_command();
  } else if (gate_held) {
    velocity = gamepad_command(reading);
  }

  isaac::TwistCpuT twist;
  twist.linear = holoscan::schema::Vector3d{velocity[0], velocity[1], 0.0F};
  twist.angular = holoscan::schema::Vector3d{0.0F, 0.0F, velocity[2]};
  return velocity_commands.emit(std::move(twist), options);
}

}  // namespace isaac::os::ops
