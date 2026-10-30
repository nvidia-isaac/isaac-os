/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_TELEOP_EVDEV_DEVICE_H
#define ISAAC_OS_OPERATORS_TELEOP_EVDEV_DEVICE_H

#include <array>
#include <cstddef>
#include <string>

struct libevdev;

namespace isaac::os::ops {

struct VelocityAxis {
  int code = 0;
  bool invert = false;
};

enum class DeviceKind { kGamepad, kKeyboard };

class EvdevDevice {
 public:
  static constexpr std::size_t kMaxEventsPerPoll = 128;

  struct Config {
    std::string device;                      ///< evdev path, name substring, or "" to auto-detect
    DeviceKind kind = DeviceKind::kGamepad;  ///< what to auto-detect when \c device is empty
    std::array<VelocityAxis, 3> axes;        ///< [lin_x, lin_y, ang_z]
    double deadzone = 0.10;                  ///< fraction of full deflection treated as zero
  };

  struct Reading {
    bool connected = false;
    std::array<float, 3> axes{0.f, 0.f, 0.f};  ///< normalised [-1, 1], deadzone applied
  };

  ~EvdevDevice();

  bool open_or_defer(const Config& cfg);  ///< open now (deferring if absent); true if opened
  void close();
  Reading poll();
  int vendor_id() const;
  int product_id() const;
  std::string device_name() const;
  /// Input queries return zero/false while disconnected.
  bool button_held(int code) const;  ///< is EV_KEY \c code held?
  int axis_raw(int code) const;      ///< raw EV_ABS \c code value
  /// Normalizes to [-1, 1], applying the deadzone and inversion.
  static float normalize_axis(int raw, int min, int max, int flat, double deadzone, bool invert);

 private:
  bool try_open();

  Config cfg_{};
  libevdev* dev_ = nullptr;
  int reopen_countdown_ = 0;
};

std::string hid_driver_of(const std::string& device_path);
std::string resolve_device_path(const std::string& device_or_name, DeviceKind kind,
                                const std::string& event_dir = "/dev/input");
bool is_ps5_dualsense(int vendor, int product);

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_TELEOP_EVDEV_DEVICE_H
