/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/teleop/evdev_device.h"

#include <dirent.h>
#include <fcntl.h>
#include <libevdev/libevdev.h>
#include <linux/input.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <filesystem>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {

constexpr int kReopenIntervalTicks = 50;  // ticks between reopen attempts while disconnected

EvdevDevice::~EvdevDevice() { close(); }

bool EvdevDevice::open_or_defer(const Config& cfg) {
  cfg_ = cfg;
  if (!try_open()) {
    const std::string_view kind = cfg_.kind == DeviceKind::kGamepad ? "gamepad" : "keyboard";
    HOLOSCAN_LOG_WARN(
        "No {} detected for \"{}\"; will retry (list devices with evtest or "
        "/proc/bus/input/devices)",
        kind, cfg_.device);
    reopen_countdown_ = kReopenIntervalTicks;
    return false;
  }
  return true;
}

void EvdevDevice::close() {
  if (!dev_) {
    return;
  }
  const int fd = libevdev_get_fd(dev_);
  libevdev_free(dev_);
  dev_ = nullptr;
  if (fd >= 0) {
    ::close(fd);
  }
}

bool EvdevDevice::try_open() {
  const std::string path = resolve_device_path(cfg_.device, cfg_.kind);
  if (path.empty()) {
    return false;
  }
  const int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK);
  if (fd < 0) {
    return false;
  }
  if (libevdev_new_from_fd(fd, &dev_) < 0) {
    dev_ = nullptr;
    ::close(fd);
    return false;
  }
  HOLOSCAN_LOG_INFO("Opened evdev device {} (\"{}\")", path, libevdev_get_name(dev_));
  return true;
}

float EvdevDevice::normalize_axis(int raw, int min, int max, int flat, double deadzone,
                                  bool invert) {
  if (max <= min) {
    return 0.f;
  }
  const float half = 0.5f * static_cast<float>(max - min);
  const float value = std::clamp(
      (static_cast<float>(raw) - 0.5f * static_cast<float>(min + max)) / half, -1.f, 1.f);
  const float dz = std::clamp(
      std::max(static_cast<float>(deadzone), static_cast<float>(flat) / half), 0.f, 0.99f);
  const float magnitude = std::fabs(value);
  if (magnitude <= dz) {
    return 0.f;
  }
  const float scaled = (magnitude - dz) / (1.f - dz);  // rescale: just past dz -> ~0, full -> 1
  return (value < 0.f) != invert ? -scaled : scaled;
}

EvdevDevice::Reading EvdevDevice::poll() {
  Reading reading;
  if (!dev_) {
    if (--reopen_countdown_ <= 0) {  // an empty (auto-detect) config retries too
      reopen_countdown_ = kReopenIntervalTicks;
      try_open();
    }
    if (!dev_) {
      return reading;
    }
  }

  input_event ev;
  int return_code = LIBEVDEV_READ_STATUS_SUCCESS;
  std::size_t drained = 0;
  do {
    return_code = libevdev_next_event(dev_, LIBEVDEV_READ_FLAG_NORMAL, &ev);
    while (return_code == LIBEVDEV_READ_STATUS_SYNC) {
      ++drained;
      if (drained >= kMaxEventsPerPoll) {
        break;
      }
      return_code = libevdev_next_event(dev_, LIBEVDEV_READ_FLAG_SYNC, &ev);
    }
    if (return_code != LIBEVDEV_READ_STATUS_SUCCESS) {
      break;
    }
    ++drained;
  } while (drained < kMaxEventsPerPoll);

  if (return_code == -ENODEV) {  // real disconnect: fail safe, then defer to periodic reopen
    HOLOSCAN_LOG_WARN("evdev device disconnected — zeroing command and will retry");
    close();
    reopen_countdown_ = kReopenIntervalTicks;
    return reading;
  }

  reading.connected = true;
  for (std::size_t i = 0; i < reading.axes.size(); ++i) {
    const int raw = libevdev_get_event_value(dev_, EV_ABS, cfg_.axes[i].code);
    const input_absinfo* info = libevdev_get_abs_info(dev_, cfg_.axes[i].code);
    reading.axes[i] = info ? normalize_axis(raw, info->minimum, info->maximum, info->flat,
                                            cfg_.deadzone, cfg_.axes[i].invert)
                           : 0.f;
  }
  return reading;
}

int EvdevDevice::axis_raw(int code) const {
  return dev_ ? libevdev_get_event_value(dev_, EV_ABS, code) : 0;
}

bool EvdevDevice::button_held(int code) const {
  return dev_ && libevdev_get_event_value(dev_, EV_KEY, code) != 0;
}

int EvdevDevice::vendor_id() const { return dev_ ? libevdev_get_id_vendor(dev_) : 0; }

int EvdevDevice::product_id() const { return dev_ ? libevdev_get_id_product(dev_) : 0; }

std::string EvdevDevice::device_name() const {
  const char* name = dev_ ? libevdev_get_name(dev_) : nullptr;
  return name ? std::string(name) : std::string();
}

namespace {
std::string realpath_str(const std::string& path) {
  std::error_code error;
  const auto resolved = std::filesystem::canonical(path, error);
  return error ? std::string{} : resolved.string();
}

std::string basename_of(const std::string& path) {
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

bool contains_ci(std::string_view haystack, std::string_view needle) {
  const auto match = std::ranges::search(haystack, needle, [](unsigned char a, unsigned char b) {
    return std::tolower(a) == std::tolower(b);
  });
  return match.begin() != haystack.end();
}
struct DeviceCandidate {
  std::string path;
  std::string name;
  bool matches(std::string_view query) const { return contains_ci(name, query); }
  bool matches_exactly(std::string_view query) const {
    return name.size() == query.size() && matches(query);
  }
};

std::string find_device_by_name(std::string_view query,
                                const std::vector<DeviceCandidate>& candidates) {
  std::string first_match;
  for (const auto& candidate : candidates) {
    if (!candidate.matches(query)) {
      continue;
    }
    if (candidate.matches_exactly(query)) {
      return candidate.path;
    }
    if (first_match.empty()) {
      first_match = candidate.path;
    }
  }
  return first_match;
}
}  // namespace

std::string hid_driver_of(const std::string& device_path) {
  const std::string event = basename_of(realpath_str(device_path));
  return basename_of(realpath_str("/sys/class/input/" + event + "/device/device/driver"));
}

namespace {
bool has_kind_caps(libevdev* dev, DeviceKind kind) {
  if (kind == DeviceKind::kGamepad) {
    return libevdev_has_event_code(dev, EV_ABS, ABS_X) &&
           libevdev_has_event_code(dev, EV_ABS, ABS_Y) &&
           libevdev_has_event_code(dev, EV_KEY, BTN_GAMEPAD);
  }
  return libevdev_has_event_code(dev, EV_KEY, KEY_W) &&
         libevdev_has_event_code(dev, EV_KEY, KEY_SPACE);
}
std::string probe_device_name(const std::string& path, DeviceKind kind) {
  const int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK);
  if (fd < 0) {
    return {};
  }
  libevdev* dev = nullptr;
  const char* name = libevdev_new_from_fd(fd, &dev) == 0 ? libevdev_get_name(dev) : nullptr;
  std::string found = name != nullptr && has_kind_caps(dev, kind) ? name : "";
  libevdev_free(dev);  // ignores nullptr
  ::close(fd);
  return found;
}
std::vector<DeviceCandidate> scan_devices(const std::string& event_dir, DeviceKind kind) {
  DIR* dir = opendir(event_dir.c_str());
  if (!dir) {
    return {};
  }
  std::vector<DeviceCandidate> candidates;
  for (dirent* entry = readdir(dir); entry != nullptr; entry = readdir(dir)) {
    if (std::string_view(entry->d_name).rfind("event", 0) != 0) {
      continue;
    }
    const std::string path = event_dir + "/" + entry->d_name;
    std::string name = probe_device_name(path, kind);
    if (!name.empty()) {
      candidates.emplace_back(path, std::move(name));
    }
  }
  closedir(dir);
  std::ranges::sort(candidates, [](const auto& a, const auto& b) {
    return std::pair(a.path.size(), std::string_view(a.path)) <  // shorter path first, so
           std::pair(b.path.size(), std::string_view(b.path));   // event9 sorts before event10
  });
  return candidates;
}
}  // namespace

std::string resolve_device_path(const std::string& device_or_name, DeviceKind kind,
                                const std::string& event_dir) {
  if (device_or_name.rfind("/dev/", 0) == 0) {
    return device_or_name;
  }
  return find_device_by_name(device_or_name, scan_devices(event_dir, kind));
}

bool is_ps5_dualsense(int vendor, int product) { return vendor == 0x054c && product == 0x0ce6; }

}  // namespace isaac::os::ops
