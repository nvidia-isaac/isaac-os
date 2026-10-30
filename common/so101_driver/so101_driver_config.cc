/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/so101_driver/so101_driver_config.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/config/codec.hpp>

namespace isaac::os {
namespace {

void validate(const so101::FeetechDriverOptions& options) {
  if (options.serial_port.empty()) {
    throw std::runtime_error("The robot serial port must not be empty.");
  }
  if (options.calibration_file.empty() ||
      options.calibration_file.find("PLACEHOLDER") != std::string::npos) {
    throw std::runtime_error("The robot requires a calibration file instead of a placeholder.");
  }
  if (options.home_position.size() != 6U ||
      !std::all_of(options.home_position.begin(), options.home_position.end(),
                   [](double position) { return std::isfinite(position); })) {
    throw std::runtime_error("The robot home position must contain six finite values.");
  }
  if (options.run_speed < 0 || options.run_acceleration < 0 || options.position_p_gain < 0 ||
      options.position_i_gain < 0 || options.position_d_gain < 0) {
    throw std::runtime_error("The robot servo settings must not be negative.");
  }
}

}  // namespace

so101::FeetechDriverOptions decode_so101_driver_options(
    const holoscan::config::Document& document) {
  so101::FeetechDriverOptions options{
      .serial_port = document.get<std::string>("/real_robot/serial_port"),
      .calibration_file = document.get<std::string>("/real_robot/calibration_file"),
      .home_position = document.get<std::vector<double>>("/real_robot/home_position"),
      .run_speed = document.get<int>("/real_robot/run_speed"),
      .run_acceleration = document.get<int>("/real_robot/run_acceleration"),
      .position_p_gain = document.get<int>("/real_robot/position_p_gain"),
      .position_i_gain = document.get<int>("/real_robot/position_i_gain"),
      .position_d_gain = document.get<int>("/real_robot/position_d_gain"),
  };
  validate(options);
  return options;
}

}  // namespace isaac::os
