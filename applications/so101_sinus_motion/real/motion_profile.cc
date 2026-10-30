/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "applications/so101_sinus_motion/real/motion_profile.h"

#include <CLI/CLI.hpp>
#include <chrono>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>

#include <holoscan/config/io.hpp>

#include "common/so101_driver/so101_driver_config.h"

namespace {

/// Initialized before signal registration; the handler only assigns a sig_atomic_t.
volatile std::sig_atomic_t& interrupted() {
  static volatile std::sig_atomic_t flag = 0;
  return flag;
}

void interrupt(int) { interrupted() = 1; }

struct ProfileClock {
  auto now() const { return std::chrono::steady_clock::now(); }
  void sleep_until(std::chrono::steady_clock::time_point time) const {
    std::this_thread::sleep_until(time);
  }
};

/// Runs the diagnostic and preserves the position hold when measurement fails.
int run_profile(const std::filesystem::path& config_path,
                const std::filesystem::path& output_path) {
  const auto document = holoscan::config_io::load(config_path, holoscan::config_io::Format::kYaml);
  const auto options = isaac::os::decode_so101_driver_options(document);
  if (std::filesystem::exists(output_path)) {
    throw isaac::os::MotionProfileError(
        "The feedback file already exists. Choose a new output path.");
  }
  std::ofstream output(output_path);
  if (!output) {
    throw isaac::os::MotionProfileError(
        "Unable to create the feedback file. Check its parent directory.");
  }
  interrupted() = 0;
  std::signal(SIGINT, interrupt);
  std::signal(SIGTERM, interrupt);
  isaac::so101::FeetechSerialDriver driver;
  std::cerr << "The arm will home, then move shoulder pan 0.20 radians and back three times. "
               "Each target is held for three seconds.\n";
  try {
    driver.initialize(options, [](isaac::so101::DriverLogLevel, std::string_view message) {
      std::cerr << message << '\n';
    });
  } catch (...) {
    output.close();
    std::error_code cleanup_error;
    std::filesystem::remove(output_path, cleanup_error);
    if (cleanup_error) {
      std::cerr << "Unable to remove the empty feedback file " << output_path << ": "
                << cleanup_error.message() << ". Remove it before retrying.\n";
    }
    throw;
  }
  try {
    ProfileClock timing;
    isaac::os::measure_motion_profile(driver, output, timing, [] { return interrupted() != 0; });
    output.flush();
    if (!output) {
      throw isaac::os::MotionProfileError("Unable to finish saving the motion-test feedback.");
    }
  } catch (...) {
    if (!isaac::os::hold_after_motion_profile(driver)) {
      std::cerr
          << "Unable to confirm the position hold after three attempts. The arm may still move. "
             "Keep clear and switch off follower power before handling the arm. "
             "The unpowered arm can fall.\n";
    }
    throw;
  }
  if (!isaac::os::hold_after_motion_profile(driver)) {
    throw isaac::os::MotionProfileError(
        "Unable to confirm the final position hold after three attempts. The arm may still move. "
        "Keep clear and switch off follower power before handling the arm. "
        "The unpowered arm can fall.");
  }
  if (interrupted()) {
    std::cerr << "The motion test was interrupted. The arm is holding its current position.\n";
    return 130;
  }
  std::cerr << "The motion test completed. The arm is holding its current position.\n";
  return 0;
}

}  // namespace

/// Separate bounded hardware diagnostic: initialization still uses the standard homing path,
/// then graph-command speed and acceleration apply to six isolated target writes.
int main(int argc, char** argv) {
  std::filesystem::path config_path;
  std::filesystem::path output_path;
  CLI::App app{"Measure SO-101 servo speed and acceleration with isolated shoulder-pan targets"};
  app.add_option("--config-path", config_path, "Real-robot YAML configuration")->required();
  app.add_option("--output", output_path, "New CSV file for measured joint feedback")->required();
  try {
    app.parse(argc, argv);
    return run_profile(config_path, output_path);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
