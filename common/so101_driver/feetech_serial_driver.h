/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/so101_driver/so101_types.h"

namespace isaac::so101 {

/**
 * Log levels emitted by the SO-101 serial driver.
 */
enum class DriverLogLevel { kInfo, kWarning, kError };

/**
 * Runtime configuration for a Feetech SO-101 arm.
 */
struct FeetechDriverOptions {
  std::string serial_port;
  std::string calibration_file;
  std::vector<double> home_position;
  int run_speed = 0;
  int run_acceleration = 0;
  int position_p_gain = 0;
  int position_i_gain = 0;
  int position_d_gain = 0;
};

/**
 * Callback used to report driver activity without depending on an SDK logger.
 */
using DriverLogCallback = std::function<void(DriverLogLevel, std::string_view)>;

/**
 * Fatal error raised while preparing, executing, or verifying homing.
 */
class FeetechHomingError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * Serial-bus operations used by the SO-101 driver.
 */
class FeetechBus {
 public:
  virtual ~FeetechBus() = default;

  virtual bool begin(int baud_rate, const char* serial_port) = 0;
  virtual void end() = 0;
  virtual int write_byte(std::uint8_t id, std::uint8_t address, std::uint8_t value) = 0;
  virtual int write_word(std::uint8_t id, std::uint8_t address, std::uint16_t value) = 0;
  virtual int read_byte(std::uint8_t id, std::uint8_t address) = 0;
  virtual int read_word(std::uint8_t id, std::uint8_t address) = 0;
  virtual void sync_write_positions(std::uint8_t* ids, std::uint8_t count, std::int16_t* positions,
                                    std::uint16_t* speeds, std::uint8_t* accelerations) = 0;
  virtual void sync_read_begin(std::uint8_t count, std::uint8_t length, std::uint32_t timeout) = 0;
  virtual void sync_read_end() = 0;
  virtual int sync_read_packet_tx(std::uint8_t* ids, std::uint8_t count, std::uint8_t address,
                                  std::uint8_t length) = 0;
  virtual int sync_read_packet_rx(std::uint8_t id, std::uint8_t* data) = 0;
  virtual std::uint8_t last_status() = 0;
  virtual int sync_read_rx_word(std::uint8_t negative_bit) = 0;
};

/**
 * Direct Feetech STS-3215 serial driver for an SO-101 follower arm.
 *
 * Joint positions and velocities use radians and radians per second. This
 * class has no Holoscan dependency.
 */
class FeetechSerialDriver {
 public:
  FeetechSerialDriver();
  explicit FeetechSerialDriver(std::unique_ptr<FeetechBus> bus);
  ~FeetechSerialDriver();

  FeetechSerialDriver(const FeetechSerialDriver&) = delete;
  FeetechSerialDriver& operator=(const FeetechSerialDriver&) = delete;

  /**
   * Configure and home the arm; preserve the original startup error after verified holding
   * or best-effort torque disable if holding cannot be established.
   */
  void initialize(const FeetechDriverOptions& options, DriverLogCallback log_callback = {});

  bool send_action(const std::vector<float>& action);
  bool hold_present_position() noexcept;
  bool read_present_positions(std::vector<float>& positions);
  bool read_present_positions_and_velocities(std::vector<float>& positions,
                                             std::vector<float>& velocities);
  std::vector<std::string> get_ordered_joint_names() const;
  int num_joints() const;

 private:
  struct HomingResult {
    std::vector<float> final_position;
    std::vector<float> residuals;
    float largest_error = 0.0F;
    int largest_error_joint = 0;
    bool final_position_available = false;
    bool converged = false;
  };

  bool read_present_state(std::vector<float>& positions, std::vector<float>* velocities);
  /**
   * Read calibrated state and optional signed raw encoders into joint-indexed buffers.
   * The caller holds motor_mem_mutex_ and sizes each buffer to calib_.size().
   * Return false when any servo feedback cannot be read reliably.
   */
  bool read_present_state_locked(std::vector<float>& positions, std::vector<float>* velocities,
                                 std::vector<std::int16_t>* encoder_positions = nullptr);
  void apply_calibration();
  void apply_runtime_integral_gain();
  bool apply_shutdown_integral_gain_locked();
  void configure_gripper_protection();
  void configure_pid();
  void connect(std::string_view port);
  void disconnect();
  void disable_torque();
  void disable_torque_best_effort() noexcept;
  /**
   * Hold verified raw positions and clear integral gain after failed startup.
   * Return false on any failure so initialization can disable torque before rethrowing.
   */
  bool hold_after_startup_failure() noexcept;
  void enable_torque();
  void home();
  std::vector<float> make_reachable_home_target(const std::vector<float>& requested) const;
  bool read_homing_start_position(std::vector<float>& current);
  void send_homing_trajectory(const std::vector<float>& current, const std::vector<float>& target);
  HomingResult wait_for_homing_convergence(const std::vector<float>& target);
  void validate_homing_result(const std::vector<float>& target, const HomingResult& result) const;
  void load_calibration();
  bool send_action(const std::vector<float>& action, std::uint16_t speed,
                   std::uint8_t acceleration);
  bool send_action_locked(const std::vector<float>& action, std::uint16_t speed,
                          std::uint8_t acceleration, bool allow_during_shutdown);
  void validate_calibration();
  void log(DriverLogLevel level, std::string message) const;

  std::unique_ptr<FeetechBus> bus_;
  mutable std::mutex motor_mem_mutex_;
  std::unordered_map<std::string, MotorCalibration> calib_;
  std::vector<std::uint8_t> sync_ids_;
  FeetechDriverOptions options_;
  DriverLogCallback log_callback_;
  bool connected_ = false;
  bool sync_ready_ = false;
  bool read_failure_reported_ = false;
  bool shutdown_started_ = false;
  bool shutdown_hold_complete_ = false;
  std::uint16_t run_speed_cached_ = 0;
  std::uint8_t run_acceleration_cached_ = 0;
  std::uint8_t position_p_gain_cached_ = 0;
  std::uint8_t position_i_gain_cached_ = 0;
  std::uint8_t position_d_gain_cached_ = 0;
  std::uint8_t homing_integral_gain_cached_ = 0;
};

}  // namespace isaac::so101
