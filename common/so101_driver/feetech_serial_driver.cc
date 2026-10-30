/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/so101_driver/feetech_serial_driver.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>

#include "ftservo/SCServo.h"
#include "nlohmann/json.hpp"

#include "common/so101_driver/so101_unit_conversions.h"

namespace isaac::so101 {
namespace {
constexpr int kDefaultBaud = 1'000'000;
constexpr std::uint8_t kMaxAccelerationCeiling = 254;
constexpr int kMaxHomingOffset = (1 << 11) - 1;
constexpr std::uint16_t kHomingSpeed = 400;
constexpr std::uint8_t kHomingAcceleration = 50;
// A proportional-only position loop leaves a standing error on gravity-loaded joints, which the
// kHomeReachTolRad gate cannot accept. Homing therefore runs with integral action regardless of
// the configured run-time I gain.
constexpr std::uint8_t kHomingIntegralGain = 16;
constexpr float kHomeReachTolRad = 4.0f * UnitConversions::kRadiansPerEncoderCount;
constexpr int kHomingReadRetries = 3;
constexpr auto kHomingReadRetryDelay = std::chrono::milliseconds(200);
constexpr auto kHomingBusSettleDelay = std::chrono::milliseconds(50);
constexpr int kHomingSteps = 300;
constexpr auto kHomingStepInterval = std::chrono::milliseconds(10);
constexpr auto kHomingConvergenceTimeout = std::chrono::seconds(2);
constexpr auto kHomingConvergencePollInterval = std::chrono::milliseconds(50);
constexpr std::uint16_t kGripperTorqueLimit = 500;
constexpr std::uint16_t kGripperProtectionCurrent = 250;
constexpr std::uint8_t kGripperOverloadTorque = 25;
constexpr std::uint8_t kPositionOperatingMode = 0;
constexpr int kRegisterWriteAttempts = 3;
constexpr auto kRegisterWriteRetryDelay = std::chrono::milliseconds(10);

enum class FeetechAddr : std::uint8_t {
  kReturnDelayTime = 7,
  kMinPositionLimitL = 9,
  kMaxPositionLimitL = 11,
  kPCoefficient = 21,
  kDCoefficient = 22,
  kICoefficient = 23,
  kProtectionCurrentL = 28,
  kHomingOffsetL = 31,
  kOperatingMode = 33,
  kOverloadTorque = 36,
  kTorqueEnable = 40,
  kAcceleration = 41,
  kGoalPositionL = 42,
  kTorqueLimitL = 48,
  kLock = 55,
  kPresentPositionL = 56,
  kMaximumAcceleration = 85,
};

constexpr std::uint8_t register_address(FeetechAddr address) {
  return static_cast<std::uint8_t>(address);
}

class FtServoBus final : public FeetechBus {
 public:
  bool begin(int baud_rate, const char* serial_port) override {
    return sms_.begin(baud_rate, serial_port);
  }
  void end() override { sms_.end(); }
  int write_byte(std::uint8_t id, std::uint8_t address, std::uint8_t value) override {
    return sms_.writeByte(id, address, value);
  }
  int write_word(std::uint8_t id, std::uint8_t address, std::uint16_t value) override {
    return sms_.writeWord(id, address, value);
  }
  int read_byte(std::uint8_t id, std::uint8_t address) override {
    return sms_.readByte(id, address);
  }
  int read_word(std::uint8_t id, std::uint8_t address) override {
    return sms_.readWord(id, address);
  }
  void sync_write_positions(std::uint8_t* ids, std::uint8_t count, std::int16_t* positions,
                            std::uint16_t* speeds, std::uint8_t* accelerations) override {
    sms_.SyncWritePosEx(ids, count, positions, speeds, accelerations);
  }
  void sync_read_begin(std::uint8_t count, std::uint8_t length, std::uint32_t timeout) override {
    sms_.syncReadBegin(count, length, timeout);
  }
  void sync_read_end() override { sms_.syncReadEnd(); }
  int sync_read_packet_tx(std::uint8_t* ids, std::uint8_t count, std::uint8_t address,
                          std::uint8_t length) override {
    return sms_.syncReadPacketTx(ids, count, address, length);
  }
  int sync_read_packet_rx(std::uint8_t id, std::uint8_t* data) override {
    return sms_.syncReadPacketRx(id, data);
  }
  std::uint8_t last_status() override { return sms_.getState(); }
  int sync_read_rx_word(std::uint8_t negative_bit) override {
    return sms_.syncReadRxPacketToWrod(negative_bit);
  }

 private:
  SMS_STS sms_;
};

template <typename... Args>
std::string make_message(Args&&... args) {
  std::ostringstream stream;
  (stream << ... << std::forward<Args>(args));
  return stream.str();
}

std::string join(const std::vector<float>& values) {
  std::ostringstream stream;
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) stream << ", ";
    stream << values[index];
  }
  return stream.str();
}

std::string hexadecimal_byte(std::byte value) {
  constexpr char kHexDigits[] = "0123456789ABCDEF";
  constexpr std::byte kLowNibble{0x0FU};
  std::string result{"0x00"};
  result[2] = kHexDigits[std::to_integer<std::size_t>((value >> 4U) & kLowNibble)];
  result[3] = kHexDigits[std::to_integer<std::size_t>(value & kLowNibble)];
  return result;
}

std::string describe_servo_status(std::byte status) {
  constexpr std::byte kVoltage{0x01U};
  constexpr std::byte kAngle{0x02U};
  constexpr std::byte kOverheat{0x04U};
  constexpr std::byte kOvercurrent{0x08U};
  constexpr std::byte kOverload{0x20U};
  constexpr std::byte kKnownStatusMask = kVoltage | kAngle | kOverheat | kOvercurrent | kOverload;

  std::vector<std::string> conditions;
  if ((status & kVoltage) != std::byte{}) conditions.emplace_back("voltage fault");
  if ((status & kAngle) != std::byte{}) conditions.emplace_back("angle fault");
  if ((status & kOverheat) != std::byte{}) conditions.emplace_back("overheat");
  if ((status & kOvercurrent) != std::byte{}) conditions.emplace_back("overcurrent");
  if ((status & kOverload) != std::byte{}) conditions.emplace_back("overload");
  if (const std::byte unknown = status & ~kKnownStatusMask; unknown != std::byte{}) {
    conditions.emplace_back(make_message("unknown bits ", hexadecimal_byte(unknown)));
  }

  std::ostringstream stream;
  for (std::size_t index = 0; index < conditions.size(); ++index) {
    if (index != 0U) stream << ", ";
    stream << conditions[index];
  }
  return stream.str();
}

bool is_gripper(std::string_view joint_name) { return joint_name == "gripper"; }

float joint_encoder_to_rad(std::string_view joint_name, std::int16_t value,
                           const MotorCalibration& calibration) {
  if (is_gripper(joint_name)) {
    return UnitConversions::gripper_encoder_to_rad(value, calibration);
  }
  return UnitConversions::encoder_to_rad(value, calibration);
}

std::int16_t joint_rad_to_encoder(std::string_view joint_name, float value,
                                  const MotorCalibration& calibration) {
  if (is_gripper(joint_name)) {
    return UnitConversions::gripper_rad_to_encoder(value, calibration);
  }
  return UnitConversions::rad_to_encoder(value, calibration);
}

std::array<float, 2> joint_reachable_rad_range(std::string_view joint_name,
                                               const MotorCalibration& calibration) {
  if (is_gripper(joint_name)) {
    return UnitConversions::gripper_reachable_rad_range(calibration);
  }
  return UnitConversions::reachable_rad_range(calibration);
}

void require_write_ack(int result, std::uint8_t id, std::string_view operation) {
  if (result == 0) {
    throw std::runtime_error(make_message("Unable to ", operation, " for motor ID ",
                                          static_cast<int>(id),
                                          ": the controller did not acknowledge the write."));
  }
}

int write_byte_with_retry(FeetechBus& bus, std::uint8_t id, std::uint8_t address,
                          std::uint8_t value) {
  for (int attempt = 1; attempt <= kRegisterWriteAttempts; ++attempt) {
    const int result = bus.write_byte(id, address, value);
    if (result != 0 || attempt == kRegisterWriteAttempts) return result;
    std::this_thread::sleep_for(kRegisterWriteRetryDelay);
  }
  return 0;
}
}  // namespace

FeetechSerialDriver::FeetechSerialDriver() : FeetechSerialDriver(std::make_unique<FtServoBus>()) {}

FeetechSerialDriver::FeetechSerialDriver(std::unique_ptr<FeetechBus> bus) : bus_(std::move(bus)) {
  if (!bus_) {
    throw std::invalid_argument("The SO-101 serial driver requires a bus implementation.");
  }
}

FeetechSerialDriver::~FeetechSerialDriver() { disconnect(); }

/**
 * End any synchronous-read session and close the bus without changing servo torque.
 */
void FeetechSerialDriver::disconnect() {
  std::lock_guard<std::mutex> lock(motor_mem_mutex_);
  if (!connected_) return;
  if (sync_ready_) {
    bus_->sync_read_end();
    sync_ready_ = false;
  }
  bus_->end();
  connected_ = false;
}

/**
 * Configure and home the arm, recovering startup failures with a verified hold or torque disable.
 */
void FeetechSerialDriver::initialize(const FeetechDriverOptions& options,
                                     DriverLogCallback log_callback) {
  options_ = options;
  log_callback_ = std::move(log_callback);
  run_speed_cached_ = static_cast<std::uint16_t>(std::clamp(options_.run_speed, 0, 0xFFFF));
  run_acceleration_cached_ = static_cast<std::uint8_t>(
      std::clamp(options_.run_acceleration, 0, static_cast<int>(kMaxAccelerationCeiling)));
  constexpr int kMaxGain = 254;
  position_p_gain_cached_ =
      static_cast<std::uint8_t>(std::clamp(options_.position_p_gain, 0, kMaxGain));
  position_i_gain_cached_ =
      static_cast<std::uint8_t>(std::clamp(options_.position_i_gain, 0, kMaxGain));
  position_d_gain_cached_ =
      static_cast<std::uint8_t>(std::clamp(options_.position_d_gain, 0, kMaxGain));
  homing_integral_gain_cached_ = std::max(position_i_gain_cached_, kHomingIntegralGain);
  if (connected_) return;
  read_failure_reported_ = false;
  shutdown_started_ = false;
  shutdown_hold_complete_ = false;

  load_calibration();
  connect(options_.serial_port);
  bool torque_enabled = false;
  try {
    disable_torque();
    apply_calibration();
    validate_calibration();
    configure_pid();
    configure_gripper_protection();
    enable_torque();
    torque_enabled = true;
    home();
    apply_runtime_integral_gain();
  } catch (...) {
    if (!torque_enabled || !hold_after_startup_failure()) {
      disable_torque_best_effort();
    }
    disconnect();
    throw;
  }
  log(DriverLogLevel::kInfo,
      make_message("SO-101 arm initialized on port: ", options_.serial_port));
}

void FeetechSerialDriver::connect(std::string_view port) {
  std::lock_guard<std::mutex> lock(motor_mem_mutex_);
  const std::string port_str(port);
  if (!bus_->begin(kDefaultBaud, port_str.c_str())) {
    throw std::runtime_error("Unable to connect to the SO-101 arm on port " + port_str + ".");
  }
  connected_ = true;
}

void FeetechSerialDriver::load_calibration() {
  std::ifstream file(options_.calibration_file);
  if (!file)
    throw std::runtime_error("Unable to open calibration file: " + options_.calibration_file);
  const nlohmann::json json = nlohmann::json::parse(file);
  if (!json.is_object())
    throw std::runtime_error("The calibration file does not contain a JSON object.");

  calib_.clear();
  std::unordered_set<std::uint8_t> ids;
  for (const auto& [motor_name, motor_data] : json.items()) {
    if (!motor_data.is_object())
      throw std::runtime_error("The calibration file contains an invalid motor entry.");
    MotorCalibration calibration;
    calibration.id = motor_data.at("id").get<std::uint8_t>();
    calibration.drive_mode = motor_data.at("drive_mode").get<std::uint8_t>();
    calibration.homing_offset = motor_data.at("homing_offset").get<int>();
    calibration.range_min = motor_data.at("range_min").get<std::uint16_t>();
    calibration.range_max = motor_data.at("range_max").get<std::uint16_t>();
    if (calibration.range_max <= calibration.range_min) {
      throw std::runtime_error(
          "Motor '" + motor_name +
          "' has an invalid calibration range: the maximum must exceed the minimum.");
    }
    if (calibration.drive_mode > 1U) {
      throw std::runtime_error("Motor '" + motor_name +
                               "' has an invalid direction mode; expected 0 or 1.");
    }
    if (calibration.homing_offset < -kMaxHomingOffset ||
        calibration.homing_offset > kMaxHomingOffset) {
      throw std::runtime_error("Motor '" + motor_name +
                               "' has an invalid homing offset; expected a value from -2047 to "
                               "2047.");
    }
    const auto [_, inserted] = ids.insert(calibration.id);
    if (!inserted) {
      throw std::runtime_error("The calibration file assigns motor ID " +
                               std::to_string(calibration.id) + " more than once.");
    }
    calib_[motor_name] = calibration;
  }
  if (calib_.size() != kSo101JointNames.size()) {
    throw std::runtime_error("The SO-101 calibration must contain exactly six motors.");
  }
  sync_ids_.clear();
  for (std::uint8_t id = 1; id <= static_cast<std::uint8_t>(ids.size()); ++id) {
    if (!ids.contains(id)) {
      throw std::runtime_error("The calibration file has no motor entry with ID " +
                               std::to_string(id) + ".");
    }
    sync_ids_.push_back(id);
  }
  const std::vector<std::string> ordered_names = get_ordered_joint_names();
  for (std::size_t index = 0; index < kSo101JointNames.size(); ++index) {
    if (ordered_names[index] != kSo101JointNames[index]) {
      throw std::runtime_error(make_message("Motor ID ", index + 1U, " is calibrated as '",
                                            ordered_names[index], "'; expected '",
                                            kSo101JointNames[index], "'."));
    }
  }
}

void FeetechSerialDriver::apply_calibration() {
  std::lock_guard<std::mutex> lock(motor_mem_mutex_);
  for (const auto& [_, calibration] : calib_) {
    const std::uint8_t id = calibration.id;
    require_write_ack(
        bus_->write_byte(id, register_address(FeetechAddr::kOperatingMode), kPositionOperatingMode),
        id, "select position-control mode");
    const std::uint16_t offset =
        calibration.homing_offset < 0
            ? static_cast<std::uint16_t>(-calibration.homing_offset) | (1U << 11U)
            : static_cast<std::uint16_t>(calibration.homing_offset);
    require_write_ack(bus_->write_word(id, register_address(FeetechAddr::kHomingOffsetL), offset),
                      id, "write the homing offset");
    require_write_ack(bus_->write_word(id, register_address(FeetechAddr::kMinPositionLimitL),
                                       calibration.range_min),
                      id, "write the minimum position limit");
    require_write_ack(bus_->write_word(id, register_address(FeetechAddr::kMaxPositionLimitL),
                                       calibration.range_max),
                      id, "write the maximum position limit");
    require_write_ack(bus_->write_byte(id, register_address(FeetechAddr::kReturnDelayTime), 0), id,
                      "disable the response delay");
    require_write_ack(bus_->write_byte(id, register_address(FeetechAddr::kMaximumAcceleration),
                                       kMaxAccelerationCeiling),
                      id, "write the maximum acceleration");
    require_write_ack(bus_->write_byte(id, register_address(FeetechAddr::kAcceleration),
                                       run_acceleration_cached_),
                      id, "write the run acceleration");
  }
}

void FeetechSerialDriver::validate_calibration() {
  std::lock_guard<std::mutex> lock(motor_mem_mutex_);
  for (const auto& [name, calibration] : calib_) {
    const std::uint8_t id = calibration.id;
    if (calibration.range_min !=
        bus_->read_word(id, register_address(FeetechAddr::kMinPositionLimitL))) {
      throw std::runtime_error("The configured minimum range for motor '" + name +
                               "' does not match the controller.");
    }
    if (calibration.range_max !=
        bus_->read_word(id, register_address(FeetechAddr::kMaxPositionLimitL))) {
      throw std::runtime_error("The configured maximum range for motor '" + name +
                               "' does not match the controller.");
    }
    int homing_offset =
        static_cast<int>(bus_->read_word(id, register_address(FeetechAddr::kHomingOffsetL)));
    if (homing_offset & (1 << 11)) homing_offset = -(homing_offset & ~(1 << 11));
    if (calibration.homing_offset != homing_offset) {
      throw std::runtime_error("The configured homing offset for motor '" + name +
                               "' does not match the controller.");
    }
    if (kPositionOperatingMode !=
        bus_->read_byte(id, register_address(FeetechAddr::kOperatingMode))) {
      throw std::runtime_error("Motor '" + name + "' is not in position-control mode.");
    }
  }
}

void FeetechSerialDriver::enable_torque() {
  std::lock_guard<std::mutex> lock(motor_mem_mutex_);
  for (const auto& [_, calibration] : calib_) {
    require_write_ack(write_byte_with_retry(*bus_, calibration.id,
                                            register_address(FeetechAddr::kTorqueEnable), 1),
                      calibration.id, "enable torque");
    require_write_ack(
        write_byte_with_retry(*bus_, calibration.id, register_address(FeetechAddr::kLock), 1),
        calibration.id, "lock the servo configuration");
  }
}

void FeetechSerialDriver::disable_torque() {
  std::lock_guard<std::mutex> lock(motor_mem_mutex_);
  for (const auto& [_, calibration] : calib_) {
    require_write_ack(write_byte_with_retry(*bus_, calibration.id,
                                            register_address(FeetechAddr::kTorqueEnable), 0),
                      calibration.id, "disable torque");
    require_write_ack(
        write_byte_with_retry(*bus_, calibration.id, register_address(FeetechAddr::kLock), 0),
        calibration.id, "unlock the servo configuration");
  }
}

void FeetechSerialDriver::disable_torque_best_effort() noexcept {
  std::uint8_t torque_failure_mask = 0U;
  std::uint8_t unlock_failure_mask = 0U;
  try {
    std::lock_guard<std::mutex> lock(motor_mem_mutex_);
    if (!connected_) return;
    for (const std::uint8_t id : sync_ids_) {
      if (write_byte_with_retry(*bus_, id, register_address(FeetechAddr::kTorqueEnable), 0) == 0) {
        torque_failure_mask |= static_cast<std::uint8_t>(1U << (id - 1U));
      }
      if (write_byte_with_retry(*bus_, id, register_address(FeetechAddr::kLock), 0) == 0) {
        unlock_failure_mask |= static_cast<std::uint8_t>(1U << (id - 1U));
      }
    }
  } catch (...) {
    try {
      log(DriverLogLevel::kError,
          "Unable to finish disabling torque after initialization failed. Switch off follower "
          "power before handling the arm.");
    } catch (...) {
    }
    return;
  }

  for (const std::uint8_t id : sync_ids_) {
    const std::uint8_t id_mask = static_cast<std::uint8_t>(1U << (id - 1U));
    try {
      if ((torque_failure_mask & id_mask) != 0U) {
        log(DriverLogLevel::kError,
            make_message("Unable to confirm that torque was disabled for motor ID ",
                         static_cast<int>(id),
                         " after initialization failed. Switch off follower power before "
                         "handling the arm."));
      } else if ((unlock_failure_mask & id_mask) != 0U) {
        log(DriverLogLevel::kWarning,
            make_message("Torque was disabled for motor ID ", static_cast<int>(id),
                         ", but its configuration could not be unlocked."));
      }
    } catch (...) {
    }
  }
}

void FeetechSerialDriver::configure_pid() {
  {
    std::lock_guard<std::mutex> lock(motor_mem_mutex_);
    for (const auto& [_, calibration] : calib_) {
      require_write_ack(
          bus_->write_byte(calibration.id, register_address(FeetechAddr::kPCoefficient),
                           position_p_gain_cached_),
          calibration.id, "write the position-loop P gain");
      require_write_ack(
          bus_->write_byte(calibration.id, register_address(FeetechAddr::kICoefficient),
                           homing_integral_gain_cached_),
          calibration.id, "write the position-loop I gain");
      require_write_ack(
          bus_->write_byte(calibration.id, register_address(FeetechAddr::kDCoefficient),
                           position_d_gain_cached_),
          calibration.id, "write the position-loop D gain");
    }
  }
  log(DriverLogLevel::kInfo, make_message("Configured runtime position-loop gains: P=",
                                          static_cast<int>(position_p_gain_cached_),
                                          ", I=", static_cast<int>(position_i_gain_cached_),
                                          ", D=", static_cast<int>(position_d_gain_cached_), "."));
  if (homing_integral_gain_cached_ != position_i_gain_cached_) {
    log(DriverLogLevel::kInfo,
        make_message("Homing temporarily uses P=", static_cast<int>(position_p_gain_cached_),
                     ", I=", static_cast<int>(homing_integral_gain_cached_), ", D=",
                     static_cast<int>(position_d_gain_cached_), " to reach the homing tolerance."));
  } else {
    log(DriverLogLevel::kInfo, "Homing uses the configured runtime position-loop gains.");
  }
}

void FeetechSerialDriver::apply_runtime_integral_gain() {
  if (homing_integral_gain_cached_ != position_i_gain_cached_) {
    using enum FeetechAddr;
    std::scoped_lock lock(motor_mem_mutex_);
    for (const auto& [_, calibration] : calib_) {
      // The I coefficient lives in the locked configuration area, so enable_torque()'s lock has
      // to be lifted for the write and restored afterwards.
      require_write_ack(write_byte_with_retry(*bus_, calibration.id, register_address(kLock), 0),
                        calibration.id, "unlock the servo configuration");
      require_write_ack(
          write_byte_with_retry(*bus_, calibration.id, register_address(kICoefficient),
                                position_i_gain_cached_),
          calibration.id, "write the position-loop I gain");
      require_write_ack(write_byte_with_retry(*bus_, calibration.id, register_address(kLock), 1),
                        calibration.id, "lock the servo configuration");
    }
  }
  log(DriverLogLevel::kInfo,
      make_message("Homing complete; applied runtime position-loop gains: P=",
                   static_cast<int>(position_p_gain_cached_),
                   ", I=", static_cast<int>(position_i_gain_cached_),
                   ", D=", static_cast<int>(position_d_gain_cached_), "."));
}

void FeetechSerialDriver::configure_gripper_protection() {
  std::lock_guard<std::mutex> lock(motor_mem_mutex_);
  const std::uint8_t id = calib_.at("gripper").id;
  require_write_ack(
      bus_->write_word(id, register_address(FeetechAddr::kTorqueLimitL), kGripperTorqueLimit), id,
      "write the gripper torque limit");
  require_write_ack(bus_->write_word(id, register_address(FeetechAddr::kProtectionCurrentL),
                                     kGripperProtectionCurrent),
                    id, "write the gripper protection current");
  require_write_ack(
      bus_->write_byte(id, register_address(FeetechAddr::kOverloadTorque), kGripperOverloadTorque),
      id, "write the gripper overload threshold");
}

bool FeetechSerialDriver::send_action(const std::vector<float>& action) {
  return send_action(action, run_speed_cached_, run_acceleration_cached_);
}

bool FeetechSerialDriver::send_action(const std::vector<float>& action, std::uint16_t speed,
                                      std::uint8_t acceleration) {
  std::scoped_lock lock(motor_mem_mutex_);
  return send_action_locked(action, speed, acceleration, false);
}

bool FeetechSerialDriver::send_action_locked(const std::vector<float>& action, std::uint16_t speed,
                                             std::uint8_t acceleration,
                                             bool allow_during_shutdown) {
  if (action.size() != calib_.size()) {
    log(DriverLogLevel::kError,
        make_message("Unable to send the arm command: received ", action.size(), " joints but ",
                     calib_.size(), " are calibrated."));
    return false;
  }
  if (shutdown_started_ && !allow_during_shutdown) return false;
  std::vector<std::int16_t> positions(calib_.size());
  std::vector<std::uint16_t> speeds(calib_.size(), speed);
  std::vector<std::uint8_t> accelerations(calib_.size(), acceleration);
  for (const auto& [name, calibration] : calib_) {
    positions[calibration.id - 1] =
        joint_rad_to_encoder(name, action[calibration.id - 1], calibration);
  }
  bus_->sync_write_positions(sync_ids_.data(), static_cast<std::uint8_t>(calib_.size()),
                             positions.data(), speeds.data(), accelerations.data());
  return true;
}

/**
 * Clear integral gain and relock every servo while holding the motor mutex.
 * Return false if any step fails; still attempt the remaining servos.
 */
bool FeetechSerialDriver::apply_shutdown_integral_gain_locked() {
  // Reach needs integral gain to overcome gravity, but retaining it after the final target
  // can produce delayed corrective ticks. Keep torque and P/D holding active.
  const auto write = [this](std::uint8_t id, FeetechAddr address, std::uint8_t value,
                            std::string_view operation) {
    try {
      require_write_ack(write_byte_with_retry(*bus_, id, register_address(address), value), id,
                        operation);
      return true;
    } catch (const std::runtime_error& error) {
      log(DriverLogLevel::kError,
          make_message("Unable to finish preparing the shutdown hold: ", error.what(), "."));
      return false;
    }
  };
  using enum FeetechAddr;
  bool complete = true;
  for (const std::uint8_t id : sync_ids_) {
    const bool unlocked = write(id, kLock, 0, "unlock the servo configuration for holding");
    const bool gain_written =
        unlocked && write(id, kICoefficient, 0, "disable integral correction for holding");
    // Always attempt to relock, including after exceptions or a missing unlock acknowledgement.
    const bool locked = write(id, kLock, 1, "lock the servo configuration after holding");
    complete = unlocked && gain_written && locked && complete;
  }
  return complete;
}

/**
 * Verify raw-position holding and clear integral gain; return false if recovery cannot be
 * verified.
 */
bool FeetechSerialDriver::hold_after_startup_failure() noexcept {
  try {
    std::scoped_lock lock(motor_mem_mutex_);
    using enum FeetechAddr;
    shutdown_started_ = true;
    std::vector<float> current(calib_.size());
    std::vector<std::int16_t> encoder_positions(calib_.size());
    if (!read_present_state_locked(current, nullptr, &encoder_positions)) return false;
    for (const auto& [name, calibration] : calib_) {
      const auto id = static_cast<std::uint8_t>(calibration.id);
      // Preserve the raw measurement: radian conversion clamps to calibration limits.
      const int measured = encoder_positions[id - 1];
      const auto target =
          static_cast<std::uint16_t>(measured < 0 ? (-measured | (1U << 15U)) : measured);
      require_write_ack(bus_->write_word(id, register_address(kGoalPositionL), target), id,
                        "hold the measured position");
      if (bus_->read_word(id, register_address(kGoalPositionL)) != target ||
          bus_->read_byte(id, register_address(kTorqueEnable)) != 1) {
        log(DriverLogLevel::kError, "The arm hold could not be verified after startup failed.");
        return false;
      }
    }
    if (!apply_shutdown_integral_gain_locked()) return false;
    shutdown_hold_complete_ = true;
    log(DriverLogLevel::kWarning,
        "Arm startup failed. Torque remains enabled to hold the measured position. "
        "Support the arm before switching off servo power.");
    return true;
  } catch (const std::exception& error) {
    log(DriverLogLevel::kError,
        make_message("Unable to hold the arm after startup failed: ", error.what()));
    return false;
  } catch (...) {
    return false;
  }
}

/**
 * Latch the current joint positions and clear integral gain for shutdown holding.
 * Return false if reading, commanding, or gain configuration fails.
 */
bool FeetechSerialDriver::hold_present_position() noexcept {
  try {
    std::vector<float> current;
    {
      std::lock_guard<std::mutex> lock(motor_mem_mutex_);
      if (!connected_) {
        log(DriverLogLevel::kWarning, "Unable to hold the arm position: the arm is not connected.");
        return false;
      }
      shutdown_started_ = true;
      if (shutdown_hold_complete_) return true;
      current.resize(calib_.size());
      if (!read_present_state_locked(current, nullptr)) {
        log(DriverLogLevel::kError,
            "Unable to hold the arm position: the present positions could not be read.");
        return false;
      }
      if (!send_action_locked(current, run_speed_cached_, run_acceleration_cached_, true)) {
        return false;
      }
      // Retry only configuration preparation; never resend the final motion target.
      if (!apply_shutdown_integral_gain_locked() && !apply_shutdown_integral_gain_locked()) {
        log(DriverLogLevel::kError,
            "Shutdown holding gains could not be fully configured after retrying. "
            "Keep clear of the arm and disconnect power before handling it.");
        return false;
      }
      shutdown_hold_complete_ = true;
    }
    log(DriverLogLevel::kInfo,
        make_message("Holding the arm at its present position (radians): [", join(current), "]."));
    return true;
  } catch (const std::exception& error) {
    log(DriverLogLevel::kError,
        make_message("Unable to hold the arm position: ", error.what(), "."));
    return false;
  }
}

bool FeetechSerialDriver::read_present_positions(std::vector<float>& positions) {
  return read_present_state(positions, nullptr);
}

bool FeetechSerialDriver::read_present_positions_and_velocities(std::vector<float>& positions,
                                                                std::vector<float>& velocities) {
  if (velocities.size() != calib_.size()) {
    throw std::runtime_error(make_message("The velocity buffer has ", velocities.size(),
                                          " joints but ", calib_.size(), " are calibrated."));
  }
  return read_present_state(positions, &velocities);
}

/**
 * Read calibrated positions and optional velocities under the motor mutex.
 * The caller supplies one entry per calibrated joint in each requested buffer.
 * Throw if the position buffer has the wrong size; return false on unreliable feedback.
 */
bool FeetechSerialDriver::read_present_state(std::vector<float>& positions,
                                             std::vector<float>* velocities) {
  if (positions.size() != calib_.size()) {
    throw std::runtime_error(make_message("The position buffer has ", positions.size(),
                                          " joints but ", calib_.size(), " are calibrated."));
  }
  std::lock_guard<std::mutex> lock(motor_mem_mutex_);
  return read_present_state_locked(positions, velocities);
}

/**
 * Read joint-indexed state and optional raw encoders while the caller holds the motor mutex.
 */
bool FeetechSerialDriver::read_present_state_locked(std::vector<float>& positions,
                                                    std::vector<float>* velocities,
                                                    std::vector<std::int16_t>* encoder_positions) {
  if (!sync_ready_) {
    bus_->sync_read_begin(static_cast<std::uint8_t>(calib_.size()), 4, 5);
    sync_ready_ = true;
  }
  bus_->sync_read_packet_tx(sync_ids_.data(), static_cast<std::uint8_t>(sync_ids_.size()),
                            register_address(FeetechAddr::kPresentPositionL), 4);
  std::uint8_t received[4];
  for (const auto& [name, calibration] : calib_) {
    if (!bus_->sync_read_packet_rx(calibration.id, received)) {
      if (!read_failure_reported_) {
        log(DriverLogLevel::kError,
            make_message("Unable to read the present state for motor '", name, "'. Retrying."));
        read_failure_reported_ = true;
      }
      return false;
    }
    if (const std::byte status{bus_->last_status()}; status != std::byte{}) {
      log(DriverLogLevel::kError,
          make_message("Unable to use the present state for motor '", name, "' (ID ",
                       static_cast<int>(calibration.id), "): the controller reported status ",
                       hexadecimal_byte(status), " (", describe_servo_status(status),
                       "). Stop motion, switch off follower power, and check the servo voltage, "
                       "joint travel, load, temperature, and wiring before retrying."));
      return false;
    }
    const std::int16_t position = static_cast<std::int16_t>(bus_->sync_read_rx_word(15));
    const std::int16_t speed = static_cast<std::int16_t>(bus_->sync_read_rx_word(15));
    if (encoder_positions) (*encoder_positions)[calibration.id - 1] = position;
    positions[calibration.id - 1] = joint_encoder_to_rad(name, position, calibration);
    if (velocities) {
      (*velocities)[calibration.id - 1] =
          UnitConversions::encoder_speed_to_rad_per_s(speed, calibration);
    }
  }
  read_failure_reported_ = false;
  return true;
}

std::vector<float> FeetechSerialDriver::make_reachable_home_target(
    const std::vector<float>& requested) const {
  std::vector<float> target(requested);
  for (const auto& [name, calibration] : calib_) {
    const auto [minimum, maximum] = joint_reachable_rad_range(name, calibration);
    const int index = calibration.id - 1;
    target[index] = std::clamp(requested[index], minimum, maximum);
    if (std::fabs(target[index] - requested[index]) > kHomeReachTolRad) {
      log(DriverLogLevel::kWarning,
          make_message("The configured home position ", requested[index], " rad for '", name,
                       "' is outside its reachable range [", minimum, ", ", maximum,
                       "] rad; using ", target[index], " rad."));
    }
  }
  return target;
}

bool FeetechSerialDriver::read_homing_start_position(std::vector<float>& current) {
  for (int attempt = 1; attempt <= kHomingReadRetries; ++attempt) {
    if (read_present_positions(current)) return true;
    if (attempt < kHomingReadRetries) std::this_thread::sleep_for(kHomingReadRetryDelay);
  }
  return false;
}

void FeetechSerialDriver::send_homing_trajectory(const std::vector<float>& current,
                                                 const std::vector<float>& target) {
  const int num = num_joints();
  for (int step = 1; step <= kHomingSteps; ++step) {
    const float alpha = static_cast<float>(step) / kHomingSteps;
    std::vector<float> command(num);
    for (int index = 0; index < num; ++index) {
      command[index] = current[index] + alpha * (target[index] - current[index]);
    }
    if (!send_action(command, kHomingSpeed, kHomingAcceleration)) {
      throw FeetechHomingError(make_message(
          "Unable to complete SO-101 homing: motion step ", step, "/", kHomingSteps,
          " was not accepted. Restart the application and check whether shutdown was requested "
          "or the serial connection was interrupted."));
    }
    std::this_thread::sleep_for(kHomingStepInterval);
  }
}

FeetechSerialDriver::HomingResult FeetechSerialDriver::wait_for_homing_convergence(
    const std::vector<float>& target) {
  const int num = num_joints();
  HomingResult result{std::vector<float>(num), std::vector<float>(num)};
  const auto convergence_deadline = std::chrono::steady_clock::now() + kHomingConvergenceTimeout;
  do {
    if (read_present_positions(result.final_position)) {
      result.final_position_available = true;
      for (int index = 0; index < num; ++index) {
        result.residuals[index] = result.final_position[index] - target[index];
      }
      const auto largest_error = std::ranges::max_element(
          result.residuals,
          [](float left, float right) { return std::fabs(left) < std::fabs(right); });
      result.largest_error = std::fabs(*largest_error);
      result.largest_error_joint =
          static_cast<int>(std::distance(result.residuals.begin(), largest_error));
      if (result.largest_error <= kHomeReachTolRad) {
        result.converged = true;
        break;
      }
    }
    if (std::chrono::steady_clock::now() >= convergence_deadline) break;
    std::this_thread::sleep_for(kHomingConvergencePollInterval);
  } while (true);
  return result;
}

void FeetechSerialDriver::validate_homing_result(const std::vector<float>& target,
                                                 const HomingResult& result) const {
  if (!result.final_position_available) {
    throw FeetechHomingError(make_message(
        "Unable to verify the SO-101 home position: no final joint positions could be read within ",
        std::chrono::duration<double>(kHomingConvergenceTimeout).count(),
        " seconds. Check the serial connection and servo power."));
  }

  log(DriverLogLevel::kInfo, make_message("Final joint positions after homing (radians): [",
                                          join(result.final_position), "]."));
  log(DriverLogLevel::kInfo,
      make_message("Difference from reachable targets after homing (radians): [",
                   join(result.residuals), "]."));
  log(DriverLogLevel::kInfo,
      make_message("Largest homing difference: ", result.largest_error, " radians at joint index ",
                   result.largest_error_joint, "."));
  if (!result.converged) {
    const std::vector<std::string> joint_names = get_ordered_joint_names();
    const int index = result.largest_error_joint;
    throw FeetechHomingError(make_message(
        "The SO-101 arm did not reach its home position within ",
        std::chrono::duration<double>(kHomingConvergenceTimeout).count(), " seconds. Joint '",
        joint_names[index], "' (index ", index, ") was measured at ", result.final_position[index],
        " radians with target ", target[index], " radians and residual ", result.residuals[index],
        " radians; the allowed difference is ", kHomeReachTolRad,
        " radians. Check for a blocked joint, incorrect calibration, or insufficient servo "
        "power."));
  }
  log(DriverLogLevel::kInfo, make_message("Homing complete; every joint is within ",
                                          kHomeReachTolRad, " radians of its reachable target."));
}

void FeetechSerialDriver::home() {
  if (options_.home_position.empty()) return;
  const int num = num_joints();
  if (static_cast<int>(options_.home_position.size()) != num) {
    throw FeetechHomingError(make_message("The configured home position has ",
                                          options_.home_position.size(), " joints; expected ", num,
                                          "."));
  }
  std::this_thread::sleep_for(kHomingBusSettleDelay);
  const std::vector<float> requested(options_.home_position.begin(), options_.home_position.end());
  const std::vector<float> target = make_reachable_home_target(requested);
  std::vector<float> current(num);
  if (!read_homing_start_position(current)) {
    throw FeetechHomingError(make_message(
        "Unable to start SO-101 homing: no healthy joint state could be read after ",
        kHomingReadRetries,
        " attempts. Check the serial connection, servo power, and reported controller faults."));
  }
  send_homing_trajectory(current, target);
  validate_homing_result(target, wait_for_homing_convergence(target));
}

int FeetechSerialDriver::num_joints() const { return static_cast<int>(calib_.size()); }

std::vector<std::string> FeetechSerialDriver::get_ordered_joint_names() const {
  std::vector<std::string> names(calib_.size());
  for (const auto& [name, calibration] : calib_) {
    const int index = calibration.id - 1;
    if (index < 0 || static_cast<size_t>(index) >= names.size()) {
      throw std::runtime_error("Motor '" + name + "' has an out-of-range ID " +
                               std::to_string(calibration.id) + ".");
    }
    names[index] = name;
  }
  return names;
}

void FeetechSerialDriver::log(DriverLogLevel level, std::string message) const {
  if (log_callback_) log_callback_(level, message);
}

}  // namespace isaac::so101
