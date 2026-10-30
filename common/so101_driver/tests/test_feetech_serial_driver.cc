/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "common/so101_driver/feetech_serial_driver.h"
#include "common/so101_driver/tests/feetech_test_support.h"

namespace isaac::so101 {
namespace {

constexpr std::uint8_t kOperatingModeAddress = 33U;
constexpr std::uint8_t kReturnDelayAddress = 7U;
constexpr std::uint8_t kTorqueEnableAddress = 40U;
constexpr std::uint8_t kAccelerationAddress = 41U;
constexpr std::uint8_t kICoefficientAddress = 23U;
constexpr std::uint8_t kLockAddress = 55U;

using testing::CalibrationFile;
using testing::FakeFeetechBus;
using testing::options_for;

TEST(FeetechSerialDriver, RejectsDuplicateMotorIdsBeforeOpeningTheSerialPort) {
  const CalibrationFile calibration{"so101_duplicate_ids.json", {1, 2, 3, 4, 5, 5}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus* bus_view = bus.get();
  FeetechSerialDriver driver{std::move(bus)};

  EXPECT_THROW(driver.initialize(options_for(calibration)), std::runtime_error);
  EXPECT_FALSE(bus_view->begin_called());
}

TEST(FeetechSerialDriver, RejectsUnexpectedJointOrderBeforeOpeningTheSerialPort) {
  const CalibrationFile calibration{"so101_swapped_joints.json", {2, 1, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus* bus_view = bus.get();
  FeetechSerialDriver driver{std::move(bus)};

  EXPECT_THROW(driver.initialize(options_for(calibration)), std::runtime_error);
  EXPECT_FALSE(bus_view->begin_called());
}

TEST(FeetechSerialDriver, RejectsUnrepresentableHomingOffsetsBeforeOpeningTheSerialPort) {
  for (const int invalid_offset : {-2048, 2048}) {
    const CalibrationFile calibration{
        "so101_invalid_homing_offset_" + std::to_string(invalid_offset) + ".json",
        {1, 2, 3, 4, 5, 6},
        {},
        {invalid_offset, 0, 0, 0, 0, 0}};
    auto bus = std::make_unique<FakeFeetechBus>();
    FakeFeetechBus* bus_view = bus.get();
    FeetechSerialDriver driver{std::move(bus)};

    EXPECT_THROW(driver.initialize(options_for(calibration)), std::runtime_error);
    EXPECT_FALSE(bus_view->begin_called());
  }
}

TEST(FeetechSerialDriver, UsesPositionModeForReversedJoints) {
  const CalibrationFile calibration{
      "so101_reversed_joint.json", {1, 2, 3, 4, 5, 6}, {1, 0, 0, 0, 0, 0}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus* bus_view = bus.get();
  FeetechSerialDriver driver{std::move(bus)};

  ASSERT_NO_THROW(driver.initialize(options_for(calibration)));
  std::size_t operating_mode_writes = 0U;
  for (const auto& write : bus_view->writes) {
    if (write.address == kOperatingModeAddress) {
      ++operating_mode_writes;
      EXPECT_EQ(write.value, 0);
    }
  }
  EXPECT_EQ(operating_mode_writes, kSo101JointNames.size());
}

TEST(FeetechSerialDriver, RejectsAnUnacknowledgedSetupWrite) {
  const CalibrationFile calibration{"so101_missing_ack.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus* bus_view = bus.get();
  bus->failed_address = kReturnDelayAddress;
  bus->failure_budget = 1;
  FeetechSerialDriver driver{std::move(bus)};

  try {
    driver.initialize(options_for(calibration));
    FAIL() << "Initialization accepted a setup write without an acknowledgement.";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("did not acknowledge"), std::string::npos);
  }
  EXPECT_EQ(bus_view->begin_count, 1);
  EXPECT_EQ(bus_view->end_count, 1);
  EXPECT_NO_THROW(driver.initialize(options_for(calibration)));
  EXPECT_EQ(bus_view->begin_count, 2);
}

TEST(FeetechSerialDriver, ResetsSyncReadStateWhenHomingThrows) {
  const CalibrationFile calibration{"so101_homing_exception.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus* bus_view = bus.get();
  bus->throw_on_sync_read = true;
  FeetechSerialDriver driver{std::move(bus)};
  FeetechDriverOptions options = options_for(calibration);
  options.home_position.assign(6U, 0.0);

  EXPECT_THROW(driver.initialize(options), std::runtime_error);
  EXPECT_EQ(bus_view->sync_read_begin_count, 1);
  EXPECT_EQ(bus_view->sync_read_end_count, 1);
  EXPECT_EQ(bus_view->end_count, 1);

  bus_view->throw_on_sync_read = false;
  options.home_position.clear();
  EXPECT_NO_THROW(driver.initialize(options));
  EXPECT_EQ(bus_view->begin_count, 2);
}

TEST(FeetechSerialDriver, AcceptsHealthyStatusFromEveryPresentStateResponse) {
  const CalibrationFile calibration{"so101_healthy_status.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  const FakeFeetechBus* bus_view = bus.get();
  FeetechSerialDriver driver{std::move(bus)};

  ASSERT_NO_THROW(driver.initialize(options_for(calibration)));
  std::vector<float> positions(6U);
  EXPECT_TRUE(driver.read_present_positions(positions));
  EXPECT_EQ(bus_view->last_status_count, 6);
}

TEST(FeetechSerialDriver, RejectsOverloadStatusDuringHomingAndRollsBack) {
  const CalibrationFile calibration{"so101_overload_status.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus& bus_view = *bus;
  for (std::uint8_t id = 1; id < 6; ++id) {
    bus->present_positions[id] = 2048;
  }
  bus->present_positions[6] = 1000;
  bus->statuses[2] = 0x20U;
  FeetechSerialDriver driver{std::move(bus)};
  FeetechDriverOptions options = options_for(calibration);
  options.home_position.assign(6U, 0.0);
  std::string status_log;

  try {
    driver.initialize(options, [&status_log](DriverLogLevel level, std::string_view message) {
      if (level == DriverLogLevel::kError) {
        status_log = message;
      }
    });
    FAIL() << "Initialization accepted a servo overload during homing.";
  } catch (const FeetechHomingError& error) {
    EXPECT_NE(std::string{error.what()}.find("no healthy joint state"), std::string::npos);
  }

  EXPECT_NE(status_log.find("'shoulder_lift' (ID 2)"), std::string::npos);
  EXPECT_NE(status_log.find("0x20 (overload)"), std::string::npos);
  EXPECT_NE(status_log.find("switch off follower power"), std::string::npos);
  EXPECT_GT(bus_view.last_status_count, 0);
  EXPECT_EQ(bus_view.begin_count, 1);
  EXPECT_EQ(bus_view.end_count, 1);
  EXPECT_EQ(bus_view.sync_read_begin_count, 1);
  EXPECT_EQ(bus_view.sync_read_end_count, 1);
  for (std::uint8_t id = 1; id <= 6; ++id) {
    EXPECT_EQ(bus_view.read_byte(id, kTorqueEnableAddress), 0) << "motor ID " << int{id};
  }
}

/**
 * Disable every servo when final homing feedback and the recovery hold cannot be read.
 */
TEST(FeetechSerialDriver, RejectsAnUnreadableFinalHomePositionAndRollsBack) {
  const CalibrationFile calibration{"so101_unreadable_final_home.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus& bus_view = *bus;
  for (std::uint8_t id = 1; id < 6; ++id) {
    bus->present_positions[id] = 2048;
  }
  bus->present_positions[6] = 1000;
  bus->max_successful_sync_reads = 1;
  FeetechSerialDriver driver{std::move(bus)};
  FeetechDriverOptions options = options_for(calibration);
  options.home_position.assign(6U, 0.0);

  try {
    driver.initialize(options);
    FAIL() << "Initialization accepted an unreadable final home position.";
  } catch (const FeetechHomingError& error) {
    const std::string message{error.what()};
    EXPECT_NE(message.find("Unable to verify the SO-101 home position"), std::string::npos);
    EXPECT_NE(message.find("serial connection and servo power"), std::string::npos);
  }

  EXPECT_EQ(bus_view.begin_count, 1);
  EXPECT_EQ(bus_view.end_count, 1);
  EXPECT_EQ(bus_view.sync_read_begin_count, 1);
  EXPECT_EQ(bus_view.sync_read_end_count, 1);
  for (std::uint8_t id = 1; id <= 6; ++id) {
    EXPECT_EQ(bus_view.read_byte(id, kTorqueEnableAddress), 0) << "motor ID " << int{id};
  }
}

/**
 * Preserve raw positions outside calibration limits and clear integral gain after failed homing.
 */
TEST(FeetechSerialDriver, RejectsANonconvergentHomeAndHoldsPosition) {
  const CalibrationFile calibration{"so101_nonconvergent_home.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus& bus_view = *bus;
  for (std::uint8_t id = 1; id < 6; ++id) {
    bus->present_positions[id] = 2048;
  }
  bus->present_positions[2] = 1800;
  // The hold must preserve raw feedback even outside the calibration range.
  bus->present_positions[6] = 999;
  FeetechSerialDriver driver{std::move(bus)};
  FeetechDriverOptions options = options_for(calibration);
  options.home_position.assign(6U, 0.0);

  try {
    driver.initialize(options);
    FAIL() << "Initialization accepted a home position that did not converge.";
  } catch (const FeetechHomingError& error) {
    const std::string message{error.what()};
    EXPECT_NE(message.find("did not reach its home position"), std::string::npos);
    EXPECT_NE(message.find("shoulder_lift"), std::string::npos);
    EXPECT_NE(message.find("(index 1)"), std::string::npos);
    EXPECT_NE(message.find("measured at"), std::string::npos);
    EXPECT_NE(message.find("target"), std::string::npos);
    EXPECT_NE(message.find("residual"), std::string::npos);
  }

  EXPECT_EQ(bus_view.begin_count, 1);
  EXPECT_EQ(bus_view.end_count, 1);
  EXPECT_EQ(bus_view.sync_read_begin_count, 1);
  EXPECT_EQ(bus_view.sync_read_end_count, 1);
  EXPECT_EQ(bus_view.read_word(2, 42U), 1800);
  EXPECT_EQ(bus_view.read_word(6, 42U), 999);
  for (std::uint8_t id = 1; id <= 6; ++id) {
    EXPECT_EQ(bus_view.read_byte(id, kICoefficientAddress), 0);
    EXPECT_EQ(bus_view.read_byte(id, kLockAddress), 1);
    EXPECT_EQ(bus_view.read_byte(id, kTorqueEnableAddress), 1) << "motor ID " << int{id};
  }
}

/**
 * Disable all servos when a hold write fails while preserving the original homing error.
 */
TEST(FeetechSerialDriver, DisablesTorqueWhenStartupHoldWriteFails) {
  const CalibrationFile calibration{"so101_failed_startup_hold.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus& bus_view = *bus;
  for (std::uint8_t id = 1; id < 6; ++id) {
    bus->present_positions[id] = 2048;
  }
  bus->present_positions[2] = 1800;
  bus->present_positions[6] = 1000;
  bus->failed_id = 2;
  bus->failed_address = 42;
  FeetechSerialDriver driver{std::move(bus)};
  FeetechDriverOptions options = options_for(calibration);
  options.home_position.assign(6U, 0.0);

  try {
    driver.initialize(options);
    FAIL() << "Initialization accepted a home position that did not converge.";
  } catch (const FeetechHomingError& error) {
    const std::string message{error.what()};
    EXPECT_NE(message.find("did not reach its home position"), std::string::npos);
    EXPECT_NE(message.find("shoulder_lift"), std::string::npos);
    EXPECT_NE(message.find("(index 1)"), std::string::npos);
    EXPECT_NE(message.find("measured at"), std::string::npos);
    EXPECT_NE(message.find("target"), std::string::npos);
    EXPECT_NE(message.find("residual"), std::string::npos);
  }

  EXPECT_EQ(bus_view.begin_count, 1);
  EXPECT_EQ(bus_view.end_count, 1);
  EXPECT_EQ(bus_view.sync_read_begin_count, 1);
  EXPECT_EQ(bus_view.sync_read_end_count, 1);
  for (std::uint8_t id = 1; id <= 6; ++id) {
    EXPECT_EQ(bus_view.read_byte(id, kTorqueEnableAddress), 0) << "motor ID " << int{id};
  }
}

/**
 * Disable all servos when hold readback fails while preserving the original homing error.
 */
TEST(FeetechSerialDriver, DisablesTorqueWhenStartupHoldReadbackFails) {
  const CalibrationFile calibration{"so101_failed_hold_readback.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus& bus_view = *bus;
  for (std::uint8_t id = 1; id < 6; ++id) {
    bus->present_positions[id] = 2048;
  }
  bus->present_positions[2] = 1800;
  bus->present_positions[6] = 1000;
  bus->corrupt_goal_readback = true;
  FeetechSerialDriver driver{std::move(bus)};
  FeetechDriverOptions options = options_for(calibration);
  options.home_position.assign(6U, 0.0);

  try {
    driver.initialize(options);
    FAIL() << "Initialization accepted a home position that did not converge.";
  } catch (const FeetechHomingError& error) {
    const std::string message{error.what()};
    EXPECT_NE(message.find("did not reach its home position"), std::string::npos);
    EXPECT_NE(message.find("shoulder_lift"), std::string::npos);
    EXPECT_NE(message.find("(index 1)"), std::string::npos);
    EXPECT_NE(message.find("measured at"), std::string::npos);
    EXPECT_NE(message.find("target"), std::string::npos);
    EXPECT_NE(message.find("residual"), std::string::npos);
  }

  EXPECT_EQ(bus_view.begin_count, 1);
  EXPECT_EQ(bus_view.end_count, 1);
  EXPECT_EQ(bus_view.sync_read_begin_count, 1);
  EXPECT_EQ(bus_view.sync_read_end_count, 1);
  for (std::uint8_t id = 1; id <= 6; ++id) {
    EXPECT_EQ(bus_view.read_byte(id, kTorqueEnableAddress), 0) << "motor ID " << int{id};
  }
}

/**
 * Retain a verified position hold when a homing trajectory write is interrupted.
 */
TEST(FeetechSerialDriver, RejectsAnInterruptedHomingTrajectoryAndHoldsPosition) {
  const CalibrationFile calibration{"so101_interrupted_home.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus& bus_view = *bus;
  for (std::uint8_t id = 1; id < 6; ++id) {
    bus->present_positions[id] = 2048;
  }
  bus->present_positions[6] = 1000;
  std::atomic homing_write_started{false};
  std::atomic hold_succeeded{false};
  bus->on_sync_write = [&homing_write_started] { homing_write_started.store(true); };
  FeetechSerialDriver driver{std::move(bus)};
  FeetechDriverOptions options = options_for(calibration);
  options.home_position.assign(6U, 0.0);
  std::jthread interrupt_homing(
      [&driver, &hold_succeeded, &homing_write_started](std::stop_token stop_token) {
        while (!homing_write_started.load() && !stop_token.stop_requested()) {
          std::this_thread::yield();
        }
        if (homing_write_started.load()) {
          hold_succeeded.store(driver.hold_present_position());
        }
      });

  try {
    driver.initialize(options);
    FAIL() << "Initialization accepted an interrupted homing trajectory.";
  } catch (const FeetechHomingError& error) {
    const std::string message{error.what()};
    EXPECT_NE(message.find("Unable to complete SO-101 homing"), std::string::npos);
    EXPECT_NE(message.find("motion step"), std::string::npos);
  }
  interrupt_homing.request_stop();
  interrupt_homing.join();

  EXPECT_TRUE(hold_succeeded.load());
  EXPECT_EQ(bus_view.begin_count, 1);
  EXPECT_EQ(bus_view.end_count, 1);
  for (std::uint8_t id = 1; id <= 6; ++id) {
    EXPECT_EQ(bus_view.read_byte(id, kTorqueEnableAddress), 1) << "motor ID " << int{id};
  }
}

TEST(FeetechSerialDriver, RetriesAMissedTorqueEnableAcknowledgement) {
  const CalibrationFile calibration{"so101_torque_retry.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus* bus_view = bus.get();
  bus->failed_id = 4;
  bus->failed_address = kTorqueEnableAddress;
  bus->failed_value = 1;
  bus->failure_budget = 1;
  FeetechSerialDriver driver{std::move(bus)};
  FeetechDriverOptions options = options_for(calibration);
  options.position_p_gain = 16;
  options.position_i_gain = 16;
  options.position_d_gain = 32;
  std::string gain_log;

  ASSERT_NO_THROW(driver.initialize(options, [&gain_log](DriverLogLevel, std::string_view message) {
    if (message.starts_with("Configured runtime position-loop gains")) gain_log = message;
  }));
  EXPECT_EQ(bus_view->read_byte(4, kTorqueEnableAddress), 1);
  EXPECT_EQ(gain_log, "Configured runtime position-loop gains: P=16, I=16, D=32.");
}

TEST(FeetechSerialDriver, HomesWithIntegralActionThenRestoresTheConfiguredIGain) {
  const CalibrationFile calibration{"so101_homing_i_gain.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus* bus_view = bus.get();
  FeetechSerialDriver driver{std::move(bus)};
  FeetechDriverOptions options = options_for(calibration);
  options.position_p_gain = 16;
  options.position_i_gain = 0;
  options.position_d_gain = 32;
  std::vector<std::string> gain_logs;

  ASSERT_NO_THROW(
      driver.initialize(options, [&gain_logs](DriverLogLevel, std::string_view message) {
        if (message.starts_with("Configured runtime position-loop gains") ||
            message.starts_with("Homing temporarily uses") ||
            message.starts_with("Homing complete; applied runtime position-loop gains")) {
          gain_logs.emplace_back(message);
        }
      }));

  const std::vector<std::string> expected_logs{
      "Configured runtime position-loop gains: P=16, I=0, D=32.",
      "Homing temporarily uses P=16, I=16, D=32 to reach the homing tolerance.",
      "Homing complete; applied runtime position-loop gains: P=16, I=0, D=32."};
  EXPECT_EQ(gain_logs, expected_logs);
  for (std::uint8_t id = 1; id <= 6; ++id) {
    EXPECT_EQ(bus_view->read_byte(id, kICoefficientAddress), 0) << "motor ID " << int{id};
    EXPECT_EQ(bus_view->read_byte(id, kLockAddress), 1) << "motor ID " << int{id};
  }
}

TEST(FeetechSerialDriver, KeepsTheConfiguredIGainWhenItAlreadyAllowsHoming) {
  const CalibrationFile calibration{"so101_homing_i_gain_kept.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus* bus_view = bus.get();
  FeetechSerialDriver driver{std::move(bus)};
  FeetechDriverOptions options = options_for(calibration);
  options.position_p_gain = 16;
  options.position_i_gain = 32;
  options.position_d_gain = 32;

  std::vector<std::string> gain_logs;
  ASSERT_NO_THROW(
      driver.initialize(options, [&gain_logs](DriverLogLevel, std::string_view message) {
        if (message.starts_with("Configured runtime position-loop gains") ||
            message.starts_with("Homing uses") || message.starts_with("Homing temporarily uses") ||
            message.starts_with("Homing complete; applied runtime position-loop gains")) {
          gain_logs.emplace_back(message);
        }
      }));
  const std::vector<std::string> expected_logs{
      "Configured runtime position-loop gains: P=16, I=32, D=32.",
      "Homing uses the configured runtime position-loop gains.",
      "Homing complete; applied runtime position-loop gains: P=16, I=32, D=32."};
  EXPECT_EQ(gain_logs, expected_logs);

  for (std::uint8_t id = 1; id <= 6; ++id) {
    EXPECT_EQ(bus_view->read_byte(id, kICoefficientAddress), 32) << "motor ID " << int{id};
  }
}

TEST(FeetechSerialDriver, DisablesEveryMotorWhenTorqueEnableFails) {
  const CalibrationFile calibration{"so101_torque_rollback.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus* bus_view = bus.get();
  bus->failed_id = 4;
  bus->failed_address = kTorqueEnableAddress;
  bus->failed_value = 1;
  FeetechSerialDriver driver{std::move(bus)};

  EXPECT_THROW(driver.initialize(options_for(calibration)), std::runtime_error);
  for (std::uint8_t id = 1; id <= 6; ++id) {
    EXPECT_EQ(bus_view->read_byte(id, kTorqueEnableAddress), 0) << "motor ID " << int{id};
  }
}

TEST(FeetechSerialDriver, StreamsTheConfiguredMotionProfileWithEveryCommand) {
  for (const int acceleration : {0, 150}) {
    const CalibrationFile calibration{std::format("so101_motion_profile_{}.json", acceleration),
                                      {1, 2, 3, 4, 5, 6}};
    auto bus = std::make_unique<FakeFeetechBus>();
    FakeFeetechBus* bus_view = bus.get();
    FeetechSerialDriver driver{std::move(bus)};
    FeetechDriverOptions options = options_for(calibration);
    options.run_speed = 1500;
    options.run_acceleration = acceleration;

    ASSERT_NO_THROW(driver.initialize(options));
    EXPECT_EQ(bus_view->read_byte(1, kAccelerationAddress), acceleration);
    ASSERT_TRUE(driver.send_action(std::vector<float>(6U, 0.0F)));
    ASSERT_EQ(bus_view->sync_writes.size(), 1U);
    EXPECT_EQ(bus_view->sync_writes.front().ids,
              (std::vector<std::uint8_t>{1U, 2U, 3U, 4U, 5U, 6U}));
    EXPECT_EQ(bus_view->sync_writes.front().speeds, (std::vector<std::uint16_t>(6U, 1500U)));
    EXPECT_EQ(bus_view->sync_writes.front().accelerations,
              (std::vector<std::uint8_t>(6U, static_cast<std::uint8_t>(acceleration))));
  }
}

TEST(FeetechSerialDriver, UsesCalibratedClosedEndpointForGripperCommandsAndState) {
  for (const int drive_mode : {0, 1}) {
    std::array<int, 6U> drive_modes{};
    drive_modes[5] = drive_mode;
    const CalibrationFile calibration{
        std::format("so101_gripper_origin_{}.json", drive_mode), {1, 2, 3, 4, 5, 6}, drive_modes};
    auto bus = std::make_unique<FakeFeetechBus>();
    FakeFeetechBus* bus_view = bus.get();
    const std::int16_t closed_encoder = drive_mode == 0 ? 1000 : 3000;
    bus->present_positions[6U] = closed_encoder;
    FeetechSerialDriver driver{std::move(bus)};

    ASSERT_NO_THROW(driver.initialize(options_for(calibration)));
    ASSERT_TRUE(driver.send_action(std::vector<float>(6U, 0.0F)));
    ASSERT_EQ(bus_view->sync_writes.size(), 1U);
    std::vector<std::int16_t> expected_command(6U, 2048);
    expected_command[5] = closed_encoder;
    EXPECT_EQ(bus_view->sync_writes.front().positions, expected_command);

    std::vector<float> positions(6U);
    ASSERT_TRUE(driver.read_present_positions(positions));
    EXPECT_FLOAT_EQ(positions[5], 0.0F);

    ASSERT_TRUE(driver.hold_present_position());
    ASSERT_EQ(bus_view->sync_writes.size(), 2U);
    EXPECT_EQ(bus_view->sync_writes.back().positions[5], closed_encoder);
  }
}

TEST(FeetechSerialDriver, ShutdownRemovesIntegralCorrectionWithoutDisablingTorque) {
  const CalibrationFile calibration{"so101_shutdown_gain.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus* bus_view = bus.get();
  FeetechSerialDriver driver{std::move(bus)};
  auto options = options_for(calibration);
  options.position_i_gain = 16;
  ASSERT_NO_THROW(driver.initialize(options));
  ASSERT_TRUE(driver.hold_present_position());
  for (std::uint8_t id = 1; id <= 6; ++id) {
    EXPECT_EQ(bus_view->read_byte(id, kICoefficientAddress), 0);
    EXPECT_EQ(bus_view->read_byte(id, kTorqueEnableAddress), 1);
    EXPECT_EQ(bus_view->read_byte(id, kLockAddress), 1);
  }
  const auto writes = bus_view->writes.size();
  ASSERT_TRUE(driver.hold_present_position());
  EXPECT_EQ(bus_view->writes.size(), writes);
}

TEST(FeetechSerialDriver, ShutdownRetriesTransientConfigurationFailuresBeforeReturning) {
  struct Failure {
    bool throw_on_failure;
    std::uint8_t address;
    int value;
  };
  for (const auto [throw_on_failure, address, value] :
       {Failure{false, kLockAddress, 0}, Failure{false, kICoefficientAddress, 0},
        Failure{false, kLockAddress, 1}, Failure{true, kLockAddress, 0},
        Failure{true, kICoefficientAddress, 0}, Failure{true, kLockAddress, 1}}) {
    const CalibrationFile calibration{"so101_shutdown_transient.json", {1, 2, 3, 4, 5, 6}};
    auto bus = std::make_unique<FakeFeetechBus>();
    FakeFeetechBus* bus_view = bus.get();
    FeetechSerialDriver driver{std::move(bus)};
    auto options = options_for(calibration);
    options.position_i_gain = 16;
    ASSERT_NO_THROW(driver.initialize(options));
    bus_view->failed_id = 3;
    bus_view->failed_address = address;
    bus_view->failed_value = value;
    bus_view->throw_on_failed_write = throw_on_failure;
    // Exhaust the first preparation attempt, then allow the bounded retry to succeed.
    bus_view->failure_budget = throw_on_failure ? 1 : 3;
    const auto motion_writes = bus_view->sync_writes.size();
    ASSERT_TRUE(driver.hold_present_position());
    EXPECT_EQ(bus_view->sync_writes.size(), motion_writes + 1);
    for (std::uint8_t id = 1; id <= 6; ++id) {
      EXPECT_EQ(bus_view->read_byte(id, kICoefficientAddress), 0);
      EXPECT_EQ(bus_view->read_byte(id, kTorqueEnableAddress), 1);
      EXPECT_EQ(bus_view->read_byte(id, kLockAddress), 1);
    }
    const auto configuration_writes = bus_view->writes.size();
    EXPECT_TRUE(driver.hold_present_position());
    EXPECT_EQ(bus_view->writes.size(), configuration_writes);
    EXPECT_EQ(bus_view->sync_writes.size(), motion_writes + 1);
    EXPECT_FALSE(driver.send_action(std::vector<float>(6U, 0.25F)));
  }
}

TEST(FeetechSerialDriver, ShutdownWriteFailuresDoNotSkipRemainingServos) {
  struct Failure {
    bool throw_on_failure;
    std::uint8_t address;
    int value;
  };
  for (const auto [throw_on_failure, address, value] :
       {Failure{false, kLockAddress, 0}, Failure{false, kICoefficientAddress, 0},
        Failure{false, kLockAddress, 1}, Failure{true, kLockAddress, 0},
        Failure{true, kICoefficientAddress, 0}, Failure{true, kLockAddress, 1}}) {
    const CalibrationFile calibration{"so101_shutdown_failure.json", {1, 2, 3, 4, 5, 6}};
    auto bus = std::make_unique<FakeFeetechBus>();
    FakeFeetechBus* bus_view = bus.get();
    FeetechSerialDriver driver{std::move(bus)};
    auto options = options_for(calibration);
    options.position_i_gain = 16;
    ASSERT_NO_THROW(driver.initialize(options));
    bus_view->failed_id = 3;
    bus_view->failed_address = address;
    bus_view->failed_value = value;
    bus_view->throw_on_failed_write = throw_on_failure;
    const auto writes_before_hold = bus_view->writes.size();
    EXPECT_FALSE(driver.hold_present_position());
    for (std::uint8_t id = 1; id <= 6; ++id) {
      EXPECT_EQ(bus_view->read_byte(id, kTorqueEnableAddress), 1);
      if (id != 3) {
        EXPECT_EQ(bus_view->read_byte(id, kICoefficientAddress), 0);
        EXPECT_EQ(bus_view->read_byte(id, kLockAddress), 1);
      }
    }
    bool relock_attempted = false;
    for (std::size_t i = writes_before_hold; i < bus_view->writes.size(); ++i) {
      const auto& write = bus_view->writes[i];
      if (write.id == 3 && write.address == kLockAddress && write.value == 1) {
        relock_attempted = true;
      }
    }
    EXPECT_TRUE(relock_attempted);
    EXPECT_FALSE(driver.send_action(std::vector<float>(6U, 0.25F)));
    bus_view->failure_budget = 0;
    EXPECT_TRUE(driver.hold_present_position());
    for (std::uint8_t id = 1; id <= 6; ++id) {
      EXPECT_EQ(bus_view->read_byte(id, kICoefficientAddress), 0);
      EXPECT_EQ(bus_view->read_byte(id, kLockAddress), 1);
    }
  }
}

TEST(FeetechSerialDriver, RejectsEveryMotionCommandAfterTheShutdownHold) {
  const CalibrationFile calibration{"so101_shutdown_latch.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  FakeFeetechBus* bus_view = bus.get();
  for (std::uint8_t id = 1; id <= 6; ++id) {
    bus->present_positions[id] = 2048;
  }
  FeetechSerialDriver driver{std::move(bus)};

  ASSERT_NO_THROW(driver.initialize(options_for(calibration)));
  ASSERT_TRUE(driver.send_action(std::vector<float>(6U, 0.25F)));
  ASSERT_TRUE(driver.hold_present_position());
  ASSERT_EQ(bus_view->sync_writes.size(), 2U);
  const FakeFeetechBus::SyncWrite shutdown_hold = bus_view->sync_writes.back();

  EXPECT_FALSE(driver.send_action(std::vector<float>(6U, -0.25F)));
  EXPECT_TRUE(driver.hold_present_position());
  ASSERT_EQ(bus_view->sync_writes.size(), 2U);
  EXPECT_EQ(bus_view->sync_writes.back().positions, shutdown_hold.positions);
}

TEST(FeetechSerialDriver, ReportsReadFailureOnceUntilACompleteReadSucceeds) {
  const CalibrationFile calibration{"so101_read_failure_streak.json", {1, 2, 3, 4, 5, 6}};
  auto bus = std::make_unique<FakeFeetechBus>();
  auto* bus_view = bus.get();
  FeetechSerialDriver driver{std::move(bus)};
  int read_errors = 0;
  driver.initialize(options_for(calibration),
                    [&read_errors](DriverLogLevel level, std::string_view message) {
                      if (level == DriverLogLevel::kError &&
                          message.starts_with("Unable to read the present state")) {
                        ++read_errors;
                      }
                    });
  std::vector<float> positions(6U);
  bus_view->max_successful_sync_reads = 0;
  for (int attempt = 0; attempt < 100; ++attempt)
    EXPECT_FALSE(driver.read_present_positions(positions));
  EXPECT_EQ(read_errors, 1);
  bus_view->max_successful_sync_reads = -1;
  ASSERT_TRUE(driver.read_present_positions(positions));
  bus_view->max_successful_sync_reads = 0;
  EXPECT_FALSE(driver.read_present_positions(positions));
  EXPECT_EQ(read_errors, 2);
}

}  // namespace
}  // namespace isaac::so101
