/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "common/mcap_logger/tests/scoped_temp_directory.h"
#include "common/so101_driver/feetech_serial_driver.h"

namespace isaac::so101::testing {
class SyntheticReadError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class SyntheticWriteError : public std::runtime_error {
 public:
  /**
   * Construct the injected write failure used to exercise exception recovery.
   */
  SyntheticWriteError() : std::runtime_error("Synthetic servo write failure.") {}
};

class FakeFeetechBus final : public FeetechBus {
 public:
  struct Write {
    std::uint8_t id;
    std::uint8_t address;
    int value;
  };

  struct SyncWrite {
    std::vector<std::uint8_t> ids;
    std::vector<std::int16_t> positions;
    std::vector<std::uint16_t> speeds;
    std::vector<std::uint8_t> accelerations;
  };

  /**
   * Record a successful simulated connection without opening a serial device.
   */
  bool begin(int, const char*) override {
    ++begin_count;
    return true;
  }
  /**
   * Record a simulated disconnection.
   */
  void end() override { ++end_count; }
  /**
   * Record an 8-bit register write and apply configured write failures.
   */
  int write_byte(std::uint8_t id, std::uint8_t address, std::uint8_t value) override {
    return record_write(id, address, value);
  }
  /**
   * Record a 16-bit register write and apply configured write failures.
   */
  int write_word(std::uint8_t id, std::uint8_t address, std::uint16_t value) override {
    return record_write(id, address, value);
  }
  /**
   * Return the stored register value, defaulting untouched registers to zero.
   */
  int read_byte(std::uint8_t id, std::uint8_t address) override {
    return read_.registers[register_key(id, address)];
  }
  /**
   * Return the simulated register, or inject a failed goal readback for hold verification.
   */
  int read_word(std::uint8_t id, std::uint8_t address) override {
    if (address == 42U && corrupt_goal_readback) return -1;
    return read_.registers[register_key(id, address)];
  }
  /**
   * Capture a complete motion command, then invoke the optional fault-injection callback.
   */
  void sync_write_positions(std::uint8_t* ids, std::uint8_t count, std::int16_t* positions,
                            std::uint16_t* speeds, std::uint8_t* accelerations) override {
    sync_writes.push_back({
        .ids = {ids, ids + count},
        .positions = {positions, positions + count},
        .speeds = {speeds, speeds + count},
        .accelerations = {accelerations, accelerations + count},
    });
    if (on_sync_write) on_sync_write();
  }
  /**
   * Record creation of a simulated synchronous-read session.
   */
  void sync_read_begin(std::uint8_t, std::uint8_t, std::uint32_t) override {
    ++sync_read_begin_count;
  }
  /**
   * Record termination of the simulated synchronous-read session.
   */
  void sync_read_end() override { ++sync_read_end_count; }
  /**
   * Start a feedback transaction, applying read exceptions, callbacks, and success limits.
   */
  int sync_read_packet_tx(std::uint8_t*, std::uint8_t, std::uint8_t, std::uint8_t) override {
    if (throw_on_sync_read) throw SyntheticReadError("The test bus could not read joint state.");
    ++sync_read_transaction_count;
    if (on_sync_read) on_sync_read();
    read_.available =
        max_successful_sync_reads < 0 || sync_read_transaction_count <= max_successful_sync_reads;
    return 1;
  }
  /**
   * Select a servo for decoding, or report unavailable feedback after the read limit.
   */
  int sync_read_packet_rx(std::uint8_t id, std::uint8_t*) override {
    if (!read_.available) return 0;
    read_.id = id;
    read_.next_word_is_position = true;
    return 1;
  }
  /**
   * Count status checks and return the selected servo fault byte, defaulting to healthy.
   */
  std::uint8_t last_status() override {
    ++last_status_count;
    const auto status = statuses.find(read_.id);
    return status == statuses.end() ? 0U : status->second;
  }
  /**
   * Return the selected raw position for the first word and zero velocity thereafter.
   */
  int sync_read_rx_word(std::uint8_t) override {
    if (!read_.next_word_is_position) return 0;
    read_.next_word_is_position = false;
    const auto position = present_positions.find(read_.id);
    return position == present_positions.end() ? 0 : position->second;
  }

  /**
   * Report whether the driver attempted a simulated connection.
   */
  bool begin_called() const { return begin_count > 0; }
  bool corrupt_goal_readback{};
  int begin_count{};
  int end_count{};
  int sync_read_begin_count{};
  int sync_read_end_count{};
  bool throw_on_sync_read{};
  bool throw_on_failed_write{};
  int max_successful_sync_reads{-1};
  int sync_read_transaction_count{};
  int last_status_count{};
  int failed_id{-1};
  int failed_address{-1};
  int failed_value{-1};
  int failure_budget{-1};
  std::function<void()> on_sync_write;
  std::function<void()> on_sync_read;
  std::vector<Write> writes;
  std::vector<SyncWrite> sync_writes;
  std::unordered_map<std::uint8_t, std::int16_t> present_positions;
  std::unordered_map<std::uint8_t, std::uint8_t> statuses;

  /**
   * Pack a servo ID and register address into a unique simulated-register key.
   */
  static std::uint16_t register_key(std::uint8_t id, std::uint8_t address) {
    return static_cast<std::uint16_t>((static_cast<std::uint16_t>(id) << 8U) | address);
  }

  struct ReadState {
    std::uint8_t id{};
    bool next_word_is_position{};
    bool available{true};
    std::unordered_map<std::uint16_t, int> registers;
  };
  ReadState read_;
  /**
   * Record every attempted write; update registers only when fault injection permits it.
   */
  int record_write(std::uint8_t id, std::uint8_t address, int value) {
    writes.push_back(Write{.id = id, .address = address, .value = value});
    if (const bool matches_failure =
            address == failed_address && (failed_id < 0 || id == failed_id) &&
            (failed_value < 0 || value == failed_value) && failure_budget != 0;
        matches_failure) {
      if (failure_budget > 0) --failure_budget;
      if (throw_on_failed_write) throw SyntheticWriteError{};
      return 0;
    }
    read_.registers[register_key(id, address)] = value;
    return 1;
  }
};

class CalibrationFile {
 public:
  /**
   * Write temporary six-joint calibration with configurable IDs, directions, and offsets.
   */
  CalibrationFile(std::string_view name, const std::array<int, 6U>& ids,
                  const std::array<int, 6U>& drive_modes = {},
                  const std::array<int, 6U>& homing_offsets = {})
      : path_(directory_.path(name)) {
    std::ofstream file(path_);
    file << "{\n";
    for (std::size_t index = 0; index < kSo101JointNames.size(); ++index) {
      file << R"(  ")" << kSo101JointNames[index] << R"(": {"id": )" << ids[index]
           << R"(, "drive_mode": )" << drive_modes[index] << R"(, "homing_offset": )"
           << homing_offsets[index] << R"(, "range_min": 1000, "range_max": 3000})";
      file << (index + 1U == kSo101JointNames.size() ? "\n" : ",\n");
    }
    file << "}\n";
  }

  CalibrationFile(const CalibrationFile&) = delete;
  CalibrationFile& operator=(const CalibrationFile&) = delete;

  /**
   * Return the calibration path, valid for the lifetime of this temporary-file owner.
   */
  const std::filesystem::path& path() const { return path_; }

 private:
  isaac::os::mcap_logger::ScopedTempDirectory directory_;
  std::filesystem::path path_;
};

/**
 * Create driver options using the fake serial port and the supplied calibration file.
 */
inline FeetechDriverOptions options_for(const CalibrationFile& calibration) {
  return {
      .serial_port = "fake-port",
      .calibration_file = calibration.path().string(),
  };
}

}  // namespace isaac::so101::testing
