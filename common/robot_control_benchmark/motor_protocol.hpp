/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Simulated 8-byte motor command: STX 0xAA, joint id, cmd 0x01, float32
// radians little-endian, CRC-8 (poly 0x07) over the first seven bytes.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "common/robot_control_benchmark/payloads.hpp"

namespace isaac::os::benchmark {

inline constexpr std::uint8_t kCmdStx = 0xAAU;
inline constexpr std::uint8_t kCmdPosition = 0x01U;

inline std::uint8_t crc8(const std::uint8_t* data, std::size_t len) {
  std::uint8_t crc = 0x00U;
  for (std::size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x80U) ? static_cast<std::uint8_t>((crc << 1) ^ 0x07U)
                          : static_cast<std::uint8_t>(crc << 1);
    }
  }
  return crc;
}

inline void encode_joint_packet(std::uint8_t* pkt, std::uint8_t joint_id, float setpoint_rad) {
  pkt[0] = kCmdStx;
  pkt[1] = joint_id;
  pkt[2] = kCmdPosition;
  std::memcpy(&pkt[3], &setpoint_rad, sizeof(float));
  pkt[7] = crc8(pkt, 7);
}

inline void encode_batch(const JointSetpoints& sp, MotorCommandBatch& out) {
  out.seq = sp.seq;
  out.t_src_ns = sp.t_src_ns;
  for (int j = 0; j < kNumJoints; ++j) {
    encode_joint_packet(&out.packets[j * kPacketBytes], static_cast<std::uint8_t>(j),
                        static_cast<float>(sp.q[j]));
  }
}

}  // namespace isaac::os::benchmark
