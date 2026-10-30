/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Holoscan 5 POD port contracts for common/robot_control_benchmark/payloads.hpp. Field tables must
// cover every object byte exactly once.

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/channel/pod_type.hpp>

#include "common/robot_control_benchmark/payloads.hpp"

namespace holoscan {

template <>
struct ChannelTypeIdentity<isaac::os::benchmark::Pose6> {
  static constexpr std::string_view name{"isaac.os.benchmark.Pose6"};
  static constexpr std::uint32_t version{1U};
};

template <>
struct ChannelPodLayout<isaac::os::benchmark::Pose6> {
  using Payload = isaac::os::benchmark::Pose6;
  static constexpr auto fields =
      pod_fields(pod_field("seq", &Payload::seq, offsetof(Payload, seq)),
                 pod_field("t_src_ns", &Payload::t_src_ns, offsetof(Payload, t_src_ns)),
                 pod_field("x", &Payload::x, offsetof(Payload, x)),
                 pod_field("y", &Payload::y, offsetof(Payload, y)),
                 pod_field("z", &Payload::z, offsetof(Payload, z)),
                 pod_field("rx", &Payload::rx, offsetof(Payload, rx)),
                 pod_field("ry", &Payload::ry, offsetof(Payload, ry)),
                 pod_field("rz", &Payload::rz, offsetof(Payload, rz)),
                 pod_field("version", &Payload::version, offsetof(Payload, version)),
                 pod_field("produced_ns", &Payload::produced_ns, offsetof(Payload, produced_ns)));
};

template <>
struct ChannelTypeIdentity<isaac::os::benchmark::JointStateSample> {
  static constexpr std::string_view name{"isaac.os.benchmark.JointStateSample"};
  static constexpr std::uint32_t version{2U};
};

template <>
struct ChannelPodLayout<isaac::os::benchmark::JointStateSample> {
  using Payload = isaac::os::benchmark::JointStateSample;
  static constexpr auto fields =
      pod_fields(pod_field("sequence", &Payload::sequence, offsetof(Payload, sequence)),
                 pod_field("scheduled_ns", &Payload::scheduled_ns, offsetof(Payload, scheduled_ns)),
                 pod_field("callback_start_ns", &Payload::callback_start_ns,
                           offsetof(Payload, callback_start_ns)),
                 pod_field("position", &Payload::position, offsetof(Payload, position)));
};

template <>
struct ChannelTypeIdentity<isaac::os::benchmark::JointSetpoints> {
  static constexpr std::string_view name{"isaac.os.benchmark.JointSetpoints"};
  static constexpr std::uint32_t version{3U};
};

template <>
struct ChannelPodLayout<isaac::os::benchmark::JointSetpoints> {
  using Payload = isaac::os::benchmark::JointSetpoints;
  static constexpr auto fields = pod_fields(
      pod_field("seq", &Payload::seq, offsetof(Payload, seq)),
      pod_field("t_src_ns", &Payload::t_src_ns, offsetof(Payload, t_src_ns)),
      pod_field("q", &Payload::q, offsetof(Payload, q)),
      pod_field("callback_start_ns", &Payload::callback_start_ns,
                offsetof(Payload, callback_start_ns)),
      pod_field("measured_q", &Payload::measured_q, offsetof(Payload, measured_q)),
      pod_field("target_produced_ns", &Payload::target_produced_ns,
                offsetof(Payload, target_produced_ns)),
      pod_field("target_use_ns", &Payload::target_use_ns, offsetof(Payload, target_use_ns)),
      pod_field("target_version", &Payload::target_version, offsetof(Payload, target_version)));
};

template <>
struct ChannelTypeIdentity<isaac::os::benchmark::MotorCommandBatch> {
  static constexpr std::string_view name{"isaac.os.benchmark.MotorCommandBatch"};
  static constexpr std::uint32_t version{3U};
};

template <>
struct ChannelPodLayout<isaac::os::benchmark::MotorCommandBatch> {
  using Payload = isaac::os::benchmark::MotorCommandBatch;
  static constexpr auto fields = pod_fields(
      pod_field("seq", &Payload::seq, offsetof(Payload, seq)),
      pod_field("t_src_ns", &Payload::t_src_ns, offsetof(Payload, t_src_ns)),
      pod_field("packets", &Payload::packets, offsetof(Payload, packets)),
      pod_field("callback_start_ns", &Payload::callback_start_ns,
                offsetof(Payload, callback_start_ns)),
      pod_field("target_produced_ns", &Payload::target_produced_ns,
                offsetof(Payload, target_produced_ns)),
      pod_field("target_use_ns", &Payload::target_use_ns, offsetof(Payload, target_use_ns)),
      pod_field("target_version", &Payload::target_version, offsetof(Payload, target_version)),
      pod_field("safe_q", &Payload::safe_q, offsetof(Payload, safe_q)),
      pod_field("safety_non_finite", &Payload::safety_non_finite,
                offsetof(Payload, safety_non_finite)),
      pod_field("safety_range_clamps", &Payload::safety_range_clamps,
                offsetof(Payload, safety_range_clamps)),
      pod_field("safety_step_clamps", &Payload::safety_step_clamps,
                offsetof(Payload, safety_step_clamps)),
      pod_field("safety_recovery_steps", &Payload::safety_recovery_steps,
                offsetof(Payload, safety_recovery_steps)));
};

}  // namespace holoscan
