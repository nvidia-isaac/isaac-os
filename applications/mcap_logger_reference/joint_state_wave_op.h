/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "joint_state_cpu_schema_traits.h"

namespace isaac::os::mcap_logger_reference {

enum class Waveform { kSine, kCosine };

// Emits a single-joint isaac::JointStateCpuT every tick, tracing out one full period of a sine
// or cosine wave (position) with its analytically-correct velocity (the position's time
// derivative), driven purely by holoscan::OnClock -- no external trigger/driver is needed.
//
// Elapsed time is computed as tick_count * kPeriod (not a wall-clock read) so the waveform is
// exactly reproducible regardless of scheduling jitter -- the same run, or two different runs,
// produce bit-identical position/velocity sequences.
class JointStateWaveOp final : public holoscan::Operator<> {
 public:
  static constexpr auto kPeriod = std::chrono::milliseconds(50);  // 20 Hz sample rate.
  static constexpr float kAmplitude = 1.0F;                       // Peak position, radians.
  static constexpr float kAngularFrequency =
      static_cast<float>(std::numbers::pi);  // rad/s (0.5 Hz wave).

  explicit JointStateWaveOp(Waveform waveform) : waveform_(waveform) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(joint_state, "joint_state"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = kPeriod});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    const float elapsed_s =
        static_cast<float>(tick_count_) * std::chrono::duration<float>(kPeriod).count();
    const float angle = kAngularFrequency * elapsed_s;
    const bool sine = waveform_ == Waveform::kSine;
    const float position = kAmplitude * (sine ? std::sin(angle) : std::cos(angle));
    const float velocity =
        kAmplitude * kAngularFrequency * (sine ? std::cos(angle) : -std::sin(angle));
    ++tick_count_;

    isaac::JointStateCpuT value;
    value.position = {position};
    value.velocity = {velocity};

    // Deliberately emit(value) with no EmitOptions/capture_time: an explicit
    // EmitOptions{.capture_time = context.clock().now()} (the pattern
    // operators/mock/mock_joint_state_op.cc uses successfully for an in-process, add_flow-
    // connected output) fails every tick with a message-less ErrorCode::kInvalidArgument once this
    // port is exported/attached (as HsdkMcapLogger::attach() does here) rather than add_flow-
    // connected -- reproduced directly while building this app. Root cause not further isolated
    // (closed-source runtime past holoscan::Output<T>::emit()'s header); plain emit(value) avoids
    // it entirely.
    return joint_state.emit(value);
  }

  holoscan::Output<isaac::JointStateCpuT> joint_state;

 private:
  Waveform waveform_;
  std::uint64_t tick_count_ = 0;
};

}  // namespace isaac::os::mcap_logger_reference
