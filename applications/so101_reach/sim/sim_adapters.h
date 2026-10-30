/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_APPLICATIONS_SO101_REACH_SIM_SIM_ADAPTERS_H
#define ISAAC_OS_APPLICATIONS_SO101_REACH_SIM_SIM_ADAPTERS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>

#include "common/so101_driver/so101_types.h"
#include "joint_state_cpu_schema_traits.h"
#include "state_sample_cpu_schema_traits.h"

namespace isaac::os::ops {

inline isaac::StateSampleCpuT make_so101_sim_state(isaac::JointStateCpuT input,
                                                   std::uint64_t sequence) {
  const auto valid = [](const auto& values) {
    return values.size() == 6U &&
           std::ranges::all_of(values, [](float value) { return std::isfinite(value); });
  };
  if (!valid(input.position) || !valid(input.velocity)) {
    throw std::invalid_argument(
        "Simulated SO-101 state must have six finite positions and velocities.");
  }
  if (!input.name.empty() && !std::ranges::equal(input.name, so101::kSo101JointNames)) {
    throw std::invalid_argument(
        "Simulated SO-101 joints must use shoulder_pan, shoulder_lift, "
        "elbow_flex, wrist_flex, wrist_roll, gripper order.");
  }
  isaac::StateSampleCpuT result;
  result.position = std::move(input.position);
  result.velocity = std::move(input.velocity);
  result.source_sequence = sequence;
  return result;
}

class So101SimStateOp final : public holoscan::Operator<> {
 public:
  explicit So101SimStateOp(std::size_t message_count) : message_count_(message_count) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(joints, "joints");
    spec.output(state, "state");
  }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{joints});
    return result;
  }
  void stop() override {
    if (sequence_ == 0U) {
      HOLOSCAN_LOG_WARN("No simulated SO-101 joint states were received.");
    } else if (last_frame_id_ == 0U) {
      HOLOSCAN_LOG_INFO("Received {} simulated SO-101 joint states.", sequence_);
    } else {
      HOLOSCAN_LOG_INFO(
          "Received {} simulated SO-101 joint states; {} repeated the previous physics step.",
          sequence_, repeated_);
    }
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto input = joints.receive();
    if (!input) return holoscan::make_unexpected(std::move(input).error());
    // The simulator stamps each physics step, so an unchanged frame ID repeats the last step.
    if (const std::uint64_t frame_id = input->metadata.frame_id; frame_id != 0U) {
      if (frame_id == last_frame_id_) ++repeated_;
      last_frame_id_ = frame_id;
    }
    isaac::StateSampleCpuT sample;
    try {
      sample = make_so101_sim_state(std::move(input->data), ++sequence_);
    } catch (const std::invalid_argument& error) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument, error.what()});
    }
    auto emitted = state.emit(std::move(sample), {.capture_time = input->metadata.capture_time()});
    if (!emitted) return emitted;
    if (message_count_ != 0U && sequence_ >= message_count_) context.request_completion();
    return {};
  }
  holoscan::Input<isaac::JointStateCpuT> joints;
  holoscan::Output<isaac::StateSampleCpuT> state;

 private:
  std::size_t message_count_;
  std::uint64_t sequence_{};
  std::uint64_t repeated_{};
  std::uint64_t last_frame_id_{};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_APPLICATIONS_SO101_REACH_SIM_SIM_ADAPTERS_H
