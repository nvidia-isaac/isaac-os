/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_SINUS_MOTION_SINUS_MOTION_OP_H
#define ISAAC_OS_OPERATORS_SINUS_MOTION_SINUS_MOTION_OP_H

#include <cstdint>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "joint_command_cpu_schema_traits.h"
#include "state_sample_cpu_schema_traits.h"

namespace isaac::os::ops {

/// Shape of the sine that `SinusMotionOp` writes to every joint.
struct SinusMotionParams {
  /// Position each joint oscillates about, in radians. Needs exactly `num_joints` entries.
  std::vector<double> default_pos{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  /// Joint count. State samples of any other size are ignored rather than partially commanded.
  int num_joints{6};
  /// Peak deviation from `default_pos`, in radians.
  double sin_amplitude{0.20};
  /// Sine frequency in Hz. The default completes one oscillation every four seconds.
  double sin_frequency{0.25};
};

HOLOSCAN_PARAM_SCHEMA(SinusMotionParams,
                      holoscan::param_schema("com.nvidia.isaac.SinusMotionParams", 1),
                      holoscan::field<&SinusMotionParams::default_pos>("default_pos"),
                      holoscan::field<&SinusMotionParams::num_joints>("num_joints"),
                      holoscan::field<&SinusMotionParams::sin_amplitude>("sin_amplitude"),
                      holoscan::field<&SinusMotionParams::sin_frequency>("sin_frequency"));

/// Reports whether the joint count is positive, `default_pos` holds exactly that many finite
/// entries, and the amplitude and frequency are finite and non-negative.
[[nodiscard]] bool validate_sinus_motion_params(const SinusMotionParams& params);

/// Emits a sine-wave joint target on each incoming joint-state sample.
class SinusMotionOp final : public holoscan::Operator<SinusMotionParams> {
 public:
  /// Declares the joint-state input and the joint-command output.
  void setup(holoscan::OperatorSpec& spec) override;
  /// Triggers on every joint-state sample, so targets are generated at the state rate rather
  /// than from an independent timer.
  [[nodiscard]] holoscan::Contract contract() const override;
  /// Throws `std::runtime_error` if the configured parameters are invalid.
  void start() override;
  /// Writes `default_pos + amplitude * sin(2 * pi * frequency * t)` to every joint, with `t`
  /// measured from the first sample seen. Samples whose joint count disagrees with
  /// `num_joints` are dropped without emitting.
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  /// Measured joint state. Its arrival paces the operator, and its capture time and trace id
  /// are forwarded to the emitted command.
  holoscan::Input<isaac::StateSampleCpuT> state;
  /// Sine target for the downstream motion planner.
  holoscan::Output<isaac::JointCommandCpuT> command;

 private:
  std::int64_t start_time_ns_{-1};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_SINUS_MOTION_SINUS_MOTION_OP_H
