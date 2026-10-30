/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_APPLICATIONS_SO101_SINUS_MOTION_SIM_SINUS_MOTION_OP_H
#define ISAAC_OS_APPLICATIONS_SO101_SINUS_MOTION_SIM_SINUS_MOTION_OP_H

#include <cstddef>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "joint_command_cpu_schema_traits.h"
#include "joint_state_cpu_schema_traits.h"

namespace isaac::os::apps::so101_sinus_motion_sim {

// CPU-array sinus generator for the UCX/sim path. Tensor wire conversion is
// performed by the transport, before/after these graph ports.
struct SinusMotionParams {
  double sin_frequency{0.25};  // Hz
  double sin_amplitude{0.20};  // radians
  int num_joints{6};
  // Preserve YAML numeric precision until constructing the float command payload.
  std::vector<double> default_position{};
  std::vector<double> kp{};
  std::vector<double> kd{};
};

class SinusMotionOp final : public holoscan::Operator<> {
 public:
  explicit SinusMotionOp(SinusMotionParams params);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::JointStateCpuT> joint_state_in;
  holoscan::Output<isaac::JointCommandCpuT> joint_command_out;

 private:
  SinusMotionParams params_;
  double start_time_seconds_{-1.0};
};

}  // namespace isaac::os::apps::so101_sinus_motion_sim

#endif  // ISAAC_OS_APPLICATIONS_SO101_SINUS_MOTION_SIM_SINUS_MOTION_OP_H
