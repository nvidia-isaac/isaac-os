/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_AGILE_WBC_AGILE_WBC_OP_H
#define ISAAC_OS_OPERATORS_AGILE_WBC_AGILE_WBC_OP_H

// MatX 0.9.4 assumes these tuple declarations are transitively visible.
// clang-format off
#include <thrust/tuple.h>
#include <matx.h>
// clang-format on

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "common/tensorrt_engine/tensorrt_engine.h"
#include "joint_command_cpu_schema_traits.h"
#include "joint_state_cpu_schema_traits.h"
#include "operators/agile_wbc/wbc_params.h"
#include "twist_cpu_schema_traits.h"

namespace isaac::os::ops {

class AgileWbcOp final : public holoscan::Operator<WbcParams> {
 public:
  AgileWbcOp() = default;
  /// Registers the optional teleop velocity input.
  explicit AgileWbcOp(bool velocity_from_teleop) : velocity_from_teleop_(velocity_from_teleop) {}

  void setup(holoscan::OperatorSpec& spec) override;
  holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<isaac::JointStateCpuT> joint_state;
  holoscan::Input<holoscan::schema::ImuT> imu;
  /// Uses the configured standing velocity until the first command arrives.
  holoscan::Input<isaac::TwistCpuT> velocity_commands;
  holoscan::Output<isaac::JointCommandCpuT> command;

 private:
  bool velocity_from_teleop_{};

  std::unique_ptr<isaac::TensorRTEngine> tensorrt_engine_;

  cudaGraph_t graph_{};
  cudaGraphExec_t graph_exec_{};
  cudaEvent_t completion_event_{};

  int n_sdk_joints_{};

  matx::tensor_t<float, 1> t_quat_w_;
  matx::tensor_t<float, 1> t_ang_vel_b_;
  matx::tensor_t<float, 1> t_vel_cmds_;
  matx::tensor_t<float, 1> t_joint_pos_raw_;
  matx::tensor_t<float, 1> t_joint_vel_raw_;
  matx::tensor_t<float, 1> t_joint_pos_;
  matx::tensor_t<float, 1> t_joint_vel_;
  matx::tensor_t<int, 1> idx_sdk_to_policy_;

  std::unordered_map<std::string, matx::tensor_t<float, 1>> history_tensors_;
  std::unordered_map<std::string, matx::tensor_t<float, 1>> trt_outputs_;

  float h_ang_vel_scaled_[3]{};

  matx::tensor_t<float, 1> t_src_pos_;
  matx::tensor_t<float, 1> t_src_kp_;
  matx::tensor_t<float, 1> t_src_kd_;
  matx::tensor_t<float, 1> t_out_pos_;
  matx::tensor_t<float, 1> t_out_kp_;
  matx::tensor_t<float, 1> t_out_kd_;
  matx::tensor_t<int, 1> idx_sdk_src_;
  matx::tensor_t<float, 1> t_cmd_pos_host_;
  matx::tensor_t<float, 1> t_cmd_kp_host_;
  matx::tensor_t<float, 1> t_cmd_kd_host_;
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_AGILE_WBC_AGILE_WBC_OP_H
