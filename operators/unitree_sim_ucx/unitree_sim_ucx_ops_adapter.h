/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "joint_state_cpu_schema_traits.h"

namespace isaac::os::ops {

/// Publish a matched simulator step to all control consumers, including GR00T and startup.
/// WBC also aligns its inputs; keep this shared boundary on its lane to preserve delivery order.
class UnitreeSimStateOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override;

  holoscan::Input<isaac::JointStateCpuT> joint_state_input;
  holoscan::Input<holoscan::schema::ImuT> imu_input;
  holoscan::Output<isaac::JointStateCpuT> joint_state;
  holoscan::Output<holoscan::schema::ImuT> imu;
};

/// Expose SDK-owned image pixels to tensor consumers without copying them.
class UnitreeSimImageToTensorOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override;

  holoscan::Input<holoscan::schema::ImageT> input;
  holoscan::Output<holoscan::Tensor> output;
};

}  // namespace isaac::os::ops
