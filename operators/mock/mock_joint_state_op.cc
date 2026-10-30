/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/mock/mock_joint_state_op.h"

#include <chrono>
#include <utility>

namespace isaac::os::ops {

using namespace std::chrono_literals;

void MockJointStateOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(joint_state, "out_joint_state");
  spec.output(imu, "out_imu");
}

holoscan::Contract MockJointStateOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnClock{.period = 20ms});
  if (compute_call_limit_) {
    contract.complete_after(holoscan::ComputeCalls{.limit = *compute_call_limit_});
  }
  return contract;
}

void MockJointStateOp::start() {
  joint_state_message_.position.resize(joint_count_);
  joint_state_message_.velocity.resize(joint_count_);
  for (std::size_t index = 0; index < joint_count_; ++index) {
    joint_state_message_.position[index] = static_cast<float>(index + 1U) * 0.01F;
    joint_state_message_.velocity[index] = -static_cast<float>(index + 1U) * 0.02F;
  }
  imu_message_.orientation = holoscan::schema::Quaternionf{1.0F, 0.0F, 0.0F, 0.0F};
  imu_message_.angular_velocity = holoscan::schema::Vector3f{0.0F, 0.0F, 0.0F};
  imu_message_.linear_acceleration = holoscan::schema::Vector3f{0.0F, 0.0F, 0.0F};
}

holoscan::expected<void, holoscan::Error> MockJointStateOp::compute(
    holoscan::ExecutionContext& context) {
  auto capture_time = context.clock().now();
  if (!capture_time) {
    return holoscan::make_unexpected(std::move(capture_time).error());
  }
  const holoscan::EmitOptions options{.capture_time = *capture_time};
  // Publish matching IMU evidence before joint state triggers WBC.
  auto emitted = imu.emit(imu_message_, options);
  if (!emitted) {
    return emitted;
  }
  return joint_state.emit(joint_state_message_, options);
}

}  // namespace isaac::os::ops
