/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/unitree_sim_ucx/unitree_sim_ucx_ops_adapter.h"

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

namespace isaac::os::ops {

void UnitreeSimStateOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(joint_state_input, "in_joint_state");
  spec.input(imu_input, "in_imu");
  spec.output(joint_state, "out_joint_state");
  spec.output(imu, "out_imu");
}

holoscan::Contract UnitreeSimStateOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnAligned{.ports = holoscan::ports(joint_state_input, imu_input),
                                     .window = std::chrono::nanoseconds{0}});
  result.input(joint_state_input, holoscan::Exact{});
  result.input(imu_input, holoscan::Exact{});
  return result;
}

holoscan::expected<void, holoscan::Error> UnitreeSimStateOp::compute(holoscan::ExecutionContext&) {
  auto state = joint_state_input.receive();
  if (!state) return holoscan::make_unexpected(std::move(state).error());
  auto inertial = imu_input.receive();
  if (!inertial) return holoscan::make_unexpected(std::move(inertial).error());
  auto emitted = imu.emit(
      std::move(inertial->data),
      {.capture_time = inertial->metadata.capture_time(), .flags = inertial->metadata.flags});
  if (!emitted) return emitted;
  return joint_state.emit(std::move(state->data), {.capture_time = state->metadata.capture_time(),
                                                   .flags = state->metadata.flags});
}

void UnitreeSimImageToTensorOp::setup(holoscan::OperatorSpec& spec) {
  const holoscan::TensorRepresentation representation{
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 3U};
  spec.input(input, "input")
      .expects_tensor(holoscan::TensorInputSpec{
          .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice}});
  spec.output(output, "output")
      // The SDK requires a bounded output resource to route downstream CUDA streams.
      // emit() forwards the existing SDK-owned backing; compute() does not allocate a loan.
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = representation,
          .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = 1920U * 1080U * 3U},
          .storage = holoscan::TensorOutputStorage::kRuntimePool});
}

holoscan::Contract UnitreeSimImageToTensorOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{input});
  return result;
}

holoscan::expected<void, holoscan::Error> UnitreeSimImageToTensorOp::compute(
    holoscan::ExecutionContext&) {
  auto received = input.receive();
  if (!received) return holoscan::make_unexpected(std::move(received).error());
  const auto& image = received->data;
  if (!image.data || image.encoding != holoscan::schema::ImageEncoding_RGB8 || image.width <= 0 ||
      image.width > 1920 || image.height <= 0 || image.height > 1080 ||
      image.data->dtype().code != kDLUInt || image.data->dtype().bits != 8U ||
      image.data->dtype().lanes != 1U ||
      image.data->shape() != std::vector<std::int64_t>{image.height, image.width, 3} ||
      !image.data->is_contiguous()) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                        "The simulator camera frame must be a contiguous RGB8 image within the "
                        "configured bounds."});
  }
  return output.emit(*image.data, {.capture_time = received->metadata.capture_time(),
                                   .flags = received->metadata.flags});
}

}  // namespace isaac::os::ops
