/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/unitree_sim_ucx/synthetic_clock_advance_op.h"

#include <string>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {

void SyntheticClockAdvanceOp::setup(holoscan::OperatorSpec& spec) { spec.input(input, "input"); }

holoscan::Contract SyntheticClockAdvanceOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{input});
  return contract;
}

holoscan::expected<void, holoscan::Error> SyntheticClockAdvanceOp::compute(
    [[maybe_unused]] holoscan::ExecutionContext& context) {
  auto received = input.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  if (!clock_) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "A synthetic clock is required."});
  }
  const auto current_time_ns = clock_->timestamp();
  if (received->data.time_ns < current_time_ns) {
    const auto message = "The received simulation timestamp (" +
                         std::to_string(received->data.time_ns) +
                         " ns) is older than the current synthetic time (" +
                         std::to_string(current_time_ns) + " ns).";
    HOLOSCAN_LOG_ERROR("{}", message);
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, message});
  }
  clock_->advance_to(received->data.time_ns);
  return {};
}

}  // namespace isaac::os::ops
