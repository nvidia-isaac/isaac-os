/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/message_drop_sink/message_drop_sink_op.h"

#include <utility>

#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {

void MessageDropSinkOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(in_message, "in_message")
      .expects_tensor(holoscan::TensorInputSpec{
          .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice}});
}

holoscan::Contract MessageDropSinkOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{in_message});
  return contract;
}

holoscan::expected<void, holoscan::Error> MessageDropSinkOp::compute(
    holoscan::ExecutionContext& /*context*/) {
  auto received = in_message.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  ++messages_received_;
  if (!logged_once_) {
    logged_once_ = true;
    HOLOSCAN_LOG_INFO("MessageDropSinkOp: receiving and discarding all messages.");
  }
  return {};
}

}  // namespace isaac::os::ops
