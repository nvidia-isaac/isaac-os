/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_MESSAGE_DROP_SINK_MESSAGE_DROP_SINK_OP_H
#define ISAAC_OS_OPERATORS_MESSAGE_DROP_SINK_MESSAGE_DROP_SINK_OP_H

#include <cstdint>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace isaac::os::ops {

// Terminates an ImageT message stream by receiving and discarding each
// frame without inspecting or forwarding it. Wired in for headless
// runs where camera frames still arrive over UCX (Sim always publishes
// them) but the app has no HolovizOp / MCAP writer to consume them.
//
// Templated on payload type would be more general, but each operator template requires its own
// port-payload concept satisfaction,
// so ImageT-specific is the pragmatic Tier-1 shape.
class MessageDropSinkOp final : public holoscan::Operator<> {
 public:
  MessageDropSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::ImageT> in_message;

 private:
  std::uint64_t messages_received_{0U};
  bool logged_once_{false};
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_MESSAGE_DROP_SINK_MESSAGE_DROP_SINK_OP_H
