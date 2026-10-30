/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_UNITREE_SIM_UCX_SYNTHETIC_CLOCK_ADVANCE_OP_H
#define ISAAC_OS_OPERATORS_UNITREE_SIM_UCX_SYNTHETIC_CLOCK_ADVANCE_OP_H

#include <memory>
#include <utility>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/schemas/time_generated.h>
#include <holoscan/time/synthetic_clock.hpp>

namespace isaac::os::ops {

// Explicit shared clock state for graph-owned operator construction.
struct SyntheticClockState {
  std::shared_ptr<holoscan::SyntheticClock> clock;
};

class SyntheticClockAdvanceOp final : public holoscan::Operator<> {
 public:
  explicit SyntheticClockAdvanceOp(SyntheticClockState state) : clock_(std::move(state.clock)) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::TimeT> input;

 private:
  std::shared_ptr<holoscan::SyntheticClock> clock_;
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_UNITREE_SIM_UCX_SYNTHETIC_CLOCK_ADVANCE_OP_H
