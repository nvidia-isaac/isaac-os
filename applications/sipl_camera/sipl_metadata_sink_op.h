/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_APPLICATIONS_SIPL_CAMERA_SIPL_METADATA_SINK_OP_H
#define ISAAC_OS_APPLICATIONS_SIPL_CAMERA_SIPL_METADATA_SINK_OP_H

#include <sipl_capture_op/sipl_capture_op.hpp>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace isaac::os::ops {

/// Consumes SIPL frame metadata that this display-only sample does not otherwise use.
class SiplMetadataSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override;

  holoscan::Input<holoscan::camera::SIPLFrameMetadataT> input;
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_APPLICATIONS_SIPL_CAMERA_SIPL_METADATA_SINK_OP_H
