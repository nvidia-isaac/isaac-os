/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "applications/sipl_camera/sipl_metadata_sink_op.h"

#include <utility>

namespace isaac::os::ops {

void SiplMetadataSinkOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(input, "input").queue_depth(1U);
}

holoscan::Contract SiplMetadataSinkOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{input});
  return result;
}

holoscan::expected<void, holoscan::Error> SiplMetadataSinkOp::compute(holoscan::ExecutionContext&) {
  auto sample = input.receive();
  if (!sample) return holoscan::make_unexpected(std::move(sample).error());
  return {};
}

}  // namespace isaac::os::ops
