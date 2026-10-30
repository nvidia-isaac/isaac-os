/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_IMAGE_CONVERSION_NV12_TO_RGB_OP_H
#define ISAAC_OS_OPERATORS_IMAGE_CONVERSION_NV12_TO_RGB_OP_H

#include <cstdint>
#include <memory>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace isaac::os::ops {

namespace detail {
class NppContext;
}

/// Converts a two-plane, full-range BT.709 NV12 image to packed RGB8 on the CUDA device.
class Nv12ToRgbOp final : public holoscan::Operator<> {
 public:
  Nv12ToRgbOp(std::uint32_t max_width, std::uint32_t max_height);
  ~Nv12ToRgbOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::schema::ImageT> image;
  holoscan::Output<holoscan::schema::ImageT> rgb_image;

 private:
  std::uint32_t max_width_;
  std::uint32_t max_height_;
  std::unique_ptr<detail::NppContext> npp_context_;
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_IMAGE_CONVERSION_NV12_TO_RGB_OP_H
