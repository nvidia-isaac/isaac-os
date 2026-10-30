/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_COMMON_SO101_DRIVER_SO101_DRIVER_CONFIG_H
#define ISAAC_OS_COMMON_SO101_DRIVER_SO101_DRIVER_CONFIG_H

#include <holoscan/config/document.hpp>

#include "common/so101_driver/feetech_serial_driver.h"

namespace isaac::os {

[[nodiscard]] so101::FeetechDriverOptions decode_so101_driver_options(
    const holoscan::config::Document& document);

}  // namespace isaac::os

#endif  // ISAAC_OS_COMMON_SO101_DRIVER_SO101_DRIVER_CONFIG_H
