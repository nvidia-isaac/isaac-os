/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_G1_LOW_LEVEL_G1_PARAMS_H
#define ISAAC_OS_OPERATORS_G1_LOW_LEVEL_G1_PARAMS_H

#include <cstdint>
#include <string>

#include <holoscan/core/parameter_schema.hpp>

namespace isaac::os::ops {

// Runtime configuration for the G1 low-level DDS operators. The subscriber and the publisher must
// agree on the joint count and the DDS endpoint, so both read the same aggregate; applications
// remain responsible for mapping their configuration format onto it.
struct G1DdsParams {
  std::string network_interface{"eth0"};
  // 29 for the G1 body alone, 43 for the body plus Dex 3-1 hands.
  std::int64_t g1_num_motors{29};
  // 0 addresses the real robot, 1 a simulator.
  std::int64_t dds_domain_id{};
  // Control mode echoed back to the robot until LowState reports the live value.
  std::int64_t mode_machine{6};
};

HOLOSCAN_PARAM_SCHEMA(G1DdsParams, holoscan::param_schema("com.nvidia.isaac.G1DdsParams", 1),
                      holoscan::field<&G1DdsParams::network_interface>("network_interface"),
                      holoscan::field<&G1DdsParams::g1_num_motors>("g1_num_motors"),
                      holoscan::field<&G1DdsParams::dds_domain_id>("dds_domain_id"),
                      holoscan::field<&G1DdsParams::mode_machine>("mode_machine"));

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_G1_LOW_LEVEL_G1_PARAMS_H
