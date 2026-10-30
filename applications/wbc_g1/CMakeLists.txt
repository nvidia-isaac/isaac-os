# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_add_resource)
include(isaac_deps/holoscan)

set(AGILE_WBC_BASE_URL "https://github.com/nvidia-isaac/WBC-AGILE/raw/v1.2/agile/data/policy")
isaac_add_resource(nvidia_wbc_agile_velocity_unitree_g1
  URL "${AGILE_WBC_BASE_URL}/velocity_g1/unitree_g1_velocity_e2e.onnx"
  EXPECTED_HASH SHA256=8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b
)

add_subdirectory(mock)
add_subdirectory(real)
add_subdirectory(sim)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
