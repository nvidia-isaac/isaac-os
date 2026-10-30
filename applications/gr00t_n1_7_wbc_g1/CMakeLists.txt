# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_add_resource)
include(isaac_deps/holoscan)

set(AGILE_WBC_BASE_URL "https://github.com/nvidia-isaac/WBC-AGILE/raw/v1.2/agile/data/policy")
if(NOT TARGET nvidia_wbc_agile_velocity_unitree_g1)
  isaac_add_resource(nvidia_wbc_agile_velocity_unitree_g1
    URL "${AGILE_WBC_BASE_URL}/velocity_g1/unitree_g1_velocity_e2e.onnx"
    EXPECTED_HASH SHA256=8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b
  )
endif()

add_library(gr00t_wbc_green_context SHARED green_context.cc)
add_library(
  isaac::applications::gr00t_n1_7_wbc_g1::green_context
  ALIAS gr00t_wbc_green_context
)
target_include_directories(gr00t_wbc_green_context PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gr00t_wbc_green_context PRIVATE holoscan::config holoscan::core)

add_library(gr00t_wbc_shutdown_signals SHARED shutdown_signals.cc)
add_library(
  isaac::applications::gr00t_n1_7_wbc_g1::shutdown_signals
  ALIAS gr00t_wbc_shutdown_signals
)
find_package(Threads REQUIRED)
target_include_directories(gr00t_wbc_shutdown_signals PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gr00t_wbc_shutdown_signals PRIVATE Threads::Threads holoscan::core)

add_subdirectory(mock)
add_subdirectory(real)
add_subdirectory(sim)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
