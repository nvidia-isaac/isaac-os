# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/cuda)
add_executable(sim_ucx_wbc_g1 main.cc)
set_target_properties(sim_ucx_wbc_g1 PROPERTIES UNITY_BUILD OFF)
target_include_directories(sim_ucx_wbc_g1 PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(sim_ucx_wbc_g1 PRIVATE
  CLI11::CLI11
  CUDA::cudart
  holoscan::config_io
  holoscan::core
  holoscan::params
  holoscan::visualization::op_holoviz
  isaac::operators::agile_wbc
  isaac::operators::message_drop_sink
  isaac::operators::teleop_twist_command
  isaac::operators::unitree_sim_ucx
)
add_dependencies(sim_ucx_wbc_g1 nvidia_wbc_agile_velocity_unitree_g1)
if(ISAAC_BUILD_TESTS)
  add_test(NAME test_sim_ucx_wbc_g1_compile COMMAND sim_ucx_wbc_g1
    --config-path=${PROJECT_SOURCE_DIR}/applications/wbc_g1/config/wbc_g1_sim_ucx_config.yaml
    --compile-only)
  set_tests_properties(test_sim_ucx_wbc_g1_compile PROPERTIES TIMEOUT 60)
  add_test(NAME test_sim_ucx_wbc_g1_config_override COMMAND sim_ucx_wbc_g1
    --config-path=${CMAKE_CURRENT_BINARY_DIR}/unused-default.yaml
    --config-path=${PROJECT_SOURCE_DIR}/applications/wbc_g1/config/wbc_g1_sim_ucx_config.yaml
    --compile-only)
  set_tests_properties(test_sim_ucx_wbc_g1_config_override PROPERTIES TIMEOUT 60)
  add_test(NAME test_sim_ucx_wbc_g1_holoviz_compile COMMAND sim_ucx_wbc_g1
    --config-path=${PROJECT_SOURCE_DIR}/applications/wbc_g1/config/wbc_g1_sim_ucx_config.yaml
    --compile-only --holoviz)
  set_tests_properties(test_sim_ucx_wbc_g1_holoviz_compile PROPERTIES TIMEOUT 60)
  foreach(display IN ITEMS headless holoviz)
    set(display_args)
    if(display STREQUAL "holoviz")
      list(APPEND display_args --holoviz)
    endif()
    add_test(NAME test_sim_ucx_wbc_g1_gamepad_${display}_compile COMMAND sim_ucx_wbc_g1
      --config-path=${CMAKE_SOURCE_DIR}/applications/wbc_g1/config/wbc_g1_sim_ucx_config.yaml
      --compile-only --gamepad ${display_args})
    set_tests_properties(test_sim_ucx_wbc_g1_gamepad_${display}_compile PROPERTIES
      TIMEOUT 60 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endforeach()
endif()
