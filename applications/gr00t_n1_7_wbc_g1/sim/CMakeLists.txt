# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/cuda)
add_executable(sim_ucx_gr00t_n1_7_wbc_g1 main.cc)
set_target_properties(sim_ucx_gr00t_n1_7_wbc_g1 PROPERTIES UNITY_BUILD OFF)
target_include_directories(sim_ucx_gr00t_n1_7_wbc_g1 PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(sim_ucx_gr00t_n1_7_wbc_g1 PRIVATE
  CLI11::CLI11
  CUDA::cudart
  holoscan::config_io
  holoscan::core
  holoscan::params
  holoscan::visualization::op_holoviz
  isaac::applications::gr00t_n1_7_wbc_g1::green_context
  isaac::applications::gr00t_n1_7_wbc_g1::shutdown_signals
  isaac::operators::agile_wbc
  isaac::operators::gr00t_action_handler
  isaac::operators::gr00t_n1_7
  isaac::operators::joint_command_combiner
  isaac::operators::start_up
  isaac::operators::teleop_twist_command
  isaac::operators::unitree_sim_ucx
)
add_dependencies(sim_ucx_gr00t_n1_7_wbc_g1
  gr00t_n1_7_g1_static_apple_sim_models
  nvidia_wbc_agile_velocity_unitree_g1
)
if(ISAAC_BUILD_TESTS)
  add_test(NAME test_sim_ucx_gr00t_n1_7_wbc_g1_compile COMMAND sim_ucx_gr00t_n1_7_wbc_g1
    --application-config-path=${PROJECT_SOURCE_DIR}/applications/gr00t_n1_7_wbc_g1/config/control.yaml
    --model-config-path=${PROJECT_SOURCE_DIR}/operators/gr00t_n1_7/config/n1_7_g1_apple_sim_trt.yaml
    --compile-only)
  set_tests_properties(test_sim_ucx_gr00t_n1_7_wbc_g1_compile PROPERTIES TIMEOUT 60
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  add_test(NAME test_sim_ucx_gr00t_n1_7_wbc_g1_holoviz_compile COMMAND sim_ucx_gr00t_n1_7_wbc_g1
    --application-config-path=${PROJECT_SOURCE_DIR}/applications/gr00t_n1_7_wbc_g1/config/control.yaml
    --model-config-path=${PROJECT_SOURCE_DIR}/operators/gr00t_n1_7/config/n1_7_g1_apple_sim_trt.yaml
    --compile-only --holoviz)
  set_tests_properties(test_sim_ucx_gr00t_n1_7_wbc_g1_holoviz_compile PROPERTIES TIMEOUT 60
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  foreach(display IN ITEMS headless holoviz)
    set(display_args)
    if(display STREQUAL "holoviz")
      list(APPEND display_args --holoviz)
    endif()
    add_test(NAME test_sim_ucx_gr00t_n1_7_wbc_g1_gamepad_${display}_compile COMMAND sim_ucx_gr00t_n1_7_wbc_g1
      --application-config-path=${PROJECT_SOURCE_DIR}/applications/gr00t_n1_7_wbc_g1/config/control.yaml
      --model-config-path=${PROJECT_SOURCE_DIR}/operators/gr00t_n1_7/config/n1_7_g1_apple_sim_trt.yaml
      --compile-only --gamepad ${display_args})
    set_tests_properties(test_sim_ucx_gr00t_n1_7_wbc_g1_gamepad_${display}_compile PROPERTIES
      TIMEOUT 60 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  endforeach()
endif()
