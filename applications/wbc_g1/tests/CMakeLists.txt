# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

add_executable(test_sim_state_pairing test_sim_state_pairing.cc)
target_link_libraries(test_sim_state_pairing PRIVATE
  GTest::gtest_main
  holoscan::core
  isaac::operators::unitree_sim_ucx
)
add_test(NAME test_sim_state_pairing COMMAND test_sim_state_pairing)
set_tests_properties(test_sim_state_pairing PROPERTIES TIMEOUT 20 LABELS "applications;cpp")

add_executable(test_wbc_params_config test_wbc_params_config.cc)
target_compile_definitions(test_wbc_params_config PRIVATE
  "ISAAC_WBC_CANONICAL_CONFIG=\"${PROJECT_SOURCE_DIR}/applications/wbc_g1/config/wbc_g1_config.yaml\""
)
target_include_directories(test_wbc_params_config PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_wbc_params_config
  PRIVATE
    GTest::gtest_main
    holoscan::config_io
    holoscan::params
)

add_test(NAME test_wbc_params_config COMMAND test_wbc_params_config)
set_tests_properties(test_wbc_params_config PROPERTIES
  LABELS "applications;cpp"
)
