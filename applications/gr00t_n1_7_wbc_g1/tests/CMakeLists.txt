# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_gr00t_wbc_config test_gr00t_wbc_config.cc)
include(isaac_deps/cuda)
include(isaac_deps/gtest)
add_executable(test_mock_state_alignment
  ../mock/mock_operators.cc
  test_mock_state_alignment.cc
)
target_include_directories(test_mock_state_alignment PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_mock_state_alignment PRIVATE
  CUDA::cudart
  GTest::gtest_main
  holoscan::core
  holoscan::sensor_io
  isaac::common::joint_cpu_messages
)
add_test(NAME test_mock_state_alignment COMMAND test_mock_state_alignment)
set_tests_properties(test_mock_state_alignment PROPERTIES
  LABELS "applications;cpp"
  TIMEOUT 15
)
add_executable(test_unitree_sim_ucx_adapter test_unitree_sim_ucx_adapter.cc)
target_link_libraries(test_unitree_sim_ucx_adapter PRIVATE
  CUDA::cudart
  GTest::gtest_main
  holoscan::core
  isaac::operators::unitree_sim_ucx
)
add_test(NAME test_unitree_sim_ucx_adapter COMMAND test_unitree_sim_ucx_adapter)
set_tests_properties(test_unitree_sim_ucx_adapter PROPERTIES
  LABELS "applications;cpp;gpu"
  TIMEOUT 30
)
target_compile_definitions(test_gr00t_wbc_config PRIVATE
  "ISAAC_GR00T_APPLICATION_CONFIG=\"${PROJECT_SOURCE_DIR}/applications/gr00t_n1_7_wbc_g1/config/control.yaml\""
  "ISAAC_GR00T_MODEL_CONFIG=\"${PROJECT_SOURCE_DIR}/operators/gr00t_n1_7/config/n1_7_g1.yaml\""
)
target_link_libraries(test_gr00t_wbc_config
  PRIVATE
    GTest::gtest_main
    holoscan::config_io
    holoscan::core
    holoscan::params
    holoscan::sensor_io
    isaac::applications::gr00t_n1_7_wbc_g1::green_context
    isaac::operators::agile_wbc
    isaac::operators::gr00t_action_handler
    isaac::operators::gr00t_n1_7
    isaac::operators::joint_command_combiner
)
add_test(NAME test_gr00t_wbc_config COMMAND test_gr00t_wbc_config)
set_tests_properties(test_gr00t_wbc_config PROPERTIES
  LABELS "applications;cpp"
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_executable(test_gr00t_wbc_real_config test_real_config.cc)
target_compile_definitions(test_gr00t_wbc_real_config PRIVATE
  "ISAAC_GR00T_WBC_CONFIG=\"${PROJECT_SOURCE_DIR}/applications/gr00t_n1_7_wbc_g1/config/control.yaml\""
  "ISAAC_GR00T_MODEL_CONFIG=\"${PROJECT_SOURCE_DIR}/operators/gr00t_n1_7/config/n1_7_g1.yaml\""
)
target_link_libraries(test_gr00t_wbc_real_config
  PRIVATE
    GTest::gtest_main
    holoscan::config
    holoscan::config_io
    holoscan::params
    isaac::applications::gr00t_n1_7_wbc_g1::green_context
    isaac::operators::agile_wbc
    isaac::operators::g1_low_level::joint_state_subscriber
    isaac::operators::gr00t_action_handler
    isaac::operators::gr00t_n1_7
    isaac::operators::start_up
    isaac::operators::teleop_twist_command
    isaac::operators::v4l2_camera
)
add_test(NAME test_gr00t_wbc_real_config COMMAND test_gr00t_wbc_real_config)
set_tests_properties(test_gr00t_wbc_real_config PROPERTIES
  LABELS "applications;cpp"
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_executable(test_gr00t_wbc_shutdown_signals test_shutdown_signals.cc)
target_link_libraries(test_gr00t_wbc_shutdown_signals
  PRIVATE
    GTest::gtest_main
    Threads::Threads
    holoscan::core
    isaac::applications::gr00t_n1_7_wbc_g1::shutdown_signals
)
add_test(NAME test_gr00t_wbc_shutdown_signals COMMAND test_gr00t_wbc_shutdown_signals)
set_tests_properties(test_gr00t_wbc_shutdown_signals PROPERTIES
  LABELS "applications;cpp"
  TIMEOUT 120
)
