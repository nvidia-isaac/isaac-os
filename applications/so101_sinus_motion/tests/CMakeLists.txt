# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

add_executable(test_so101_sinus_params_config
  test_motion_profile.cc
  test_so101_sinus_params_config.cc
)
target_compile_definitions(test_so101_sinus_params_config
  PRIVATE
    "SO101_SINUS_CONFIG=\"${CMAKE_CURRENT_SOURCE_DIR}/../config/so101_sinus_motion_config.yaml\""
)
target_link_libraries(test_so101_sinus_params_config
  PRIVATE
    GTest::gtest_main
    holoscan::config_io
    holoscan::params
    isaac::common::so101_driver
    isaac::operators::simple_motion_planner
    isaac::operators::sinus_motion
    isaac::operators::so101
)

add_test(NAME test_so101_sinus_params_config COMMAND test_so101_sinus_params_config)
set_tests_properties(test_so101_sinus_params_config PROPERTIES
  LABELS "applications;cpp"
  TIMEOUT 30
)
