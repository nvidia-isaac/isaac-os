# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

add_executable(test_so101_reach_params_config test_so101_reach_params_config.cc)
target_compile_definitions(test_so101_reach_params_config
  PRIVATE
    "SO101_REACH_CONFIG=\"${CMAKE_CURRENT_SOURCE_DIR}/../config/so101_reach_config.yaml\""
)
target_link_libraries(test_so101_reach_params_config
  PRIVATE
    GTest::gtest_main
    holoscan::config_io
    holoscan::params
    isaac::common::so101_driver
    isaac::operators::so101
    isaac::operators::so101_reach
)

add_test(NAME test_so101_reach_params_config COMMAND test_so101_reach_params_config)
set_tests_properties(test_so101_reach_params_config PROPERTIES
  LABELS "applications;cpp"
  TIMEOUT 30
)
