# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)
include(isaac_deps/holoscan)

add_executable(test_unitree_imu_conversion test_unitree_imu_conversion.cc)
target_include_directories(test_unitree_imu_conversion PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_unitree_imu_conversion
  PRIVATE
    GTest::gtest_main
    holoscan::core
    isaac::operators::g1_low_level::imu_conversion
)

add_test(NAME test_unitree_imu_conversion COMMAND test_unitree_imu_conversion)
set_tests_properties(test_unitree_imu_conversion PROPERTIES
  LABELS "cpp;operators"
  TIMEOUT 30
)
