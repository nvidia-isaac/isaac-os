# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The distortion mappings need neither cuVSLAM nor a GPU.
include(isaac_deps/gtest)

add_executable(test_calibration_mapping test_calibration_mapping.cc)
target_include_directories(test_calibration_mapping PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_calibration_mapping
  PRIVATE
    GTest::gtest_main
)

add_test(NAME test_calibration_mapping COMMAND test_calibration_mapping)
set_tests_properties(test_calibration_mapping PROPERTIES
  LABELS "cpp;operators"
  TIMEOUT 30
)
