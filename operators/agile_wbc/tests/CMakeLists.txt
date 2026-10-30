# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_agile_wbc test_agile_wbc_op.cc)
add_executable(test_wbc_params_validation
  test_wbc_params_validation.cc
  ../wbc_params_validation.cc
)

include(isaac_deps/gtest)
set(_ctest_skip_return_code 77)
target_compile_definitions(test_agile_wbc PRIVATE
  ISAAC_CTEST_SKIP_RETURN_CODE=${_ctest_skip_return_code}
  "ISAAC_WBC_TEST_MODEL=\"${CMAKE_CURRENT_SOURCE_DIR}/testdata/agile_wbc_stub.onnx\""
)
target_include_directories(test_agile_wbc PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_agile_wbc
  PRIVATE
    GTest::gtest
    isaac::common::joint_cpu_messages
    isaac::operators::agile_wbc
)

target_include_directories(test_wbc_params_validation PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_wbc_params_validation
  PRIVATE
    GTest::gtest_main
    holoscan::core
)

add_test(NAME test_agile_wbc COMMAND test_agile_wbc)
add_test(NAME test_wbc_params_validation COMMAND test_wbc_params_validation)
set_tests_properties(test_agile_wbc PROPERTIES
  LABELS "cpp;gpu;operators"
  SKIP_RETURN_CODE ${_ctest_skip_return_code}
  TIMEOUT 120
)
set_tests_properties(test_wbc_params_validation PROPERTIES
  LABELS "cpp;operators"
)
