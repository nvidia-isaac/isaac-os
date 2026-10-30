# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

# -- tensorrt_engine_test ------------------------------------------------------

add_executable(tensorrt_engine_test tensorrt_engine_test.cc)
target_compile_definitions(tensorrt_engine_test PRIVATE
  TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata"
  TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(tensorrt_engine_test
  PRIVATE
    GTest::gtest_main
    isaac::common::tensorrt_engine
)
add_test(NAME tensorrt_engine_test COMMAND tensorrt_engine_test)
set_tests_properties(tensorrt_engine_test PROPERTIES LABELS "cpp;common")
