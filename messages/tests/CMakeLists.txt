# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)
include(isaac_deps/holoscan)

set(legacy_tensor_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

holoscan_add_flatbuffer_schema(
  TARGET legacy_tensor_wire_schema
  OUTPUT_DIR "${legacy_tensor_generated_dir}"
  SCHEMAS schemas/legacy_tensor.fbs
)

add_executable(test_tensor_wire_compatibility test_tensor_wire_compatibility.cc)
add_dependencies(test_tensor_wire_compatibility legacy_tensor_wire_schema)
target_include_directories(test_tensor_wire_compatibility
  PRIVATE
    "${legacy_tensor_generated_dir}"
)
target_link_libraries(test_tensor_wire_compatibility
  PRIVATE
    GTest::gtest_main
    flatbuffers::flatbuffers
    holoscan::schema
)

add_test(NAME test_tensor_wire_compatibility COMMAND test_tensor_wire_compatibility)
set_tests_properties(test_tensor_wire_compatibility PROPERTIES
  LABELS "cpp;messages"
  TIMEOUT 30
)
