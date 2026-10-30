# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

add_executable(test_geometry_cpu_messages test_geometry_cpu_messages.cc)
target_link_libraries(test_geometry_cpu_messages
  PRIVATE
    GTest::gtest_main
    isaac::common::geometry_cpu_messages
)

add_test(NAME test_geometry_cpu_messages COMMAND test_geometry_cpu_messages)
set_tests_properties(test_geometry_cpu_messages PROPERTIES
  LABELS "cpp;common"
  TIMEOUT 30
)
