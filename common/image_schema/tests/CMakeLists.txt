# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_image_schema test_image_schema.cc)
target_link_libraries(test_image_schema
  PRIVATE
    GTest::gtest_main
    isaac::common::image_schema
)
add_test(NAME test_image_schema COMMAND test_image_schema)
set_tests_properties(test_image_schema PROPERTIES LABELS "cpp;common" TIMEOUT 30)
