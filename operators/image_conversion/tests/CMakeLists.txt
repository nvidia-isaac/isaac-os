# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_nv12_layout test_nv12_layout.cc)
target_include_directories(test_nv12_layout PRIVATE ${PROJECT_SOURCE_DIR})

add_test(NAME test_nv12_layout COMMAND test_nv12_layout)
set_tests_properties(test_nv12_layout PROPERTIES
  LABELS "cpp;operators"
  TIMEOUT 30
)
