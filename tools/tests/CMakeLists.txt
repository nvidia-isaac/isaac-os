# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(NAME host_preflight_test
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_host_preflight.py")
set_tests_properties(host_preflight_test PROPERTIES
  LABELS "tools;python"
  TIMEOUT 30
)
