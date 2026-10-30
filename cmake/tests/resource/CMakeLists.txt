# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(resource_requirements_tests NONE)

# Standalone use defaults to tests on; embedded builds honor the project option.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  option(ISAAC_BUILD_TESTS "Build Isaac tests" ON)
endif()
if(NOT ISAAC_BUILD_TESTS)
  return()
endif()

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
  NAME resource_requirements
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_resource_requirements.py"
)

set_tests_properties(resource_requirements PROPERTIES
  LABELS "cmake;resource;requirements"
  TIMEOUT 60
)

add_test(
  NAME resource_download
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_resource_download.py"
)
set_tests_properties(resource_download PROPERTIES
  LABELS "cmake;resource;download"
  TIMEOUT 60
)
