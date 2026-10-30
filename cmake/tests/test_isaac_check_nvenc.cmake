# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

include("${CMAKE_CURRENT_LIST_DIR}/../isaac_check_nvenc.cmake")

set(_fixture "${CMAKE_CURRENT_BINARY_DIR}/isaac_check_nvenc_fixture")
file(REMOVE_RECURSE "${_fixture}")
file(MAKE_DIRECTORY "${_fixture}/nvidia")
file(WRITE "${_fixture}/nvidia/libnvidia-encode.so.1" "")

set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)
set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)
set(CMAKE_FIND_USE_INSTALL_PREFIX FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)
set(CMAKE_PREFIX_PATH "")
set(CMAKE_FRAMEWORK_PATH "")
set(CMAKE_LIBRARY_PATH "${_fixture}")

isaac_check_nvenc(SYSTEM_HAS_NVENC)
if(NOT SYSTEM_HAS_NVENC)
  message(FATAL_ERROR "Expected NVENC support when the encode driver is available")
endif()

file(REMOVE_RECURSE "${_fixture}")
isaac_check_nvenc(SYSTEM_HAS_NVENC)
if(SYSTEM_HAS_NVENC)
  message(FATAL_ERROR "Expected no NVENC support without the encode driver")
endif()
