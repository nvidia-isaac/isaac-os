# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(external_consumer LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(isaac_os SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../..")

function(import_isaac_os)
  set(ISAAC_BUILD_TESTS OFF)
  set(OP_camera_info ON)
  FetchContent_MakeAvailable(isaac_os)
endfunction()

import_isaac_os()

if(TARGET isaac::operators::so101 OR TARGET isaac::common::tensorrt_engine OR TARGET nv::encoder)
  message(FATAL_ERROR "The consumer enabled unrelated components. Use a clean build directory.")
endif()

add_executable(external_consumer main.cc)
target_compile_features(external_consumer PRIVATE cxx_std_20)
target_link_libraries(external_consumer PRIVATE isaac::operators::camera_info)

enable_testing()
add_test(NAME external_consumer COMMAND external_consumer)
