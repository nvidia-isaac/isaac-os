# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

set(_fixture "${CMAKE_CURRENT_BINARY_DIR}/isaac_compile_link_options_fixture")
file(REMOVE_RECURSE "${_fixture}")
file(MAKE_DIRECTORY "${_fixture}")
file(WRITE "${_fixture}/main.cc" "static_assert(__cplusplus >= 202002L);\nint main() { return 0; }\n")
file(WRITE "${_fixture}/main.cu" "int main() { return 0; }\n")
file(WRITE "${_fixture}/CMakeLists.txt" [[
cmake_minimum_required(VERSION 4.0)
project(linker_check LANGUAGES C CXX CUDA)
set(CMAKE_LINKER_TYPE "${REQUESTED_LINKER}")
if(TEST_CASE STREQUAL NO_ALTERNATIVE)
  # Simulate unavailable alternative linkers through the probe results.
  foreach(linker IN ITEMS GOLD LLD MOLD)
    set(${linker}_FOUND FALSE)
  endforeach()
endif()
include("${ISAAC_CMAKE_DIR}/isaac_compile_link_options.cmake")
if(NOT CMAKE_LINKER_TYPE)
  message(FATAL_ERROR "Automatic linker selection did not set CMAKE_LINKER_TYPE")
endif()
if(EXPECTED_LINKER AND NOT CMAKE_LINKER_TYPE STREQUAL EXPECTED_LINKER)
  message(FATAL_ERROR "Requested ${EXPECTED_LINKER}, selected ${CMAKE_LINKER_TYPE}")
endif()
add_executable(linker_check main.cc)
add_executable(linker_check_cuda main.cu)
foreach(target IN ITEMS linker_check linker_check_cuda)
  get_target_property(linker ${target} LINKER_TYPE)
  if(NOT linker STREQUAL CMAKE_LINKER_TYPE)
    message(FATAL_ERROR "${target} did not inherit the selected linker")
  endif()
endforeach()
]])

foreach(_case IN ITEMS AUTO BFD DEFAULT NO_ALTERNATIVE)
  set(_linker "")
  set(_expected "")
  set(_build_type RelWithDebInfo)
  if(_case STREQUAL BFD OR _case STREQUAL DEFAULT)
    set(_linker "${_case}")
    set(_expected "${_case}")
  endif()
  if(_case STREQUAL BFD)
    set(_build_type Release)
  elseif(_case STREQUAL DEFAULT)
    set(_build_type FastBuild)
  elseif(_case STREQUAL NO_ALTERNATIVE)
    set(_expected DEFAULT)
    set(_build_type Debug)
  endif()
  set(_build "${_fixture}/${_case}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${_fixture}" -B "${_build}" -G "${CMAKE_GENERATOR}"
      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
      "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
      "-DISAAC_CMAKE_DIR=${CMAKE_CURRENT_LIST_DIR}/.."
      "-DREQUESTED_LINKER=${_linker}" "-DEXPECTED_LINKER=${_expected}"
      "-DTEST_CASE=${_case}"
      "-DCMAKE_BUILD_TYPE=${_build_type}"
    RESULT_VARIABLE _result
  )
  if(_result)
    message(FATAL_ERROR "Linker configuration failed: ${_case}")
  endif()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${_build}" --config "${_build_type}"
    RESULT_VARIABLE _result
  )
  if(_result)
    message(FATAL_ERROR "Linker build failed: ${_case}")
  endif()
endforeach()
