# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_build_infra test_build_infra.cpp)
target_link_libraries(test_build_infra PRIVATE GTest::gtest_main)
add_test(NAME test_build_infra COMMAND test_build_infra)

add_test(
  NAME test_isaac_os_cli_environment
  COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_isaac_os_cli_environment.sh"
)

add_test(
  NAME test_isaac_compile_link_options
  COMMAND "${CMAKE_COMMAND}"
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
    "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
    "-DCMAKE_GENERATOR=${CMAKE_GENERATOR}"
    "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_isaac_compile_link_options.cmake"
)

add_test(
  NAME test_isaac_cpm_find_package
  COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_SOURCE_DIR}/test_isaac_cpm_find_package.cmake"
)

add_test(
  NAME test_isaac_git_config
  COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_SOURCE_DIR}/test_isaac_git_config.cmake"
)

add_test(
  NAME test_isaac_check_nvenc
  COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_SOURCE_DIR}/test_isaac_check_nvenc.cmake"
)

add_test(
  NAME test_external_consumer
  COMMAND "${CMAKE_CTEST_COMMAND}"
    --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/external_consumer"
      "${CMAKE_CURRENT_BINARY_DIR}/external_consumer"
    --build-generator "${CMAKE_GENERATOR}"
    --build-config "$<CONFIG>"
    --build-noclean
    --build-target external_consumer
    --build-options
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
      "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
      "-DCPM_SOURCE_CACHE=${CPM_SOURCE_CACHE}"
      "-DHOLOHUB_CMAKE_PATH=${HOLOHUB_CMAKE_PATH}"
      "-DHoloscan_DIR=${Holoscan_DIR}"
    --test-command "${CMAKE_CTEST_COMMAND}"
      --test-dir "${CMAKE_CURRENT_BINARY_DIR}/external_consumer"
      --build-config "$<CONFIG>" --no-tests=error --output-on-failure
)
set_tests_properties(test_external_consumer PROPERTIES
  LABELS "cmake;integration"
  TIMEOUT 600
)

add_subdirectory(resource)
