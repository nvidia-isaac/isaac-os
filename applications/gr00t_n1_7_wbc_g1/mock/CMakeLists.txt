# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/cuda)
include(isaac_deps/holoscan)

set(_ctest_skip_return_code 77)

add_executable(mock_gr00t_n1_7_wbc_g1
  main.cc
  mock_operators.cc
)
target_compile_definitions(mock_gr00t_n1_7_wbc_g1 PRIVATE
  ISAAC_CTEST_SKIP_RETURN_CODE=${_ctest_skip_return_code}
)
target_include_directories(mock_gr00t_n1_7_wbc_g1 PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(mock_gr00t_n1_7_wbc_g1
  PRIVATE
    CLI11::CLI11
    CUDA::cudart
    holoscan::config_io
    holoscan::core
    holoscan::params
    holoscan::sensor_io
    isaac::applications::gr00t_n1_7_wbc_g1::green_context
    isaac::applications::gr00t_n1_7_wbc_g1::shutdown_signals
    isaac::common::joint_command_chunk_cpu
    isaac::common::joint_cpu_messages
    isaac::operators::agile_wbc
    isaac::operators::gr00t_action_handler
    isaac::operators::gr00t_n1_7
    isaac::operators::joint_command_combiner
)
add_dependencies(mock_gr00t_n1_7_wbc_g1
  gr00t_n1_7_g1_apple_to_plate_models
  nvidia_wbc_agile_velocity_unitree_g1
)

if(ISAAC_BUILD_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  # Exercise CPU placement independently of the GPU's supported SM allocations.
  set(_mock_smoke_config "${CMAKE_CURRENT_BINARY_DIR}/mock_smoke.yaml")
  configure_file(../config/control.yaml "${_mock_smoke_config}" COPYONLY)
  file(APPEND "${_mock_smoke_config}" "\ngreen_context:\n  wbc_sms: 0\n")
  add_test(NAME test_mock_gr00t_n1_7_wbc_g1
    COMMAND "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/common/benchmark/gr00t/tests/run_benchmark_affinity_smoke.py"
      "--config=${_mock_smoke_config}"
      "--model-config=${PROJECT_SOURCE_DIR}/operators/gr00t_n1_7/config/n1_7_g1.yaml"
      "--executable=$<TARGET_FILE:mock_gr00t_n1_7_wbc_g1>"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
  set_tests_properties(test_mock_gr00t_n1_7_wbc_g1 PROPERTIES
    LABELS "applications;cpp;gpu"
    SKIP_RETURN_CODE ${_ctest_skip_return_code}
    TIMEOUT 3600
  )
endif()
