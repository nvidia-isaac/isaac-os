# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/cuda)
include(isaac_deps/holoscan)

set(_ctest_skip_return_code 77)

add_executable(mock_wbc_g1 main.cc)
target_compile_definitions(mock_wbc_g1 PRIVATE
  ISAAC_CTEST_SKIP_RETURN_CODE=${_ctest_skip_return_code}
)
target_include_directories(mock_wbc_g1 PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(mock_wbc_g1
  PRIVATE
    CLI11::CLI11
    CUDA::cudart
    holoscan::config_io
    holoscan::core
    holoscan::params
    isaac::operators::agile_wbc
    isaac::operators::mock::joint_state
    isaac::operators::mock::sink
)
add_dependencies(mock_wbc_g1 nvidia_wbc_agile_velocity_unitree_g1)

if(ISAAC_BUILD_TESTS)
  set(_test_workdir "${CMAKE_CURRENT_BINARY_DIR}/test_workdir")
  set(_test_model_dir "${_test_workdir}/data/nvidia_wbc_agile_velocity_unitree_g1")
  file(MAKE_DIRECTORY "${_test_model_dir}")
  file(CREATE_LINK
    "${PROJECT_SOURCE_DIR}/operators/agile_wbc/tests/testdata/agile_wbc_stub.onnx"
    "${_test_model_dir}/unitree_g1_velocity_e2e.onnx"
    SYMBOLIC
  )

  add_test(NAME test_mock_wbc_g1
    COMMAND mock_wbc_g1
      --config-path=${PROJECT_SOURCE_DIR}/applications/wbc_g1/config/wbc_g1_config.yaml
      --message-count=4
    WORKING_DIRECTORY "${_test_workdir}"
  )
  set_tests_properties(test_mock_wbc_g1 PROPERTIES
    LABELS "applications;cpp;gpu"
    SKIP_RETURN_CODE ${_ctest_skip_return_code}
    TIMEOUT 120
  )

endif()
