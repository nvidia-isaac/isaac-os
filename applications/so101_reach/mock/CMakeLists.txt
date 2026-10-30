# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/cuda)
include(isaac_deps/holoscan)
include(isaac_deps/so101_reach_policy)

add_executable(mock_so101_reach
  main.cc
  mock_operators.cc
)
target_include_directories(mock_so101_reach PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(mock_so101_reach
  PRIVATE
    CLI11::CLI11
    CUDA::cudart
    holoscan::config_io
    holoscan::core
    holoscan::params
    isaac::common::joint_cpu_messages
    isaac::common::so101_cpu_messages
    isaac::operators::so101
    isaac::operators::so101_reach
)
add_dependencies(mock_so101_reach grip_reach_delta_policy_onnx)

if(ISAAC_BUILD_TESTS)
  add_test(NAME test_mock_so101_reach
    COMMAND mock_so101_reach
      --config-path=${PROJECT_SOURCE_DIR}/applications/so101_reach/config/so101_reach_config.yaml
      --message-count=10
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
  set_tests_properties(test_mock_so101_reach PROPERTIES
    LABELS "applications;cpp;gpu"
    SKIP_RETURN_CODE 77
    TIMEOUT 120
  )
endif()
