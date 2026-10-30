# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/holoscan)

add_executable(mock_so101_sinus_motion
  main.cc
  mock_operators.cc
)
target_include_directories(mock_so101_sinus_motion PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(mock_so101_sinus_motion
  PRIVATE
    CLI11::CLI11
    holoscan::config_io
    holoscan::core
    holoscan::params
    isaac::common::joint_cpu_messages
    isaac::common::so101_cpu_messages
    isaac::operators::simple_motion_planner
    isaac::operators::sinus_motion
    isaac::operators::so101
)

if(ISAAC_BUILD_TESTS)
  add_test(NAME test_mock_so101_sinus_motion
    COMMAND mock_so101_sinus_motion
      --config-path=${PROJECT_SOURCE_DIR}/applications/so101_sinus_motion/config/so101_sinus_motion_config.yaml
      --message-count=10
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
  set_tests_properties(test_mock_so101_sinus_motion PROPERTIES
    LABELS "applications;cpp"
    TIMEOUT 120
  )
endif()
