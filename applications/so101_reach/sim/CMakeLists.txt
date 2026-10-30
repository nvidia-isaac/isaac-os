# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/cuda)
include(isaac_deps/holoscan)
include(isaac_deps/so101_reach_policy)

add_executable(so101_reach_sim main.cc)
target_include_directories(so101_reach_sim PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(so101_reach_sim
  PRIVATE
    CLI11::CLI11
    CUDA::cudart
    holoscan::config_io
    holoscan::core
    holoscan::params
    isaac::common::joint_cpu_messages
    isaac::common::so101_cpu_messages
    isaac::common::so101_driver
    isaac::operators::so101
    isaac::operators::so101_reach
    isaac::operators::ucxx_ops
)
add_dependencies(so101_reach_sim grip_reach_delta_policy_onnx)

if(ISAAC_BUILD_TESTS)
  add_test(NAME test_so101_reach_sim_graph
    COMMAND so101_reach_sim --compile-only
      --config-path=${PROJECT_SOURCE_DIR}/applications/so101_reach/config/so101_reach_config.yaml)
  set_tests_properties(test_so101_reach_sim_graph PROPERTIES
    LABELS "applications;cpp" TIMEOUT 30)
  include(isaac_deps/gtest)
  add_executable(test_so101_reach_sim_adapters tests/test_sim_adapters.cc)
  target_include_directories(test_so101_reach_sim_adapters PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_so101_reach_sim_adapters
    PRIVATE
      GTest::gtest_main
      holoscan::core
      isaac::common::joint_cpu_messages
      isaac::common::so101_cpu_messages
      isaac::common::so101_driver
  )
  add_test(NAME test_so101_reach_sim_adapters COMMAND test_so101_reach_sim_adapters)
  set_tests_properties(test_so101_reach_sim_adapters PROPERTIES
    LABELS "applications;cpp" TIMEOUT 30)
endif()
