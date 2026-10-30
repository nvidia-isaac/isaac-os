# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/cuda)
include(isaac_deps/holoscan)

# Sinus motion op uses CPU-array graph payloads; UCX converts the wire tensors.
add_library(so101_sinus_motion_sim_op STATIC sinus_motion_op.cc)
target_include_directories(so101_sinus_motion_sim_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(so101_sinus_motion_sim_op
  PUBLIC
    holoscan::core
    isaac::common::joint_cpu_messages
)

add_executable(so101_sinus_motion_sim main.cc)
target_include_directories(so101_sinus_motion_sim PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(so101_sinus_motion_sim
  PRIVATE
    CLI11::CLI11
    CUDA::cudart
    CUDA::cuda_driver
    holoscan::config
    holoscan::config_io
    holoscan::core
    holoscan::visualization::op_holoviz
    isaac::operators::ucxx_ops
    so101_sinus_motion_sim_op
)

if(ISAAC_BUILD_TESTS)
  include(isaac_deps/gtest)
  add_executable(test_so101_sim_params tests/test_sinus_motion_params.cc)
  target_link_libraries(test_so101_sim_params
    PRIVATE
      GTest::gtest_main
      holoscan::core
      so101_sinus_motion_sim_op
  )
  add_test(NAME test_so101_sim_params COMMAND test_so101_sim_params)
  set_tests_properties(test_so101_sim_params PROPERTIES
    LABELS "cpp;applications" TIMEOUT 30)

  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/config/so101_sinus_motion_sim_config.yaml" sim_config)
  # Exercise decimal positions and gains through the application's actual YAML decoder.
  # Zero amplitude and zero frequency must both accept a nonzero decimal center.
  foreach(zero_field IN ITEMS sin_amplitude sin_frequency)
    set(decimal_config "${sim_config}")
    foreach(entry IN ITEMS "default_position;0.05" "kd;0.1" "kp;30.1")
      list(GET entry 0 key)
      list(GET entry 1 value)
      string(REGEX REPLACE "${key}:[^\n]*" "${key}: [${value}]"
        decimal_config "${decimal_config}")
    endforeach()
    string(REGEX REPLACE "${zero_field}:[^\n]*" "${zero_field}: 0.0"
      decimal_config "${decimal_config}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/decimal_${zero_field}_test.yaml" "${decimal_config}")
    add_test(NAME test_so101_sim_graph_decimal_${zero_field}
      COMMAND so101_sinus_motion_sim --no-connect
        --config-path=${CMAKE_CURRENT_BINARY_DIR}/decimal_${zero_field}_test.yaml)
    set_tests_properties(test_so101_sim_graph_decimal_${zero_field} PROPERTIES
      LABELS "cpp;applications" TIMEOUT 30)
  endforeach()
  set(camera_config "${sim_config}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/camera_test.yaml" "${camera_config}")
  add_test(NAME test_so101_sim_graph_camera
    COMMAND so101_sinus_motion_sim --no-connect --message-count=10
      --config-path=${CMAKE_CURRENT_BINARY_DIR}/camera_test.yaml)
  set_tests_properties(test_so101_sim_graph_camera PROPERTIES
    LABELS "cpp;applications" TIMEOUT 30)
  set(defaults_config "${sim_config}")
  foreach(key IN ITEMS
      camera_tensor_buffer_size
      default_position
      enable_camera
      image_flatbuffer_buffer_size
      joint_state_flatbuffer_buffer_size
      kd
      kp)
    string(REGEX REPLACE "[ \t]+${key}:[^\n]*\n" "" defaults_config "${defaults_config}")
  endforeach()
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/defaults_test.yaml" "${defaults_config}")
  add_test(NAME test_so101_sim_graph_defaults
    COMMAND so101_sinus_motion_sim --no-connect --message-count=10
      --config-path=${CMAKE_CURRENT_BINARY_DIR}/defaults_test.yaml)
  set_tests_properties(test_so101_sim_graph_defaults PROPERTIES
    LABELS "cpp;applications" TIMEOUT 30)
  string(REPLACE "enable_camera: true" "enable_camera: false" headless_config "${sim_config}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/headless_test.yaml" "${headless_config}")
  foreach(mode IN ITEMS continuous bounded)
    set(extra_args)
    if(mode STREQUAL "bounded")
      set(extra_args --message-count=10)
    endif()
    add_test(NAME test_so101_sim_graph_${mode}
      COMMAND so101_sinus_motion_sim --no-connect
        --config-path=${CMAKE_CURRENT_BINARY_DIR}/headless_test.yaml
        ${extra_args})
    set_tests_properties(test_so101_sim_graph_${mode} PROPERTIES
      LABELS "cpp;applications" TIMEOUT 30)
  endforeach()
endif()
