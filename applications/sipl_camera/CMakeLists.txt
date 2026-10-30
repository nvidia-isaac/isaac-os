# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan-camera)
if(NOT TARGET holoscan::sipl_capture_op)
  message(STATUS "Skipping sipl_camera: Holoscan Camera does not provide SIPL capture for the installed SDK")
  return()
endif()
include(isaac_deps/cli11)

add_executable(sipl_camera
  main.cc
  sipl_metadata_sink_op.cc
)
target_include_directories(sipl_camera PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(sipl_camera
  PRIVATE
    CLI11::CLI11
    holoscan::core
    holoscan::visualization::op_holoviz
    holoscan::sipl_capture_op
)

if(ISAAC_BUILD_TESTS)
  add_test(NAME test_sipl_camera_compile
    COMMAND sipl_camera
      --compile-only
      --json-config=${PROJECT_SOURCE_DIR}/common/sipl/config/vb1940_single.json
  )
  set_tests_properties(test_sipl_camera_compile PROPERTIES
    LABELS "applications;cpp;integration"
    TIMEOUT 60
  )
  add_test(NAME test_sipl_camera_stereo_compile
    COMMAND sipl_camera
      --compile-only
      --stereo
      --json-config=${PROJECT_SOURCE_DIR}/common/sipl/config/vb1940_stereo.json
  )
  set_tests_properties(test_sipl_camera_stereo_compile PROPERTIES
    LABELS "applications;cpp;integration"
    TIMEOUT 60
  )
endif()
