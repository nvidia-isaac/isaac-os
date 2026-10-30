# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

add_executable(mcap_logger_reference main.cc)
target_include_directories(mcap_logger_reference PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(mcap_logger_reference
  PRIVATE
    holoscan::core
    isaac::common::camera_info_cpu_messages
    isaac::common::geometry_cpu_messages
    isaac::common::image_schema
    isaac::common::joint_cpu_messages
    isaac::common::mcap_logger
)

if(ISAAC_BUILD_TESTS)
  add_test(NAME test_mcap_logger_reference
    COMMAND mcap_logger_reference
      --duration-s=1
      --output-path=${CMAKE_CURRENT_BINARY_DIR}/test_mcap_logger_reference_output
  )
  set_tests_properties(test_mcap_logger_reference PROPERTIES
    LABELS "applications;cpp;integration"
    TIMEOUT 30
  )

  add_test(NAME test_mcap_logger_reference_stream_only
    COMMAND mcap_logger_reference
      --duration-s=1
      --stream
      --no-file
      --websocket-port=0
  )
  set_tests_properties(test_mcap_logger_reference_stream_only PROPERTIES
    LABELS "applications;cpp;integration"
    TIMEOUT 30
  )
endif()
