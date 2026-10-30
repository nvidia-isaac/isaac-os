# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

add_executable(ros2_bridge_reference main.cc)
target_include_directories(ros2_bridge_reference PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(ros2_bridge_reference
  PRIVATE
    holoscan::core
    isaac::common::geometry_cpu_messages
    isaac::common::ros2_bridge
)

if(ISAAC_BUILD_TESTS)
  add_test(NAME test_ros2_bridge_reference
    COMMAND ros2_bridge_reference --duration-s=2
  )
  set_tests_properties(test_ros2_bridge_reference PROPERTIES
    LABELS "applications;cpp;integration"
    TIMEOUT 30
  )

  add_subdirectory(tests)
endif()
