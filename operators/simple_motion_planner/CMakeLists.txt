# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

add_library(simple_motion_planner_op SHARED
  simple_motion_planner.cc
)
add_library(isaac::operators::simple_motion_planner ALIAS simple_motion_planner_op)
target_include_directories(simple_motion_planner_op
  PUBLIC
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(simple_motion_planner_op
  PUBLIC
    holoscan::core
    isaac::common::joint_cpu_messages
    isaac::common::so101_cpu_messages
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
