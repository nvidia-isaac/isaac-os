# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(start_up_op SHARED
  ${PROJECT_SOURCE_DIR}/common/utils/regex.cc
  start_up_op.cc
  waypoint_sequencer.cc
)
add_library(isaac::operators::start_up ALIAS start_up_op)

include(isaac_deps/holoscan)
target_include_directories(start_up_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(start_up_op
  PUBLIC
    holoscan::config
    holoscan::core
    isaac::common::joint_cpu_messages
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
