# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(gr00t_action_handler_op SHARED gr00t_action_handler_op.cc)
add_library(isaac::operators::gr00t_action_handler ALIAS gr00t_action_handler_op)

include(isaac_deps/holoscan)
target_include_directories(gr00t_action_handler_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gr00t_action_handler_op
  PUBLIC
    holoscan::core
    isaac::common::joint_command_chunk_cpu
    isaac::common::joint_cpu_messages
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
