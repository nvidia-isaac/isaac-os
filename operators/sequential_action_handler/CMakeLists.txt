# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(sequential_action_handler SHARED sequential_action_handler_op.cc)
add_library(isaac::operators::sequential_action_handler ALIAS sequential_action_handler)

include(isaac_deps/holoscan)
target_include_directories(sequential_action_handler PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(sequential_action_handler
  PUBLIC
    holoscan::core
    isaac::common::joint_command_chunk_cpu
    isaac::common::joint_cpu_messages
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
