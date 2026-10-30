# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(joint_command_combiner SHARED joint_command_combiner_op.cc)
add_library(isaac::operators::joint_command_combiner ALIAS joint_command_combiner)

include(isaac_deps/holoscan)
target_include_directories(joint_command_combiner PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(joint_command_combiner
  PUBLIC
    holoscan::core
    isaac::common::joint_cpu_messages
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
