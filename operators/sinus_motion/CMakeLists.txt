# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

add_library(sinus_motion_op SHARED
  sinus_motion_op.cc
)
add_library(isaac::operators::sinus_motion ALIAS sinus_motion_op)
target_include_directories(sinus_motion_op
  PUBLIC
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(sinus_motion_op
  PUBLIC
    holoscan::core
    isaac::common::joint_cpu_messages
    isaac::common::so101_cpu_messages
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
