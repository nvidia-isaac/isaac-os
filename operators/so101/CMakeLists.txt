# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

add_library(so101_op SHARED
  so101_delta_motion_control_op.cc
  so101_io_ops.cc
)
add_library(isaac::operators::so101 ALIAS so101_op)
target_include_directories(so101_op
  PUBLIC
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(so101_op
  PUBLIC
    holoscan::core
    isaac::common::joint_cpu_messages
    isaac::common::so101_cpu_messages
    isaac::common::so101_driver
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
