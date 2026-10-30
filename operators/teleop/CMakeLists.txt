# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)
include(isaac_deps/libevdev)

add_library(teleop_twist_command_op SHARED
  evdev_device.cc
  teleop_params_config_codec.cc
  teleop_twist_command_op.cc
)
add_library(isaac::operators::teleop_twist_command ALIAS teleop_twist_command_op)

target_include_directories(teleop_twist_command_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(teleop_twist_command_op
  PUBLIC
    holoscan::config
    holoscan::core
    isaac::common::geometry_cpu_messages
  PRIVATE
    libevdev::libevdev
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
