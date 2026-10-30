# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)
include(isaac_deps/unitree-sdk2)

add_library(g1_imu_conversion SHARED
  unitree_imu_conversion.cc
)
add_library(isaac::operators::g1_low_level::imu_conversion ALIAS g1_imu_conversion)
target_include_directories(g1_imu_conversion PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(g1_imu_conversion
  PUBLIC
    holoscan::sensor_io
)

add_library(g1_low_level_joint_command_publisher_op SHARED
  g1_low_level_joint_command_publisher.cc
)
add_library(isaac::operators::g1_low_level::joint_command_publisher
  ALIAS g1_low_level_joint_command_publisher_op
)
target_include_directories(g1_low_level_joint_command_publisher_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(g1_low_level_joint_command_publisher_op
  PUBLIC
    holoscan::core
    isaac::common::joint_cpu_messages
    unitree_sdk2
)

add_library(g1_low_level_joint_state_subscriber_op SHARED
  g1_low_level_joint_state_subscriber.cc
)
add_library(isaac::operators::g1_low_level::joint_state_subscriber
  ALIAS g1_low_level_joint_state_subscriber_op
)
target_include_directories(g1_low_level_joint_state_subscriber_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(g1_low_level_joint_state_subscriber_op
  PUBLIC
    holoscan::core
    holoscan::sensor_io
    isaac::operators::g1_low_level::imu_conversion
    isaac::common::joint_cpu_messages
    unitree_sdk2
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
