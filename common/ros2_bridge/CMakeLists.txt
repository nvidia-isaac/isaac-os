# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(ros2_bridge INTERFACE)
add_library(isaac::common::ros2_bridge ALIAS ros2_bridge)

include(isaac_deps/holoscan)
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

target_include_directories(ros2_bridge INTERFACE ${PROJECT_SOURCE_DIR})
target_link_libraries(ros2_bridge INTERFACE
  # INTERFACE, not just a transitive dependency: tensor_converter.h and image_encoding.h both
  # include <fmt/format.h> and call fmt::format() directly, so every consumer compiles fmt
  # calls of its own.
  fmt::fmt
  holoscan::core
  holoscan::sensor_io
  isaac::common::camera_info_cpu_messages
  isaac::common::geometry_cpu_messages
  isaac::common::image_schema
  isaac::common::joint_cpu_messages
  isaac::common::math_cpu_messages
)
# Each ROS message package lists its generated targets in <package>_TARGETS.
target_link_libraries(ros2_bridge INTERFACE
  ${geometry_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  rclcpp::rclcpp
)

# The CUDA buffer backend is optional. Only ROS releases that ship rosidl::Buffer (Lyrical) have the
# cuda_buffer package, so Jammy/Humble builds configure without it and are host-only.
find_package(CUDAToolkit QUIET)
find_package(cuda_buffer QUIET)
if(CUDAToolkit_FOUND AND cuda_buffer_FOUND)
  target_link_libraries(ros2_bridge INTERFACE
    CUDA::cudart
    cuda_buffer::cuda_buffer
  )
  target_compile_definitions(ros2_bridge INTERFACE ISAAC_ROS2_BRIDGE_CUDA_BUFFER=1)
  message(STATUS "ros2_bridge: CUDA buffer backend enabled")
else()
  message(STATUS "ros2_bridge: cuda_buffer not found, kCudaDevice topics are unavailable")
endif()

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
