# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

set(camera_info_cpu_messages_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

holoscan_add_flatbuffer_schema(
  TARGET camera_info_cpu_generated
  SCHEMAS camera_info_cpu.fbs
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/common/math_cpu_messages
  OUTPUT_DIR "${camera_info_cpu_messages_generated_dir}"
  BFBS_EMBED
  TRAITS_HEADER camera_info_cpu_schema_traits.h
  MAX_SERIALIZED_SIZE 2048
)

add_library(camera_info_cpu_messages INTERFACE)
add_library(isaac::common::camera_info_cpu_messages ALIAS camera_info_cpu_messages)
add_dependencies(camera_info_cpu_messages camera_info_cpu_generated)
target_include_directories(camera_info_cpu_messages INTERFACE
  ${camera_info_cpu_messages_generated_dir}
)
target_link_libraries(camera_info_cpu_messages INTERFACE
  holoscan::core
  holoscan::schema
  isaac::common::math_cpu_messages
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
