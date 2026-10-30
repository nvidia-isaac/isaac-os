# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

set(geometry_cpu_messages_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

holoscan_add_flatbuffer_schema(
  TARGET odometry_cpu_generated
  SCHEMAS odometry_cpu.fbs
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/common/math_cpu_messages
  OUTPUT_DIR "${geometry_cpu_messages_generated_dir}"
  BFBS_EMBED
  TRAITS_HEADER odometry_cpu_schema_traits.h
  MAX_SERIALIZED_SIZE 2048
)

holoscan_add_flatbuffer_schema(
  TARGET pose_cpu_generated
  SCHEMAS pose_cpu.fbs
  OUTPUT_DIR "${geometry_cpu_messages_generated_dir}"
  BFBS_EMBED
  TRAITS_HEADER pose_cpu_schema_traits.h
  MAX_SERIALIZED_SIZE 512
)

holoscan_add_flatbuffer_schema(
  TARGET transform_cpu_generated
  SCHEMAS transform_cpu.fbs
  OUTPUT_DIR "${geometry_cpu_messages_generated_dir}"
  BFBS_EMBED
  TRAITS_HEADER transform_cpu_schema_traits.h
  MAX_SERIALIZED_SIZE 768
)

holoscan_add_flatbuffer_schema(
  TARGET twist_cpu_generated
  SCHEMAS twist_cpu.fbs
  OUTPUT_DIR "${geometry_cpu_messages_generated_dir}"
  BFBS_EMBED
  TRAITS_HEADER twist_cpu_schema_traits.h
  MAX_SERIALIZED_SIZE 512
)

add_library(geometry_cpu_messages INTERFACE)
add_library(isaac::common::geometry_cpu_messages ALIAS geometry_cpu_messages)
add_dependencies(geometry_cpu_messages
  odometry_cpu_generated
  pose_cpu_generated
  transform_cpu_generated
  twist_cpu_generated
)
# Treat generated FlatBuffers headers as system headers so CUDA diagnostics
# such as unused generated _va variables do not leak into consumers.
target_include_directories(geometry_cpu_messages SYSTEM INTERFACE
  ${geometry_cpu_messages_generated_dir}
)
target_link_libraries(geometry_cpu_messages INTERFACE
  holoscan::core
  holoscan::schema
  isaac::common::math_cpu_messages
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
