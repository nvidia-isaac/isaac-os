# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

set(joint_command_chunk_cpu_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

holoscan_add_flatbuffer_schema(
  TARGET joint_command_chunk_cpu_generated
  SCHEMAS joint_command_chunk_cpu.fbs
  OUTPUT_DIR "${joint_command_chunk_cpu_generated_dir}"
  BFBS_EMBED
  TRAITS_HEADER joint_command_chunk_cpu_schema_traits.h
  # Covers the 43-joint, 40-action simulator N1.7 output, including FlatBuffer overhead.
  MAX_SERIALIZED_SIZE 16384
)

add_library(joint_command_chunk_cpu INTERFACE)
add_library(isaac::common::joint_command_chunk_cpu ALIAS joint_command_chunk_cpu)
add_dependencies(joint_command_chunk_cpu joint_command_chunk_cpu_generated)
target_include_directories(joint_command_chunk_cpu INTERFACE
  ${PROJECT_SOURCE_DIR}
  ${joint_command_chunk_cpu_generated_dir}
)
target_link_libraries(joint_command_chunk_cpu INTERFACE
  holoscan::core
  holoscan::schema
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
