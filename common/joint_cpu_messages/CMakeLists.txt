# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

set(joint_cpu_messages_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

holoscan_add_flatbuffer_schema(
  TARGET joint_command_cpu_generated
  SCHEMAS joint_command_cpu.fbs
  OUTPUT_DIR "${joint_cpu_messages_generated_dir}"
  BFBS_EMBED
  TRAITS_HEADER joint_command_cpu_schema_traits.h
  MAX_SERIALIZED_SIZE 2048
)

holoscan_add_flatbuffer_schema(
  TARGET joint_state_cpu_generated
  SCHEMAS joint_state_cpu.fbs
  OUTPUT_DIR "${joint_cpu_messages_generated_dir}"
  BFBS_EMBED
  TRAITS_HEADER joint_state_cpu_schema_traits.h
  MAX_SERIALIZED_SIZE 3000
)

add_library(joint_cpu_messages INTERFACE)
add_library(isaac::common::joint_cpu_messages ALIAS joint_cpu_messages)
add_dependencies(joint_cpu_messages
  joint_command_cpu_generated
  joint_state_cpu_generated
)
# Treat generated FlatBuffers headers as system headers so CUDA diagnostics
# such as unused generated _va variables do not leak into consumers.
target_include_directories(joint_cpu_messages SYSTEM INTERFACE
  ${joint_cpu_messages_generated_dir}
)
target_link_libraries(joint_cpu_messages INTERFACE
  holoscan::core
  holoscan::schema
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
