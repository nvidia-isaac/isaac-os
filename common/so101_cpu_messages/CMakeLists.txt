# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

set(so101_cpu_messages_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

holoscan_add_flatbuffer_schema(
  TARGET policy_delta_cpu_generated
  SCHEMAS policy_delta_cpu.fbs
  OUTPUT_DIR "${so101_cpu_messages_generated_dir}"
  BFBS_EMBED
  TRAITS_HEADER policy_delta_cpu_schema_traits.h
  MAX_SERIALIZED_SIZE 128
)

holoscan_add_flatbuffer_schema(
  TARGET state_sample_cpu_generated
  SCHEMAS state_sample_cpu.fbs
  OUTPUT_DIR "${so101_cpu_messages_generated_dir}"
  BFBS_EMBED
  TRAITS_HEADER state_sample_cpu_schema_traits.h
  MAX_SERIALIZED_SIZE 256
)

add_library(so101_cpu_messages INTERFACE)
add_library(isaac::common::so101_cpu_messages ALIAS so101_cpu_messages)
add_dependencies(so101_cpu_messages
  policy_delta_cpu_generated
  state_sample_cpu_generated
)
# Treat generated FlatBuffers headers as system headers so CUDA diagnostics
# such as unused generated _va variables do not leak into consumers.
target_include_directories(so101_cpu_messages SYSTEM INTERFACE
  ${so101_cpu_messages_generated_dir}
)
target_link_libraries(so101_cpu_messages INTERFACE
  holoscan::core
  holoscan::schema
)
