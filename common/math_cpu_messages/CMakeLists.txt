# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)
include(isaac_deps/eigen)
include(isaac_deps/flatbuffers)

set(math_cpu_messages_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Embeddable structs, not standalone messages -- no root_type in the .fbs and no TRAITS_* here
# (TRAITS_HEADER and its associated metadata are only needed for a schema with its own
# registered root type).
holoscan_add_flatbuffer_schema(
  TARGET matrix3x4d_generated
  SCHEMAS matrix3x4d.fbs
  OUTPUT_DIR "${math_cpu_messages_generated_dir}"
)

holoscan_add_flatbuffer_schema(
  TARGET matrix6d_generated
  SCHEMAS matrix6d.fbs
  OUTPUT_DIR "${math_cpu_messages_generated_dir}"
)

# Compiled (not INTERFACE, unlike its sibling *_cpu_messages libs): Pack/UnPack need ordinary
# (non-inline) linkage so consumers only need the declaration in matrix*d_adapter.h in scope,
# not the definition -- see matrix3d_adapter.cc et al.
add_library(math_cpu_messages
  matrix3x4d_adapter.cc
  matrix6d_adapter.cc
)
add_library(isaac::common::math_cpu_messages ALIAS math_cpu_messages)
add_dependencies(math_cpu_messages
  matrix3x4d_generated
  matrix6d_generated
)
target_include_directories(math_cpu_messages PUBLIC
  ${math_cpu_messages_generated_dir}
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(math_cpu_messages PUBLIC
  Eigen3::Eigen
  flatbuffers::flatbuffers
  holoscan::core
  holoscan::schema
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
