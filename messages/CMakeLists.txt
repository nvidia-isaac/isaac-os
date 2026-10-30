# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/flatbuffers)
include(isaac_deps/holoscan)

# Wire tables only: graph payloads use existing CPU/sensor schemas.
get_target_property(sdk_schema_dir holoscan::fbs_tools HOLOSCAN_SCHEMA_DIR)
set(wire_schema_names
  isaac_header
  isaac_joint_command
  isaac_joint_state
  isaac_math_types
  isaac_time
)
set(wire_headers)
set(wire_schemas)
foreach(name IN LISTS wire_schema_names)
  list(APPEND wire_headers "${CMAKE_CURRENT_BINARY_DIR}/wire/${name}_generated.h")
  list(APPEND wire_schemas "${CMAKE_CURRENT_SOURCE_DIR}/schemas/${name}.fbs")
endforeach()
add_custom_command(
  OUTPUT ${wire_headers}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/wire"
  COMMAND $<TARGET_FILE:flatbuffers::flatc> --cpp
    -I "${sdk_schema_dir}"
    -I "${CMAKE_CURRENT_SOURCE_DIR}/schemas"
    -o "${CMAKE_CURRENT_BINARY_DIR}/wire"
    ${wire_schemas}
  DEPENDS ${wire_schemas} "${sdk_schema_dir}/tensor.fbs" flatbuffers::flatc
  CODEGEN
  VERBATIM
)
add_library(isaac_messages_cc INTERFACE ${wire_headers})
add_library(isaac::messages ALIAS isaac_messages_cc)
target_include_directories(isaac_messages_cc INTERFACE
  "${CMAKE_CURRENT_BINARY_DIR}/wire"
)
target_link_libraries(isaac_messages_cc INTERFACE
  flatbuffers::flatbuffers
  holoscan::schema
)
if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
