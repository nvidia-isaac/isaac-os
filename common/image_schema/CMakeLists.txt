# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

add_library(image_schema SHARED
  image_schema.cc
)
add_library(isaac::common::image_schema ALIAS image_schema)

target_include_directories(image_schema PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(image_schema
  PUBLIC
    holoscan::core
    holoscan::sensor_io
  PRIVATE
    # PRIVATE: fmt::format() is only called from image_schema.cc, not from any header this
    # target installs, so consumers don't need it on their own compile.
    fmt::fmt
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
