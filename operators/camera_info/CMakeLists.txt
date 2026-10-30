# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

add_library(camera_info_op SHARED camera_info_op.cc)
add_library(isaac::operators::camera_info ALIAS camera_info_op)
target_include_directories(camera_info_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(camera_info_op
  PUBLIC
    holoscan::core
    holoscan::sensor_io
    isaac::common::camera_info_cpu_messages
)
