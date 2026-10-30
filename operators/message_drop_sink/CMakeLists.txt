# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

add_library(message_drop_sink_op SHARED message_drop_sink_op.cc)
add_library(isaac::operators::message_drop_sink ALIAS message_drop_sink_op)
target_include_directories(message_drop_sink_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(message_drop_sink_op
  PUBLIC
    holoscan::core
    holoscan::sensor_io
)
