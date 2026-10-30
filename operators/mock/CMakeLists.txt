# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

add_library(mock_joint_state_op SHARED mock_joint_state_op.cc)
add_library(isaac::operators::mock::joint_state ALIAS mock_joint_state_op)
target_include_directories(mock_joint_state_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mock_joint_state_op
  PUBLIC
    holoscan::core
    holoscan::sensor_io
    isaac::common::joint_cpu_messages
)

add_library(mock_sink_op SHARED mock_sink_op.cc)
add_library(isaac::operators::mock::sink ALIAS mock_sink_op)
target_include_directories(mock_sink_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mock_sink_op
  PUBLIC
    holoscan::core
    isaac::common::joint_cpu_messages
)
