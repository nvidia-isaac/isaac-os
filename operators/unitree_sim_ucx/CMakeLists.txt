# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


include(isaac_deps/cuda)

add_library(unitree_sim_ucx_op SHARED
  synthetic_clock_advance_op.cc
  unitree_sim_ucx_ops_adapter.cc
)
add_library(isaac::operators::unitree_sim_ucx ALIAS unitree_sim_ucx_op)
target_include_directories(unitree_sim_ucx_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(unitree_sim_ucx_op PUBLIC
  holoscan::core
  holoscan::schema
  holoscan::sensor_io
  isaac::common::joint_cpu_messages
  isaac::operators::ucxx_ops
)
target_link_libraries(unitree_sim_ucx_op PRIVATE CUDA::cudart)
