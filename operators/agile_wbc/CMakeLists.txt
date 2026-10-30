# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(agile_wbc_op SHARED
  agile_wbc_op.cu
  wbc_params_validation.cc
)
add_library(isaac::operators::agile_wbc ALIAS agile_wbc_op)

include(isaac_deps/cuda)
include(isaac_deps/holoscan)
include(isaac_deps/matx)
target_include_directories(agile_wbc_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(agile_wbc_op
  PUBLIC
    CUDA::cudart
    holoscan::core
    holoscan::sensor_io
    isaac::common::tensorrt_engine
    isaac::common::geometry_cpu_messages
    isaac::common::joint_cpu_messages
    matx::matx
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
