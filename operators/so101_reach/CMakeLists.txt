# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(so101_reach_fused_op SHARED
  so101_forward_kinematics.cc
  so101_reach_fused_op.cu
)
add_library(isaac::operators::so101_reach ALIAS so101_reach_fused_op)

include(isaac_deps/cuda)
include(isaac_deps/holoscan)
include(isaac_deps/tensorrt)
target_include_directories(so101_reach_fused_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(so101_reach_fused_op
  PUBLIC
    holoscan::core
    isaac::common::tensorrt_engine
    isaac::common::so101_cpu_messages
  PRIVATE
    CUDA::cudart
    TensorRT::nvinfer
)
set_target_properties(so101_reach_fused_op PROPERTIES
  CUDA_ARCHITECTURES native
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
