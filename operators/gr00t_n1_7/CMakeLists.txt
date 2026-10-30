# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(gr00t_n1_7_fused_op SHARED
  gr00t_n1_7_fused_op.cu
)
add_library(isaac::operators::gr00t_n1_7 ALIAS gr00t_n1_7_fused_op)

include(isaac_deps/cuda)
include(isaac_deps/gr00t_n1_7_models)
include(isaac_deps/holoscan)
include(isaac_deps/matx)
include(isaac_deps/tensorrt)
target_include_directories(gr00t_n1_7_fused_op PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gr00t_n1_7_fused_op
  PRIVATE
    CUDA::cuda_driver
  PUBLIC
    CUDA::cudart
    CUDA::curand
    holoscan::core
    isaac::common::tensorrt_engine
    isaac::common::joint_command_chunk_cpu
    isaac::common::joint_cpu_messages
    matx::matx
    TensorRT::nvinfer
)
# Preserve an explicit configure-time architecture even when SDK imports shadow the variable.
get_property(_gr00t_cuda_architectures CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY VALUE)
if(NOT _gr00t_cuda_architectures)
  set(_gr00t_cuda_architectures native)
endif()
set_target_properties(gr00t_n1_7_fused_op PROPERTIES
  CUDA_ARCHITECTURES "${_gr00t_cuda_architectures}"
)
unset(_gr00t_cuda_architectures)
set_source_files_properties(gr00t_n1_7_fused_op.cu PROPERTIES
  COMPILE_OPTIONS "-Xcompiler=-Wno-pedantic"
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
