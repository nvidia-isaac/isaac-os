# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cuda)
include(isaac_deps/holoscan)
include(isaac_deps/tensorrt)

add_library(tensorrt_engine SHARED tensorrt_engine.cc)
add_library(isaac::common::tensorrt_engine ALIAS tensorrt_engine)

target_include_directories(tensorrt_engine PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(tensorrt_engine
  PUBLIC
    CUDA::cudart
    holoscan::core
    TensorRT::nvinfer
  PRIVATE
    TensorRT::nvinfer_plugin
    TensorRT::nvonnxparser
)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
