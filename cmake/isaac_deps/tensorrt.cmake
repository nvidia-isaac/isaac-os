# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TensorRT dependency setup. Creates imported targets TensorRT::nvinfer,
# TensorRT::nvinfer_plugin, and TensorRT::nvonnxparser from the libraries that
# ship in the Holoscan base image. Headers and libs live in the standard
# multiarch system paths.

if(TARGET TensorRT::nvonnxparser)
  return()
endif()

find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h REQUIRED
  PATHS /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu)
mark_as_advanced(TensorRT_INCLUDE_DIR)

foreach(_trt_lib nvinfer nvinfer_plugin nvonnxparser)
  find_library(TensorRT_${_trt_lib}_LIBRARY NAMES ${_trt_lib} REQUIRED)
  mark_as_advanced(TensorRT_${_trt_lib}_LIBRARY)
  add_library(TensorRT::${_trt_lib} SHARED IMPORTED GLOBAL)
  set_target_properties(TensorRT::${_trt_lib} PROPERTIES
    IMPORTED_LOCATION "${TensorRT_${_trt_lib}_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
endforeach()
