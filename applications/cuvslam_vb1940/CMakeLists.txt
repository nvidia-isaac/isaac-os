# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This application captures through Holoscan Camera's SIPL operator, which is built only where the
# NvSIPL headers and the L4T user-space libraries are, so ask the dependency rather than the sensor
# bridge backend: the bridge selects a receiver and answers for SIPL only when it also outranks
# FUSA, which has nothing to do with whether this operator exists.
include(isaac_deps/holoscan-camera)
if(NOT TARGET holoscan::sipl_capture_op)
  message(STATUS
    "Skipping cuvslam_vb1940: Holoscan Camera does not provide SIPL capture for the "
    "installed SDK."
  )
  return()
endif()

include(isaac_deps/cli11)
include(isaac_deps/cuda)
include(isaac_deps/holoscan)

add_executable(cuvslam_vb1940 main.cc)
target_include_directories(cuvslam_vb1940 PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(cuvslam_vb1940
  PRIVATE
    CLI11::CLI11
    CUDA::nppidei
    holoscan::config_io
    holoscan::core
    holoscan::visualization::op_holoviz
    holoscan::params
    holoscan::sensor_io
    holoscan::sipl_capture_op
    isaac::common::camera_info_cpu_messages
    isaac::common::geometry_cpu_messages
    isaac::common::pose_tree
    isaac::operators::camera_info
    isaac::operators::cuvslam
    isaac::operators::image_conversion
)
