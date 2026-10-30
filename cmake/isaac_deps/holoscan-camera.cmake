# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Holoscan Camera dependency setup.

set(ISAAC_HOLOSCAN_CAMERA_GIT_TAG v1.0.0)
set(ISAAC_HOLOSCAN_CAMERA_VERSION 1.0.0)


if(TARGET holoscan::v4l2_capture_op)
  return()
endif()

include(isaac_deps/holoscan)
include(isaac_cpm_find_package)
include(isaac_set)

set(SYSTEM_HAS_SIPL OFF)
if(EXISTS "/usr/src/jetson_sipl_api/sipl/include/NvSIPLCamera.hpp")
  set(SYSTEM_HAS_SIPL ON)
endif()
isaac_set(ISAAC_BUILD_SIPL
  DEFAULT ${SYSTEM_HAS_SIPL}
  DESCRIPTION "Build Holoscan Camera SIPL support (API v2)"
)

set(_holoscan_camera_git_repository "https://github.com/nvidia-holoscan/holoscan-camera.git")


isaac_cpm_find_package(
  NAME holoscan_camera
  VERSION ${ISAAC_HOLOSCAN_CAMERA_VERSION}
  GIT_REPOSITORY "${_holoscan_camera_git_repository}"
  GIT_TAG ${ISAAC_HOLOSCAN_CAMERA_GIT_TAG}
  OPTIONS
    "HOLOSCAN_CAMERA_BUILD_ALL OFF"
    "HOLOSCAN_CAMERA_BUILD_TESTING OFF"
    "L4T_MAJ_VER 39" # Select SIPL API v2 in Holoscan Camera.
    "OP_sipl_capture_op ${ISAAC_BUILD_SIPL}"
    "OP_v4l2_capture_op ON"
)



unset(_holoscan_camera_git_repository)
