# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOHUB_CMAKE_PATH)
  include(isaac_cpm_find_package)
  isaac_cpm_find_package(
    NAME holohub_config_helper
    URL https://raw.githubusercontent.com/nvidia-holoscan/holohub/holoscan-sdk-4.6.0/cmake/HoloHubConfigHelpers.cmake
    URL_HASH SHA256=3888a356182c6e8fc25754f8adeac8d6eb92d7165c7c3a741691db0a3cc1b802
    DOWNLOAD_NO_EXTRACT TRUE
    DOWNLOAD_ONLY TRUE
  )
  set(HOLOHUB_CMAKE_PATH "${holohub_config_helper_SOURCE_DIR}")
endif()

list(APPEND CMAKE_MODULE_PATH "${HOLOHUB_CMAKE_PATH}")
include("${HOLOHUB_CMAKE_PATH}/HoloHubConfigHelpers.cmake")
message(STATUS "Loaded HoloHub configuration helpers from: ${HOLOHUB_CMAKE_PATH}")
