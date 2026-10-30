# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NVIDIA Decode Driver dependency setup.

# Already exists?
if(TARGET nv::decode_driver)
  return()
endif()

# Find nvidia-decode library
find_library(NVIDIA_DECODE_LIBRARY
    NAMES
        nvcuvid
        libnvcuvid.so.1 # nvidia container toolkit might not mount unversioned .so
    PATHS
        /opt/nvidia/l4t-gpu-libs/openrm # L4T 39.2.1 driver location
    PATH_SUFFIXES
        nvidia                # for Tegra
    DOC "NVIDIA Video Decode library"
    REQUIRED
)

add_library(nv::decode_driver SHARED IMPORTED GLOBAL)
set_property(TARGET nv::decode_driver PROPERTY IMPORTED_LOCATION ${NVIDIA_DECODE_LIBRARY})

message(STATUS "Found NVIDIA Video Decode driver library: ${NVIDIA_DECODE_LIBRARY}")
