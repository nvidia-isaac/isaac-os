# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NVIDIA Encode Driver dependency setup.

# Already exists?
if(TARGET nv::encode_driver)
  return()
endif()

# Find nvidia-encode library
find_library(NVIDIA_ENCODE_LIBRARY
    NAMES
        nvidia-encode
        libnvidia-encode.so.1 # nvidia container toolkit might not mount unversioned .so
    PATH_SUFFIXES
        nvidia                # for Tegra
    DOC "NVIDIA Video Encode library"
    REQUIRED
)

add_library(nv::encode_driver SHARED IMPORTED GLOBAL)
set_property(TARGET nv::encode_driver PROPERTY IMPORTED_LOCATION ${NVIDIA_ENCODE_LIBRARY})

message(STATUS "Found NVIDIA Video Encode driver library: ${NVIDIA_ENCODE_LIBRARY}")
