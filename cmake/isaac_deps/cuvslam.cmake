# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cuVSLAM dependency setup.
#
# Tries find_package(cuVSLAM CONFIG) first, then falls back to downloading
# the prebuilt C++ release from GitHub.

set(CUVSLAM_VERSION "17.0.1" CACHE STRING "cuVSLAM version")

# Already exists?
if(TARGET cuvslam)
  return()
endif()

# Find from CMake config (e.g. system install).
include(isaac_target_scope)

isaac_get_imported_targets_in_scope(ISAAC_TARGETS_BEFORE)
find_package(cuVSLAM ${CUVSLAM_VERSION} QUIET CONFIG)
isaac_promote_new_imported_targets(ISAAC_TARGETS_BEFORE)

if(cuVSLAM_FOUND)
  message(STATUS "Found cuVSLAM: ${cuVSLAM_DIR} (version \"${cuVSLAM_VERSION}\")")
  return()
endif()

# ---------------------------------------------------------------------------
# Fallback: download prebuilt archive from GitHub release.
# ---------------------------------------------------------------------------

# Determine CUDA major version.
find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED CUDAToolkit_VERSION_MAJOR)
  message(FATAL_ERROR "Could not determine CUDA major version from CUDAToolkit.")
endif()
set(_CUDA_MAJOR "${CUDAToolkit_VERSION_MAJOR}")

# Normalize CPU architecture.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _proc_lower)
if(_proc_lower MATCHES "^(aarch64|arm64|armv8)")
  set(_CPU_ARCH "aarch64")
elseif(_proc_lower MATCHES "^(x86_64|amd64)")
  set(_CPU_ARCH "x86_64")
else()
  message(FATAL_ERROR "Unsupported CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}.")
endif()

# Prefer Ubuntu 22.04 builds where available for wider glibc compatibility.
if(_CPU_ARCH STREQUAL "aarch64" AND _CUDA_MAJOR EQUAL 13)
  set(_CUVSLAM_PLATFORM "thor-cuda13.0.1-ubuntu24.04")
  set(_CUVSLAM_SHA256 "e5ee5e47e8812b04755232d60949dd62bd86619b53dac038ba5941e91878913e")
elseif(_CPU_ARCH STREQUAL "x86_64" AND _CUDA_MAJOR EQUAL 13)
  set(_CUVSLAM_PLATFORM "x86_64-cuda13.2.0-ubuntu22.04")
  set(_CUVSLAM_SHA256 "9d00ae47aacff37da9ffb069f343b4b1704f633aa1af3d2efc7f96932012db9b")
else()
  message(FATAL_ERROR "Unsupported cuVSLAM platform: ${_CPU_ARCH} with CUDA ${_CUDA_MAJOR}.")
endif()
set(CUVSLAM_SHA256 "${_CUVSLAM_SHA256}" CACHE STRING "cuVSLAM archive SHA256")

# Download and extract the platform archive at configure time.
include(isaac_add_resource)
isaac_add_resource(cuvslam
  URL "https://github.com/nvidia-isaac/cuVSLAM/releases/download/v${CUVSLAM_VERSION}/cuvslam-cpp-${CUVSLAM_VERSION}-${_CUVSLAM_PLATFORM}.tar.gz"
  EXPECTED_HASH "SHA256=${CUVSLAM_SHA256}"
  CONFIG_TIME
)
# cuvslam_CONTENT_DIR is now set to ${ISAAC_RESOURCE_CACHE}/cuvslam/content

# Find headers.
unset(CUVSLAM_INCLUDE_DIRS CACHE)
find_path(CUVSLAM_INCLUDE_DIRS
  NAMES cuvslam/cuvslam2.h cuvslam/cuvslam_gpu.h
  HINTS "${cuvslam_CONTENT_DIR}"
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  REQUIRED
)

# Find library.
unset(CUVSLAM_LIBRARY CACHE)
find_library(CUVSLAM_LIBRARY
  NAMES cuvslam
  HINTS "${cuvslam_CONTENT_DIR}/bin"
  NO_DEFAULT_PATH
  REQUIRED
)

message(STATUS "Found cuVSLAM library: '${CUVSLAM_LIBRARY}'")
message(STATUS "Found cuVSLAM headers: '${CUVSLAM_INCLUDE_DIRS}'")

# Create imported target.
add_library(cuvslam SHARED IMPORTED GLOBAL)
set_target_properties(cuvslam PROPERTIES
  IMPORTED_LOCATION "${CUVSLAM_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${CUVSLAM_INCLUDE_DIRS}"
)
