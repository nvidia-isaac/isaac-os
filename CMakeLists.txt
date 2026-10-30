# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Root CMake
cmake_minimum_required(VERSION 4.0)

# Set CUDA architectures before project() call.
# MatX requires compute capability 6.0 or newer.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
   message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting it to `native`")
   set(CMAKE_CUDA_ARCHITECTURES "native")
else()
   message(STATUS "Requested CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

endif()

project(isaac_os LANGUAGES C CXX CUDA)

##############################################################################
# Global properties (CMAKE_*)
##############################################################################

# Standard install destinations. Explicit target RPATHs keep installed runtime
# artifacts independent of build and dependency-cache paths.
include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)

# Helper modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(isaac_install)

##############################################################################
# Options
##############################################################################

include(isaac_set)
isaac_set(ISAAC_BUILD_TESTS DEFAULT ON DESCRIPTION "Build tests")

# BUILD_ALL: Global flag to enable all applications, operators, and common libraries.
# Used by HoloHub cmake helpers (add_holohub_application, add_holohub_operator, etc.)
# When ON, all APP_*, OP_*, LIB_*, EXT_* options default to ON.
isaac_set(BUILD_ALL DEFAULT OFF DESCRIPTION "Build all applications, operators, and common libraries")

# NVIDIA Video Codec SDK build option
include(isaac_check_nvenc)
isaac_check_nvenc(SYSTEM_HAS_NVENC)
isaac_set(ISAAC_ENABLE_NV_VIDEO_SDK
  DEFAULT ${SYSTEM_HAS_NVENC}
  DESCRIPTION "Enable NVIDIA Video Codec SDK support"
)

##############################################################################
# Build helpers
##############################################################################

# Include app-specific config helpers
include(isaac_application_helpers)

# Compiler options
include(isaac_compile_link_options)

##############################################################################
# SDK source graph
##############################################################################

if(ISAAC_BUILD_TESTS)
  enable_testing()
  message(STATUS "CTest enabled: registering tests.")
  include(isaac_deps/gtest)
  add_subdirectory(cmake/tests)
  add_subdirectory(tools/tests)
endif()

# Keep the dependency-propagation order explicit.
add_subdirectory(messages)
add_subdirectory(applications)
add_subdirectory(operators)
add_subdirectory(common)
