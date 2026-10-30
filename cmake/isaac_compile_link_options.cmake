# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


include(CheckLinkerFlag)

##############################################################################
# Build Types
##############################################################################

# All build types
set(_CMAKE_FASTBUILD_BUILD_TYPE "FastBuild")
set(_CMAKE_DEFAULT_BUILD_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel")
set(_CMAKE_BUILD_TYPES "${_CMAKE_FASTBUILD_BUILD_TYPE};${_CMAKE_DEFAULT_BUILD_TYPES}")
message(STATUS "Available build types: ${_CMAKE_BUILD_TYPES}")

# Set the CMake configuration types if it's a multi-config generator.
get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(IS_MULTI_CONFIG)
    set(CMAKE_CONFIGURATION_TYPES ${_CMAKE_BUILD_TYPES} CACHE STRING "Available build types" FORCE)
endif()

# Set the default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type. Empty for default." FORCE)
endif()
if(DEFINED CACHE{CMAKE_BUILD_TYPE})
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${_CMAKE_BUILD_TYPES})
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

##############################################################################
# FastBuild Build Type Configuration
##############################################################################

# Compiler Flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(_FASTBUILD_DEBUG_FLAG "-gmlt")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(_FASTBUILD_DEBUG_FLAG "-g1")
else()
  set(_FASTBUILD_DEBUG_FLAG "-g")
endif()
set(_CMAKE_COMPILE_FLAGS_FASTBUILD "${_FASTBUILD_DEBUG_FLAG} -O0 -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_FASTBUILD "${_CMAKE_COMPILE_FLAGS_FASTBUILD}" CACHE STRING "Compiler flags for FastBuild" FORCE)
set(CMAKE_C_FLAGS_FASTBUILD "${_CMAKE_COMPILE_FLAGS_FASTBUILD}" CACHE STRING "Compiler flags for FastBuild" FORCE)
set(CMAKE_CUDA_FLAGS_FASTBUILD "-g -O0" CACHE STRING "NVCC flags for FastBuild" FORCE)

# Linker Flags
set(_CMAKE_LINKER_FLAGS_FASTBUILD "-Wl,--build-id,--hash-style=gnu")
set(CMAKE_EXE_LINKER_FLAGS_FASTBUILD "${_CMAKE_LINKER_FLAGS_FASTBUILD}" CACHE STRING "Linker flags for FastBuild" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_FASTBUILD "${_CMAKE_LINKER_FLAGS_FASTBUILD}" CACHE STRING "Linker flags for FastBuild" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_FASTBUILD "${_CMAKE_LINKER_FLAGS_FASTBUILD}" CACHE STRING "Linker flags for FastBuild" FORCE)

# Map for imported targets with no FastBuild configuration as fallbacks
# Note: last ; is to allow imported targets with no explicit configuration (IMPORTED_LOCATION)
set(CMAKE_MAP_IMPORTED_CONFIG_FASTBUILD "${_CMAKE_DEFAULT_BUILD_TYPES};NONE;" CACHE STRING "Fallback build types for imported targets with no FastBuild configuration" FORCE)

# Mark as advanced to hide from GUI
mark_as_advanced(
  CMAKE_CXX_FLAGS_FASTBUILD
  CMAKE_C_FLAGS_FASTBUILD
  CMAKE_CUDA_FLAGS_FASTBUILD
  CMAKE_EXE_LINKER_FLAGS_FASTBUILD
  CMAKE_STATIC_LINKER_FLAGS_FASTBUILD
  CMAKE_SHARED_LINKER_FLAGS_FASTBUILD
  CMAKE_MODULE_LINKER_FLAGS_FASTBUILD
)

##############################################################################
# Faster linkers
##############################################################################

# Respect explicit linker requests; otherwise prefer the first available alternative.
if(NOT CMAKE_LINKER_TYPE)
  foreach(linker IN ITEMS MOLD LLD GOLD)
    set(CMAKE_LINKER_TYPE "${linker}")
    check_linker_flag(CXX "" ${linker}_FOUND)
    if(${linker}_FOUND)
      break()
    endif()
    set(CMAKE_LINKER_TYPE DEFAULT)
  endforeach()
endif()
message(STATUS "CMAKE_LINKER_TYPE: ${CMAKE_LINKER_TYPE}")

##############################################################################
# Unity builds for fast compilation
##############################################################################

if(CMAKE_BUILD_TYPE MATCHES "^(FastBuild|Debug)$")
  set(CMAKE_UNITY_BUILD ON)
  set(CMAKE_UNITY_BUILD_BATCH_SIZE 16)
  message(STATUS "${CMAKE_BUILD_TYPE}: Unity builds enabled")
endif()

##############################################################################
# Reduce Debug Info memory footprint
##############################################################################
if(CMAKE_BUILD_TYPE MATCHES "^(FastBuild|Debug|RelWithDebInfo)$")
  # Split DWARF: Less memory, faster build, separate debug info
  add_compile_options(-gsplit-dwarf)
  message(STATUS "${CMAKE_BUILD_TYPE}: Split DWARF enabled")
endif()


# GDB Index: Slower link, faster GDB startup
if(CMAKE_BUILD_TYPE MATCHES "^(Debug|RelWithDebInfo)$")
  check_linker_flag(CXX "LINKER:--gdb-index" HAS_GDB_INDEX_${CMAKE_LINKER_TYPE})
  if(HAS_GDB_INDEX_${CMAKE_LINKER_TYPE})
    add_link_options(LINKER:--gdb-index)
    message(STATUS "${CMAKE_BUILD_TYPE}: GDB Index enabled")
  endif()
endif()

##############################################################################
# CXX Standard
##############################################################################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

##############################################################################
# Position Independent Code
##############################################################################

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

##############################################################################
# Export
##############################################################################

# This generates a 'compile_commands.json' file which can be read by VSCode to
# configure the include paths for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
