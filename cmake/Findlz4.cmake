# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Find module for lz4 (system package: liblz4-dev).
#
# liblz4-dev ships headers, the shared library and a pkg-config file, but no CMake package
# config. Third-party projects we build from source call `find_package(lz4)` and link
# `LZ4::lz4` (see mcap_builder), so provide the module they expect. `cmake/` is on
# CMAKE_MODULE_PATH, which subdirectories added by CPM inherit.
#
# Result variables:
#   lz4_FOUND        - whether liblz4 was found
#   lz4_VERSION      - liblz4 version, parsed from lz4.h
#   lz4_INCLUDE_DIR  - directory holding lz4.h
#   lz4_LIBRARY      - path to the liblz4 library
#
# Imported target:
#   LZ4::lz4

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(ISAAC_PC_LZ4 QUIET liblz4)
endif()

find_path(lz4_INCLUDE_DIR
    NAMES lz4frame.h
    HINTS ${ISAAC_PC_LZ4_INCLUDE_DIRS}
    DOC "liblz4 headers"
)
find_library(lz4_LIBRARY
    NAMES lz4
    HINTS ${ISAAC_PC_LZ4_LIBRARY_DIRS}
    DOC "liblz4 library"
)

if(lz4_INCLUDE_DIR AND EXISTS "${lz4_INCLUDE_DIR}/lz4.h")
  file(STRINGS "${lz4_INCLUDE_DIR}/lz4.h" _lz4_version_lines
      REGEX "^#define +LZ4_VERSION_(MAJOR|MINOR|RELEASE) +[0-9]+"
  )
  foreach(_component MAJOR MINOR RELEASE)
    string(REGEX MATCH "LZ4_VERSION_${_component} +([0-9]+)" _match "${_lz4_version_lines}")
    set(_lz4_${_component} "${CMAKE_MATCH_1}")
  endforeach()
  if(_lz4_MAJOR AND _lz4_MINOR AND _lz4_RELEASE)
    set(lz4_VERSION "${_lz4_MAJOR}.${_lz4_MINOR}.${_lz4_RELEASE}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(lz4
    REQUIRED_VARS lz4_LIBRARY lz4_INCLUDE_DIR
    VERSION_VAR lz4_VERSION
)

if(lz4_FOUND AND NOT TARGET LZ4::lz4)
  add_library(LZ4::lz4 UNKNOWN IMPORTED GLOBAL)
  set_target_properties(LZ4::lz4 PROPERTIES
      IMPORTED_LOCATION ${lz4_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${lz4_INCLUDE_DIR}
  )
endif()

# mcap_builder reports the package location as `lz4_DIR`, which only config mode would set.
set(lz4_DIR "${lz4_INCLUDE_DIR}")

mark_as_advanced(lz4_INCLUDE_DIR lz4_LIBRARY)
