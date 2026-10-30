# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CPM.cmake bootstrapper and wrapper.
#
# Loads CPM from CPM_CMAKE_FILE (set via -D or Docker ENV) or downloads as fallback.
# Provides isaac_cpm_find_package() which wraps CPMFindPackage  with some default
# arguments and promotes any new imported targets to global scope.

if(COMMAND isaac_cpm_find_package)
  return()
endif()

# -- Parameters -- #

# CPM.cmake version
set(ISAAC_CPM_VERSION 0.42.1)

# Defaults applied to every CPMAddPackage call unless the caller specifies them.
# CPM already defaults to shallow clones except when GIT_TAG is a commit hash.
# Format: KEY VALUE pairs.
set(ISAAC_CPM_DEFAULTS
  EXCLUDE_FROM_ALL       TRUE
  FIND_PACKAGE_ARGUMENTS "CONFIG"
  GIT_CONFIG             "core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=15"
  SYSTEM                 TRUE
  CACHE INTERNAL "Default arguments applied to Isaac CPM packages" FORCE
)

# -- Load CPM.cmake -- #

# CPM reads CPM_SOURCE_CACHE from the environment or CMake cache after include.
# Do not pass CPM_SOURCE_CACHE as a CPMAddPackage argument: it is not a parsed keyword in
# CPM 0.42.x and ends up appended to PATCH_COMMAND, breaking git apply.
include(isaac_set)
isaac_set(CPM_SOURCE_CACHE
  DEFAULT "${PROJECT_SOURCE_DIR}/.cache/cpm"
  TYPE PATH
  DESCRIPTION "CPM source cache dir"
)

# Only load once.
if(NOT COMMAND CPMAddPackage)
  # Resolve CPM_CMAKE_FILE: -D takes precedence over ENV.
  if(NOT DEFINED CPM_CMAKE_FILE AND DEFINED ENV{CPM_CMAKE_FILE} AND NOT "$ENV{CPM_CMAKE_FILE}" STREQUAL "")
    set(CPM_CMAKE_FILE "$ENV{CPM_CMAKE_FILE}")
  endif()

  # Use provided path if set, otherwise download.
  if(DEFINED CPM_CMAKE_FILE AND NOT CPM_CMAKE_FILE STREQUAL "")
    if(NOT EXISTS "${CPM_CMAKE_FILE}")
      message(FATAL_ERROR "CPM_CMAKE_FILE is set to '${CPM_CMAKE_FILE}' but the file does not exist.")
    endif()
    set(_cpm_cmake_file "${CPM_CMAKE_FILE}")
  else()
    message(STATUS "CPM_CMAKE_FILE not set, downloading CPM.cmake v${ISAAC_CPM_VERSION}")
    set(_cpm_cmake_file "${PROJECT_BINARY_DIR}/cmake/CPM_${ISAAC_CPM_VERSION}.cmake")
    if(NOT EXISTS "${_cpm_cmake_file}")
      file(DOWNLOAD
        "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${ISAAC_CPM_VERSION}/CPM.cmake"
        "${_cpm_cmake_file}"
        STATUS _cpm_download_status
      )
      list(GET _cpm_download_status 0 _cpm_download_code)
      if(NOT _cpm_download_code EQUAL 0)
        list(GET _cpm_download_status 1 _cpm_download_error)
        file(REMOVE "${_cpm_cmake_file}")
        message(FATAL_ERROR "Failed to download CPM.cmake v${ISAAC_CPM_VERSION}: ${_cpm_download_error}")
      endif()
    endif()
  endif()

  include("${_cpm_cmake_file}")

  # Validate that CPMAddPackage is available.
  if(NOT COMMAND CPMAddPackage)
    message(FATAL_ERROR "Failed to load CPM.cmake from ${_cpm_cmake_file} — CPMAddPackage command not found.")
  endif()
endif()

# -- isaac_cpm_find_package -- #

# Wrapper around CPMFindPackage that:
# - Applies default arguments from ISAAC_CPM_DEFAULTS (callers can override any of these)
# - Promotes new imported targets to global scope
# All arguments are forwarded to CPMFindPackage (explicit values override defaults).
# macro (not function) so that variables set by CPMFindPackage (e.g.
# CPM_PACKAGE_FOUND, <name>_SOURCE_DIR) propagate to the caller's scope.
include(isaac_target_scope)

macro(isaac_cpm_find_package)
  set(_isaac_cpm_args ${ARGN})

  # Apply defaults: iterate KEY VALUE pairs, append if key is absent.
  # Keep the defaults in the cache because CPM may unset normal-scope variables
  # while parsing a package. The wrapper is called repeatedly in one configure.
  set(_isaac_cpm_defaults ${ISAAC_CPM_DEFAULTS})
  list(LENGTH _isaac_cpm_defaults _len)
  math(EXPR _last "${_len} - 1")
  foreach(_key_index RANGE 0 ${_last} 2)
    list(GET _isaac_cpm_defaults ${_key_index} _key)
    if(NOT "${_key}" IN_LIST _isaac_cpm_args)
      math(EXPR _value_index "${_key_index} + 1")
      list(GET _isaac_cpm_defaults ${_value_index} _value)
      list(APPEND _isaac_cpm_args ${_key} "${_value}")
    endif()
  endforeach()

  # Expose imported targets to the GLOBAL scope.
  isaac_get_imported_targets_in_scope(_targets_before)
  CPMFindPackage(${_isaac_cpm_args})
  isaac_promote_new_imported_targets(_targets_before)

  # Clean up macro-scoped temporaries to avoid polluting the caller's scope.
  unset(_isaac_cpm_args)
  unset(_len)
  unset(_last)
  unset(_isaac_cpm_defaults)
  unset(_key_index)
  unset(_key)
  unset(_targets_before)
  unset(_value_index)
  unset(_value)
endmacro()
