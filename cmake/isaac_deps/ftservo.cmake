# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# FTServo dependency setup.

# Already exists?
if(TARGET ftservo)
  return()
endif()

include(isaac_cpm_find_package)

# FTServo declares a policy version below CMake's deprecation floor.
# Limit the policy override to this dependency.
isaac_cpm_find_package(
  NAME ftservo_sdk
  GITHUB_REPOSITORY ftservo/FTServo_Linux
  GIT_TAG 064a6dbf814477b970959b5b01b2fc9bc594afa3
  SOURCE_SUBDIR src
  PATCH_COMMAND git apply --check --reverse ${PROJECT_SOURCE_DIR}/third_party/ftservo/bounded-serial-read.patch
    || git apply ${PROJECT_SOURCE_DIR}/third_party/ftservo/bounded-serial-read.patch
  OPTIONS
    "CMAKE_POLICY_VERSION_MINIMUM 3.10"
)

if(NOT TARGET SCServo)
  message(FATAL_ERROR "FTServo: expected upstream target SCServo after CPM add_subdirectory(src).")
endif()

# Vendor tree uses flat includes; dependents use "ftservo/SCServo.h".
# Create shim symlink for include layout.
set(_ftservo_src "${ftservo_sdk_SOURCE_DIR}/src")
set(_ftservo_include_root "${PROJECT_BINARY_DIR}/_shims/ftservo")
set(_ftservo_shim "${_ftservo_include_root}/ftservo")
if(NOT EXISTS "${_ftservo_shim}")
  file(MAKE_DIRECTORY "${_ftservo_include_root}")
  file(CREATE_LINK "${_ftservo_src}" "${_ftservo_shim}" SYMBOLIC)
endif()

# Create an INTERFACE target that adds the include layout for "ftservo/..." (symlink src -> ftservo).
add_library(ftservo INTERFACE)
target_link_libraries(ftservo INTERFACE SCServo)
target_include_directories(ftservo INTERFACE "${_ftservo_include_root}")
add_library(isaac::ftservo ALIAS ftservo)

message(STATUS "FTServo: upstream SCServo at ${_ftservo_src}; include shim ${_ftservo_shim}")

# Clean up
unset(_ftservo_include_root)
unset(_ftservo_shim)
unset(_ftservo_src)
