# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Foxglove SDK dependency setup.
# Downloads the pre-built foxglove-sdk release via CPM.

set(ISAAC_FOXGLOVE_SDK_VERSION 0.14.3)

# Already exists?
if(TARGET isaac::foxglove_sdk)
  return()
endif()

# Select architecture-specific URL and checksum.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(_foxglove_arch "x86_64")
  set(_foxglove_sha256 "68efcf9b2cb2c94d37af77528748e13d3af548340539f70d7afffa2406d3b5e1")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(_foxglove_arch "aarch64")
  set(_foxglove_sha256 "58a8f3ae5898d03344869cb68327e85adf90c909ac9e9b60587773d11d7ebff9")
else()
  message(FATAL_ERROR "Unsupported architecture for foxglove-sdk: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(_foxglove_zip "foxglove-v${ISAAC_FOXGLOVE_SDK_VERSION}-cpp-${_foxglove_arch}-unknown-linux-gnu.zip")
set(_foxglove_url
  "https://github.com/foxglove/foxglove-sdk/releases/download/sdk%2Fv${ISAAC_FOXGLOVE_SDK_VERSION}/${_foxglove_zip}"
)

isaac_cpm_find_package(
  NAME foxglove-sdk
  VERSION ${ISAAC_FOXGLOVE_SDK_VERSION}
  URL "${_foxglove_url}"
  URL_HASH "SHA256=${_foxglove_sha256}"
  DOWNLOAD_ONLY TRUE
)

unset(_foxglove_arch)
unset(_foxglove_sha256)
unset(_foxglove_zip)
unset(_foxglove_url)

set(FOXGLOVE_SDK_ROOT "${foxglove-sdk_SOURCE_DIR}")

message(STATUS "Found foxglove-sdk at: ${FOXGLOVE_SDK_ROOT}")

# libfoxglove.so ships without a SONAME. Without one, the linker embeds the
# full build-time path as DT_NEEDED, which breaks runtime loading. Patch it
# once so consumers only record the bare soname "libfoxglove.so".
set(_foxglove_lib "${FOXGLOVE_SDK_ROOT}/lib/libfoxglove.so")
find_program(_patchelf patchelf REQUIRED)
execute_process(
  COMMAND ${_patchelf} --set-soname libfoxglove.so "${_foxglove_lib}"
  RESULT_VARIABLE _patchelf_result
)
if(NOT _patchelf_result EQUAL 0)
  message(FATAL_ERROR "patchelf failed to set SONAME on ${_foxglove_lib}")
endif()
unset(_foxglove_lib)
unset(_patchelf)
unset(_patchelf_result)

# Create imported target for foxglove C library
add_library(foxglove_c SHARED IMPORTED GLOBAL)
set_target_properties(foxglove_c PROPERTIES
  IMPORTED_LOCATION "${FOXGLOVE_SDK_ROOT}/lib/libfoxglove.so"
  INTERFACE_INCLUDE_DIRECTORIES "${FOXGLOVE_SDK_ROOT}/include"
)

# Collect C++ source files
file(GLOB_RECURSE FOXGLOVE_CPP_SOURCES "${FOXGLOVE_SDK_ROOT}/src/*.cpp")
file(GLOB_RECURSE FOXGLOVE_CPP_HEADERS "${FOXGLOVE_SDK_ROOT}/include/foxglove/*.hpp")

# Create C++ library target
add_library(foxglove_sdk SHARED ${FOXGLOVE_CPP_SOURCES} ${FOXGLOVE_CPP_HEADERS})
set_target_properties(foxglove_sdk PROPERTIES
  BUILD_RPATH "${FOXGLOVE_SDK_ROOT}/lib"
)
target_include_directories(foxglove_sdk
  PUBLIC
    "${FOXGLOVE_SDK_ROOT}/include"
)
target_link_libraries(foxglove_sdk
  PUBLIC
    foxglove_c
)

# Create alias targets matching the expected naming convention
add_library(isaac::foxglove_c ALIAS foxglove_c)
add_library(isaac::foxglove_sdk ALIAS foxglove_sdk)

# Install the built C++ wrapper and its prebuilt C runtime together.
isaac_install(
  COMPONENT DEP_foxglove_sdk
  TARGETS
    foxglove_c
    foxglove_sdk
)
