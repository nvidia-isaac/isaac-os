# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Flatbuffers dependency setup.

set(ISAAC_FLATBUFFERS_VERSION 25.12)
set(ISAAC_FLATBUFFERS_VERSION_EXACT 25.12.19)

if(TARGET flatbuffers::flatbuffers)
  return()
endif()

include(isaac_cpm_find_package)

isaac_cpm_find_package(
  NAME Flatbuffers
  VERSION ${ISAAC_FLATBUFFERS_VERSION}
  GITHUB_REPOSITORY google/flatbuffers
  GIT_TAG v${ISAAC_FLATBUFFERS_VERSION_EXACT}
  OPTIONS
    "FLATBUFFERS_BUILD_TESTS OFF"
    "FLATBUFFERS_BUILD_FLATHASH OFF"
    "FLATBUFFERS_BUILD_FLATC ON"
    "FLATBUFFERS_BUILD_FLATLIB ON"
    "FLATBUFFERS_STATIC_FLATC ON"
    "FLATBUFFERS_STRICT_MODE ON"
)

# GCC 13 emits a false-positive -Wstringop-overflow inside
# flatbuffers::AddFlatBuffer (src/reflection.cpp). Suppress it only on the
# vendored flatbuffers/flatc targets, only for GNU, without touching the
# vendored source. PRIVATE scope keeps the flag off consuming targets.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  foreach(_fb_target flatbuffers flatc)
    if(TARGET ${_fb_target})
      target_compile_options(${_fb_target} PRIVATE -Wno-stringop-overflow)
    endif()
  endforeach()
endif()

# Flatbuffers' kotlin code generators have duplicate symbols in anonymous
# namespaces that break unity builds.
if(TARGET flatc)
  set_target_properties(flatc PROPERTIES UNITY_BUILD OFF)
endif()

# When built from source via CPM, targets are "flatbuffers" and "flatc" instead
# of "flatbuffers::flatbuffers" and "flatbuffers::flatc". Create namespaced
# aliases so consumers don't need to know the provenance.
if(TARGET flatbuffers AND NOT TARGET flatbuffers::flatbuffers)
  add_library(flatbuffers::flatbuffers ALIAS flatbuffers)
endif()
if(TARGET flatc AND NOT TARGET flatbuffers::flatc)
  add_executable(flatbuffers::flatc ALIAS flatc)
endif()
