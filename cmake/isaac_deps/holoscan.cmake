# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Holoscan SDK dependency setup.

if(TARGET holoscan::core)
  return()
endif()

set(ISAAC_HOLOSCAN_VERSION 5.0)

include(isaac_target_scope)

isaac_get_imported_targets_in_scope(ISAAC_TARGETS_BEFORE)
find_package(Holoscan ${ISAAC_HOLOSCAN_VERSION} REQUIRED CONFIG)
string(REGEX MATCH "^[0-9]+" ISAAC_HOLOSCAN_FOUND_MAJOR "${Holoscan_VERSION}")
if(NOT ISAAC_HOLOSCAN_FOUND_MAJOR STREQUAL "5")
  message(FATAL_ERROR "Expected Holoscan major 5, got ${Holoscan_VERSION}")
endif()
isaac_promote_new_imported_targets(ISAAC_TARGETS_BEFORE)

# Work around fmt target conflicts on platforms where Holoscan defines only fmt::fmt-header-only.
if(TARGET fmt::fmt-header-only AND NOT TARGET fmt::fmt)
  add_library(fmt::fmt ALIAS fmt::fmt-header-only)
endif()
