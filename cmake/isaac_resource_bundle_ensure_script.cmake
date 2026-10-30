# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED ISAAC_ADD_RESOURCE_DIR OR "${ISAAC_ADD_RESOURCE_DIR}" STREQUAL "")
  message(FATAL_ERROR "ISAAC_ADD_RESOURCE_DIR is required.")
endif()
list(PREPEND CMAKE_MODULE_PATH "${ISAAC_ADD_RESOURCE_DIR}")

include(isaac_resource_bundle_helpers)

isaac_resource_bundle_ensure(
  BUNDLE_LABEL "${BUNDLE_LABEL}"
  CACHE_DIR "${CACHE_DIR}"
  LINK_DIR "${LINK_DIR}"
  LOCK_TIMEOUT "${LOCK_TIMEOUT}"
  MANIFEST_FILE "${MANIFEST_FILE}"
)
