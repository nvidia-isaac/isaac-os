# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(ISAAC_NLOHMANN_JSON_VERSION 3.10.5)

if(TARGET nlohmann_json::nlohmann_json)
  return()
endif()

include(isaac_target_scope)

isaac_get_imported_targets_in_scope(ISAAC_TARGETS_BEFORE)
find_package(nlohmann_json ${ISAAC_NLOHMANN_JSON_VERSION} REQUIRED CONFIG)
isaac_promote_new_imported_targets(ISAAC_TARGETS_BEFORE)

message(STATUS "Found nlohmann_json: ${nlohmann_json_DIR} (found suitable version \"${nlohmann_json_VERSION}\", minimum required is \"${ISAAC_NLOHMANN_JSON_VERSION}\")")
