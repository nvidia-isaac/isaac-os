# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(TARGET matx::matx)
  return()
endif()

include(isaac_target_scope)

isaac_get_imported_targets_in_scope(ISAAC_TARGETS_BEFORE)
find_package(matx 1.0.0 REQUIRED CONFIG)
isaac_promote_new_imported_targets(ISAAC_TARGETS_BEFORE)
