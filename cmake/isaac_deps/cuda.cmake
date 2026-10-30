# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CUDA Toolkit dependency setup.

if(TARGET CUDA::cudart)
  return()
endif()

include(isaac_target_scope)

isaac_get_imported_targets_in_scope(ISAAC_TARGETS_BEFORE)
find_package(CUDAToolkit REQUIRED)
isaac_promote_new_imported_targets(ISAAC_TARGETS_BEFORE)
