# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Eigen dependency setup.

set(ISAAC_EIGEN3_VERSION 3.3)

if(TARGET Eigen3::Eigen)
  return()
endif()

# Eigen is bundled with the Holoscan SDK — its FindEigen3.cmake is only
# on the module path after find_package(Holoscan).
include(isaac_deps/holoscan)

include(isaac_target_scope)

isaac_get_imported_targets_in_scope(ISAAC_TARGETS_BEFORE)
find_package(Eigen3 ${ISAAC_EIGEN3_VERSION} REQUIRED)
isaac_promote_new_imported_targets(ISAAC_TARGETS_BEFORE)

message(STATUS "Found Eigen3: ${Eigen3_DIR} (found suitable version \"${Eigen3_VERSION}\", minimum required is \"${ISAAC_EIGEN3_VERSION}\")")
