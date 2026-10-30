# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Unitree SDK2 dependency setup.

set(ISAAC_UNITREE_SDK2_VERSION 2.0.2)

if(TARGET unitree_sdk2)
  return()
endif()

include(isaac_cpm_find_package)

set(_patches ${PROJECT_SOURCE_DIR}/third_party/unitree-sdk2/cxx20-template-id-destructor.patch)

isaac_cpm_find_package(
  NAME unitree_sdk2
  GITHUB_REPOSITORY unitreerobotics/unitree_sdk2
  GIT_TAG ${ISAAC_UNITREE_SDK2_VERSION}
  PATCH_COMMAND git apply --check --reverse ${_patches} || git apply ${_patches}
  OPTIONS
    "BUILD_EXAMPLES OFF"
    "CMAKE_POLICY_VERSION_MINIMUM 3.10"
)
