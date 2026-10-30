# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MCAP dependency setup.

# Already exists?
if(TARGET mcap)
  return()
endif()

# mcap_builder discovers LZ4 and Zstd itself and silently compiles out the corresponding
# compression support when either is missing. Require LZ4 up front so a missing liblz4 fails
# here instead of degrading `compression: lz4` to uncompressed chunks at runtime. Zstd comes
# from the Holoscan SDK install tree already on CMAKE_PREFIX_PATH.
include(isaac_deps/lz4)

set(_patch_dir "${PROJECT_SOURCE_DIR}/third_party/mcap_builder")
set(_patches
  ${_patch_dir}/use_libzstd_static_target.patch
  ${_patch_dir}/switch_mcap_dependency_download_to_git_clone.patch
)

include(isaac_cpm_find_package)
isaac_cpm_find_package(
  NAME mcap_builder
  GITHUB_REPOSITORY olympus-robotics/mcap_builder
  GIT_TAG d6f3662b7204341797eaeca5ff97b8f659d02bda
  PATCH_COMMAND git apply --check --reverse ${_patches} || git apply ${_patches}
)
