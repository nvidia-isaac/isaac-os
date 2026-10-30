# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The pose tree, copied from the sensor framework release this repository's other source tree
# builds against, because the release targeted here does not carry one. See README.md for what was
# taken, what was changed, and how to renew it.

include(isaac_deps/eigen)
include(isaac_deps/holoscan)

add_library(pose_tree SHARED
  first_fit_allocator_base.cpp
  pose_tree.cpp
  pose_tree_history.cpp
)
add_library(isaac::common::pose_tree ALIAS pose_tree)

# The repository root alone. Every header here is reached as `common/pose_tree/...`, so the
# include path says who owns it and cannot shadow a header of the SDK's.
target_include_directories(pose_tree PUBLIC
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(pose_tree
  PUBLIC
    Eigen3::Eigen
    holoscan::core
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
