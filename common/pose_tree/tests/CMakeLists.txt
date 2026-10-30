# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The two source files here are copies, taken from the same place as the tree itself and kept at
# their original names. See ../README.md.

include(isaac_deps/gtest)

add_executable(test_pose_tree
  failure_unwind.cpp
  pose_tree.cpp
  pose_tree_history.cpp
)
target_link_libraries(test_pose_tree
  PRIVATE
    GTest::gtest_main
    isaac::common::pose_tree
)

add_test(NAME test_pose_tree COMMAND test_pose_tree)
set_tests_properties(test_pose_tree PROPERTIES
  LABELS "cpp;common"
  TIMEOUT 120
)
