# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)
include(isaac_deps/so101_reach_policy)

add_executable(test_so101_forward_kinematics test_so101_forward_kinematics.cc)
add_executable(test_so101_reach_fused_op test_so101_reach_fused_op.cc)

target_link_libraries(test_so101_forward_kinematics
  PRIVATE
    GTest::gtest_main
    isaac::operators::so101_reach
)
target_compile_definitions(test_so101_reach_fused_op
  PRIVATE
    SO101_REACH_MODEL_PATH="${PROJECT_SOURCE_DIR}/data/grip_reach_delta_policy_onnx/policy_fingertip_delta.onnx"
)
target_link_libraries(test_so101_reach_fused_op
  PRIVATE
    GTest::gtest_main
    isaac::operators::so101_reach
)
add_dependencies(test_so101_reach_fused_op grip_reach_delta_policy_onnx)

add_test(NAME test_so101_forward_kinematics COMMAND test_so101_forward_kinematics)
add_test(NAME test_so101_reach_fused_op COMMAND test_so101_reach_fused_op)
set_tests_properties(test_so101_forward_kinematics PROPERTIES
  LABELS "cpp;operators"
  TIMEOUT 30
)
set_tests_properties(test_so101_reach_fused_op PROPERTIES
  LABELS "cpp;gpu;operators"
  TIMEOUT 120
)
