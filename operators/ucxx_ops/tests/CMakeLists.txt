# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)
add_executable(test_joint_cpu_ucx_codec test_joint_cpu_ucx_codec.cc)
target_link_libraries(test_joint_cpu_ucx_codec PRIVATE
  GTest::gtest_main
  isaac::operators::ucxx_ops
)
add_test(NAME test_joint_cpu_ucx_codec COMMAND test_joint_cpu_ucx_codec)
set_tests_properties(test_joint_cpu_ucx_codec PROPERTIES LABELS "cpp;operators" TIMEOUT 30)
