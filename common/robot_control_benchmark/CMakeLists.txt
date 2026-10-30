# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This directory intentionally defines no library. The workload is consumed as sources by
# each runtime that needs it, the application listing them in its own
# add_executable. A CMake target here would be reachable only from the Isaac OS graph,
# which is the one thing a runtime-independent workload must not require.

if(ISAAC_BUILD_TESTS)
  include(isaac_deps/cuda)
  include(isaac_deps/gtest)

  add_executable(robot_control_arm_kinematics_test tests/arm_kinematics_test.cc)
  target_compile_features(robot_control_arm_kinematics_test PRIVATE cxx_std_20)
  target_include_directories(robot_control_arm_kinematics_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(robot_control_arm_kinematics_test PRIVATE GTest::gtest_main)

  add_test(NAME robot_control_arm_kinematics_test COMMAND robot_control_arm_kinematics_test)
  set_tests_properties(robot_control_arm_kinematics_test PROPERTIES
    LABELS "cpp;common"
    TIMEOUT 30
  )

  add_executable(robot_control_latency_stats_test tests/latency_stats_test.cc)
  target_compile_features(robot_control_latency_stats_test PRIVATE cxx_std_20)
  target_include_directories(robot_control_latency_stats_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(robot_control_latency_stats_test PRIVATE GTest::gtest_main)

  add_test(NAME robot_control_latency_stats_test COMMAND robot_control_latency_stats_test)
  set_tests_properties(robot_control_latency_stats_test PROPERTIES
    LABELS "cpp;common"
    TIMEOUT 30
  )

  add_executable(robot_control_managed_irq_guard_test tests/managed_irq_guard_test.cc)
  target_compile_features(robot_control_managed_irq_guard_test PRIVATE cxx_std_20)
  target_include_directories(robot_control_managed_irq_guard_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(robot_control_managed_irq_guard_test PRIVATE GTest::gtest_main)

  add_test(NAME robot_control_native_irq_guard_test COMMAND robot_control_managed_irq_guard_test)
  set_tests_properties(robot_control_native_irq_guard_test PROPERTIES
    LABELS "cpp;common"
    TIMEOUT 30
  )

  # Needs a device: checks parallel candidate selection against an exhaustive host search.
  # TIMEOUT is explicit because CI invokes ctest with --timeout 60.
  add_executable(robot_control_gpu_motion_ik_test
    gpu_motion_ik.cu
    tests/gpu_motion_ik_test.cc
  )
  target_compile_features(robot_control_gpu_motion_ik_test PRIVATE cuda_std_20 cxx_std_20)
  target_include_directories(robot_control_gpu_motion_ik_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(robot_control_gpu_motion_ik_test
    PRIVATE
      CUDA::cudart
      GTest::gtest_main
  )
  # A list containing sm_50 breaks CUDA 13 nvcc, so pin this target the way the sibling
  # GPU operators do rather than inheriting a global list.
  set_target_properties(robot_control_gpu_motion_ik_test PROPERTIES CUDA_ARCHITECTURES native)

  add_test(NAME robot_control_gpu_motion_ik_test COMMAND robot_control_gpu_motion_ik_test)
  set_tests_properties(robot_control_gpu_motion_ik_test PROPERTIES
    LABELS "cpp;common;gpu"
    TIMEOUT 120
  )
endif()
