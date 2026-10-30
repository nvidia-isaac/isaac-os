# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/cli11)
include(isaac_deps/cuda)
include(isaac_deps/holoscan)

add_executable(robot_control_benchmark
  gpu_contention_op.cc
  main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../../common/robot_control_benchmark/gpu_motion_ik.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../common/robot_control_benchmark/motion_backend.cc
)
target_compile_features(robot_control_benchmark PRIVATE cuda_std_20 cxx_std_20)
# -O2 is scoped to C++: passing it to nvcc would set the device optimization level.
#
# No --expt-relaxed-constexpr, deliberately. The GPU solve runs ArmModelT from
# arm_kinematics.hpp so the CPU and GPU arms share one definition, and that header keeps
# constexpr host functions off the device path specifically so no CUDA target here needs
# the flag. Reaching for std::array or std::clamp in device code would bring it back.
#
# Keep nvcc's default multiply-add contraction. Candidate selection is deterministic on
# a given device; tests compare objective values, since host/device rounding can change
# which of two nearly equivalent candidates wins.
target_compile_options(robot_control_benchmark
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
)
target_include_directories(robot_control_benchmark PRIVATE ${PROJECT_SOURCE_DIR})
# A list containing sm_50 breaks CUDA 13 nvcc, so pin this target the way the sibling
# GPU operators do rather than inheriting a global list.
set_target_properties(robot_control_benchmark PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(robot_control_benchmark
  PRIVATE
    CLI11::CLI11
    CUDA::cublas
    CUDA::cudart
    holoscan::config_io
    holoscan::core
    iceoryx2-c::shared-lib
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
