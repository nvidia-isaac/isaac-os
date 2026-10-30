# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

add_executable(test_converter test_converter.cc)
target_link_libraries(test_converter
  PRIVATE
    GTest::gtest_main
    isaac::common::ros2_bridge
)
add_test(NAME test_converter COMMAND test_converter)
set_tests_properties(test_converter PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_ros2_bridge_wiring test_ros2_bridge_wiring.cc)
target_link_libraries(test_ros2_bridge_wiring
  PRIVATE
    GTest::gtest_main
    isaac::common::ros2_bridge
)
add_test(NAME test_ros2_bridge_wiring COMMAND test_ros2_bridge_wiring)
set_tests_properties(test_ros2_bridge_wiring PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_ros2_bridge_session test_ros2_bridge_session.cc)
target_link_libraries(test_ros2_bridge_session
  PRIVATE
    GTest::gtest_main
    isaac::common::ros2_bridge
)
add_test(NAME test_ros2_bridge_session COMMAND test_ros2_bridge_session)
set_tests_properties(test_ros2_bridge_session PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_tensor_converter test_tensor_converter.cc)
target_link_libraries(test_tensor_converter
  PRIVATE
    GTest::gtest_main
    isaac::common::ros2_bridge
)
add_test(NAME test_tensor_converter COMMAND test_tensor_converter)
set_tests_properties(test_tensor_converter PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_ros2_bridge_roundtrip test_ros2_bridge_roundtrip.cc)
target_link_libraries(test_ros2_bridge_roundtrip
  PRIVATE
    GTest::gtest_main
    isaac::common::ros2_bridge
)
add_test(NAME test_ros2_bridge_roundtrip COMMAND test_ros2_bridge_roundtrip)
set_tests_properties(test_ros2_bridge_roundtrip PROPERTIES
  LABELS "cpp;common;integration"
  TIMEOUT 30
)

# One gtest executable built from ${NAME}.cc and linked against the bridge.
function(isaac_add_ros2_bridge_test NAME LABELS TIMEOUT)
  add_executable(${NAME} ${NAME}.cc)
  target_link_libraries(${NAME}
    PRIVATE
      GTest::gtest_main
      isaac::common::ros2_bridge
  )
  add_test(NAME ${NAME} COMMAND ${NAME})
  set_tests_properties(${NAME} PROPERTIES LABELS "${LABELS}" TIMEOUT ${TIMEOUT})
endfunction()

isaac_add_ros2_bridge_test(test_memory_transfer "cpp;common" 30)

# A compile-fail test builds a target that must NOT compile and passes only when the build output
# contains the expected message. The targets are excluded from the normal build.
function(isaac_add_compile_fail_test NAME SOURCE MESSAGE_REGEX)
  add_executable(${NAME} EXCLUDE_FROM_ALL ${SOURCE})
  target_link_libraries(${NAME} PRIVATE isaac::common::ros2_bridge)
  add_test(NAME ${NAME}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${NAME})
  set_tests_properties(${NAME} PROPERTIES
    LABELS "cpp;common;compile_fail"
    PASS_REGULAR_EXPRESSION "${MESSAGE_REGEX}"
    RUN_SERIAL TRUE  # each one runs cmake --build on the shared build tree
    TIMEOUT 600)
endfunction()

isaac_add_compile_fail_test(compile_fail_plain_missing_to_ros2
  compile_fail_plain_missing_to_ros2.cc
  "this \\(Ros2Type, HsdkType\\) pair is a plain type")
isaac_add_compile_fail_test(compile_fail_plain_on_tensor_type
  compile_fail_plain_on_tensor_type.cc
  "this \\(Ros2Type, HsdkType\\) pair is a tensor-schema type")

if(CUDAToolkit_FOUND AND cuda_buffer_FOUND)
  isaac_add_ros2_bridge_test(test_cuda_device_utils "cpp;common;gpu" 60)
  isaac_add_ros2_bridge_test(test_cuda_pool_device "cpp;common;gpu" 60)
  isaac_add_ros2_bridge_test(test_ros_buffer_egress "cpp;common;gpu" 120)
  isaac_add_ros2_bridge_test(test_ros_buffer_ingress "cpp;common;gpu" 120)
  isaac_add_ros2_bridge_test(test_ros2_bridge_cuda_roundtrip "cpp;common;gpu;integration" 120)
  # The same binary with a host publisher into a topic declared publisher_memory = kHost.
  add_test(NAME test_ros2_bridge_cuda_roundtrip_host_publisher
    COMMAND test_ros2_bridge_cuda_roundtrip)
  set_tests_properties(test_ros2_bridge_cuda_roundtrip_host_publisher PROPERTIES
    LABELS "cpp;common;gpu;integration" TIMEOUT 120
    ENVIRONMENT "ROS2_BRIDGE_PUBLISHER_MEMORY=host")
  # A host publisher into a topic that does not say so: slower and logged, but the same result.
  add_test(NAME test_ros2_bridge_cuda_roundtrip_host_publisher_undeclared
    COMMAND test_ros2_bridge_cuda_roundtrip)
  set_tests_properties(test_ros2_bridge_cuda_roundtrip_host_publisher_undeclared PROPERTIES
    LABELS "cpp;common;gpu;integration" TIMEOUT 120
    ENVIRONMENT "ROS2_BRIDGE_PUBLISHER_MEMORY=host_undeclared")
  # The ROS CUDA buffer backend fails to delete the ROS context's participant when a CUDA
  # subscription outlives a publisher that feeds it in the same process; the test releases them in
  # the order that avoids it.
  set_tests_properties(test_ros2_bridge_cuda_roundtrip PROPERTIES
    FAIL_REGULAR_EXPRESSION "Failed to delete participant")
  # The same binary with intra-process communication: no same-process IPC, so no warning about it.
  add_test(NAME test_ros2_bridge_cuda_roundtrip_intra_process
    COMMAND test_ros2_bridge_cuda_roundtrip)
  set_tests_properties(test_ros2_bridge_cuda_roundtrip_intra_process PROPERTIES
    LABELS "cpp;common;gpu;integration" TIMEOUT 120
    ENVIRONMENT "ROS2_BRIDGE_INTRA_PROCESS=1"
    FAIL_REGULAR_EXPRESSION "Same-process CUDA IPC;Failed to delete participant")
endif()
