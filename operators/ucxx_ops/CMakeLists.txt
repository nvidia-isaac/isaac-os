# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(ucxx_ops SHARED
  ucxx_endpoint.cc
  ucxx_explicit_instantiations.cc
  wire_codec.cc
)
add_library(isaac::operators::ucxx_ops ALIAS ucxx_ops)

include(isaac_deps/cuda)
include(isaac_deps/holoscan)

# UCX / UCXX come from the pip prebundle installed at Dockerfile build time
# (mirroring Isaac Sim's pip_ucx_prebundle approach; HSDK5 does not ship
# UCX/UCXX). Bypass find_package (RAPIDS' CMake configs pull a fragile
# transitive chain that fights HSDK5) and reference the wheel paths directly.
set(UCX_PREBUNDLE_PREFIX "/opt/ucx_prebundle")

target_include_directories(ucxx_ops
  PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${UCX_PREBUNDLE_PREFIX}/librmm/include
    ${UCX_PREBUNDLE_PREFIX}/libucx/include
    ${UCX_PREBUNDLE_PREFIX}/libucxx/include
    ${UCX_PREBUNDLE_PREFIX}/rapids_logger/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_directories(ucxx_ops
  PUBLIC
    ${UCX_PREBUNDLE_PREFIX}/librmm/lib64
    ${UCX_PREBUNDLE_PREFIX}/libucx/lib
    ${UCX_PREBUNDLE_PREFIX}/libucxx/lib64
    ${UCX_PREBUNDLE_PREFIX}/rapids_logger/lib64
)

# CUDA is public because the receiver template allocates and copies device buffers.
target_link_libraries(ucxx_ops
  PUBLIC
    CUDA::cudart
    flatbuffers::flatbuffers
    holoscan::core
    holoscan::sensor_io
    isaac::common::joint_cpu_messages
    isaac::messages
    rapids_logger
    rmm
    ucm
    ucp
    ucs
    uct
    ucxx
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
  include(isaac_deps/gtest)
  add_executable(test_ucxx_connection_failure tests/test_ucxx_connection_failure.cc)
  target_link_libraries(test_ucxx_connection_failure PRIVATE
    CUDA::cudart
    GTest::gtest_main
    holoscan::core
    isaac::common::joint_cpu_messages
    isaac::operators::ucxx_ops
  )
  # Separate processes acquire managed signals before CUDA/UCX start threads.
  foreach(case IN ITEMS
    DeferredReceiverFailureTerminatesSession
    ImuBurstBlockingFails
    ImuBurstDropOldestContinues
    InitialConnectionTimesOut
    InitialConnectionWaitsForLatePeer
    SkippedConnectionCallbackDoesNotPreventSessionFailure
    SlowCameraConsumerDoesNotFailSession
    StopInterruptsInitialConnectionWait
  )
    add_test(NAME test_ucxx_connection_failure_${case}
      COMMAND test_ucxx_connection_failure --gtest_filter=UcxxConnectionFailure.${case})
    set_tests_properties(test_ucxx_connection_failure_${case} PROPERTIES
      LABELS "cpp;gpu;operators"
      TIMEOUT 30
    )
  endforeach()
  add_test(NAME test_ucxx_receiver_frame_id
    COMMAND test_ucxx_connection_failure
      --gtest_filter=UcxxReceiver.GraphClockKeepsWireTimestampAsFrameId)
  set_tests_properties(test_ucxx_receiver_frame_id PROPERTIES LABELS "cpp;operators" TIMEOUT 30)

  add_test(NAME test_ucxx_endpoint_rejects_invalid_port
    COMMAND test_ucxx_connection_failure --gtest_filter=UcxxConnectionFailure.RejectsInvalidPort)
  set_tests_properties(test_ucxx_endpoint_rejects_invalid_port PROPERTIES
    LABELS "cpp;operators" TIMEOUT 30)
endif()
