# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

add_executable(test_frame_encoder test_frame_encoder.cc)
target_link_libraries(test_frame_encoder
  PRIVATE
    GTest::gtest_main
    isaac::common::mcap_logger
)
add_test(NAME test_frame_encoder COMMAND test_frame_encoder)
set_tests_properties(test_frame_encoder PROPERTIES LABELS "cpp;common;gpu" TIMEOUT 60)

# Round-trip coverage: encodes through NVENC and decodes back through NVDEC, so it only exists
# where the NVIDIA Video Codec SDK is enabled -- nv_video_decoder is not built otherwise.
if(ISAAC_ENABLE_NV_VIDEO_SDK)
  include(isaac_deps/cuda)

  add_executable(test_mcap_logger_encoded_video_integration test_mcap_logger_encoded_video_integration.cc)
  target_link_libraries(test_mcap_logger_encoded_video_integration
    PRIVATE
      # NPP's colour-conversion module: the decoder hands back NV12, and the channel-order
      # check needs RGB.
      CUDA::nppicc
      GTest::gtest_main
      isaac::common::nv_video_decoder
      isaac::common::image_schema
      isaac::common::mcap_logger
      mcap
  )
  add_test(NAME test_mcap_logger_encoded_video_integration COMMAND test_mcap_logger_encoded_video_integration)
  set_tests_properties(test_mcap_logger_encoded_video_integration PROPERTIES
    LABELS "cpp;common;gpu;integration"
    TIMEOUT 60
  )
endif()

add_executable(test_mcap_logger_integration test_mcap_logger_integration.cc)
target_link_libraries(test_mcap_logger_integration
  PRIVATE
    GTest::gtest_main
    isaac::common::joint_cpu_messages
    isaac::common::mcap_logger
    mcap
)
add_test(NAME test_mcap_logger_integration COMMAND test_mcap_logger_integration)
set_tests_properties(test_mcap_logger_integration PROPERTIES
  LABELS "cpp;common;integration"
  TIMEOUT 30
)

add_executable(test_mcap_logger_tensor_schema_integration test_mcap_logger_tensor_schema_integration.cc)
target_link_libraries(test_mcap_logger_tensor_schema_integration
  PRIVATE
    GTest::gtest_main
    isaac::common::image_schema
    isaac::common::mcap_logger
    mcap
)
add_test(NAME test_mcap_logger_tensor_schema_integration COMMAND test_mcap_logger_tensor_schema_integration)
set_tests_properties(test_mcap_logger_tensor_schema_integration PROPERTIES
  LABELS "cpp;common;integration"
  TIMEOUT 30
)

add_executable(test_mcap_logger_wiring test_mcap_logger_wiring.cc)
target_link_libraries(test_mcap_logger_wiring
  PRIVATE
    GTest::gtest_main
    isaac::common::image_schema
    isaac::common::joint_cpu_messages
    isaac::common::mcap_logger
)
add_test(NAME test_mcap_logger_wiring COMMAND test_mcap_logger_wiring)
set_tests_properties(test_mcap_logger_wiring PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_mcap_sink test_mcap_sink.cc)
target_link_libraries(test_mcap_sink
  PRIVATE
    GTest::gtest_main
    isaac::common::mcap_logger
    mcap
)
add_test(NAME test_mcap_sink COMMAND test_mcap_sink)
set_tests_properties(test_mcap_sink PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_tensor_frame_schema test_tensor_frame_schema.cc)
target_link_libraries(test_tensor_frame_schema
  PRIVATE
    GTest::gtest_main
    isaac::common::mcap_logger
)
add_test(NAME test_tensor_frame_schema COMMAND test_tensor_frame_schema)
set_tests_properties(test_tensor_frame_schema PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_compressed_video_proto test_compressed_video_proto.cc)
target_link_libraries(test_compressed_video_proto
  PRIVATE
    GTest::gtest_main
    isaac::common::mcap_logger
)
add_test(NAME test_compressed_video_proto COMMAND test_compressed_video_proto)
set_tests_properties(test_compressed_video_proto PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_video_encode_options test_video_encode_options.cc)
target_link_libraries(test_video_encode_options
  PRIVATE
    GTest::gtest_main
    isaac::common::mcap_logger
)
add_test(NAME test_video_encode_options COMMAND test_video_encode_options)
set_tests_properties(test_video_encode_options PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_poller_base test_poller_base.cc)
target_link_libraries(test_poller_base
  PRIVATE
    GTest::gtest_main
    isaac::common::mcap_logger
)
add_test(NAME test_poller_base COMMAND test_poller_base)
set_tests_properties(test_poller_base PROPERTIES LABELS "cpp;common" TIMEOUT 30)
