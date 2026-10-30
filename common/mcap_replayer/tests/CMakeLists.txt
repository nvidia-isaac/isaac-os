# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

add_executable(test_compressed_video_builder test_compressed_video_builder.cc)
target_link_libraries(test_compressed_video_builder
  PRIVATE
    GTest::gtest_main
    isaac::common::image_schema
    isaac::common::mcap_logger
    isaac::common::mcap_replayer
    # Constructing a McapReplayer instantiates McapFile::verify_topic<T>() here, which calls the
    # mcap reader's own code directly; mcap_replayer propagates mcap's headers but not its library.
    mcap
)
add_test(NAME test_compressed_video_builder COMMAND test_compressed_video_builder)
set_tests_properties(test_compressed_video_builder PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_decoder test_decoder.cc)
target_link_libraries(test_decoder
  PRIVATE
    GTest::gtest_main
    isaac::common::geometry_cpu_messages
    isaac::common::mcap_replayer
)
add_test(NAME test_decoder COMMAND test_decoder)
set_tests_properties(test_decoder PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_mcap_file test_mcap_file.cc)
target_link_libraries(test_mcap_file
  PRIVATE
    GTest::gtest_main
    isaac::common::geometry_cpu_messages
    isaac::common::image_schema
    isaac::common::mcap_logger
    isaac::common::mcap_replayer
    # test_mcap_file.cc includes mcap/reader.hpp directly (to read ground-truth Statistics via
    # its own independent mcap::McapReader), so it needs this link even though mcap_replayer
    # itself only propagates mcap's headers, not the library, to its consumers.
    mcap
)
add_test(NAME test_mcap_file COMMAND test_mcap_file)
set_tests_properties(test_mcap_file PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_mcap_replayer test_mcap_replayer.cc)
target_link_libraries(test_mcap_replayer
  PRIVATE
    GTest::gtest_main
    isaac::common::geometry_cpu_messages
    isaac::common::mcap_logger
    isaac::common::mcap_replayer
    # test_mcap_replayer.cc holds a McapFile by value to read a recording's own per-topic counts
    # back, which instantiates the mcap reader's inline destructor here -- mcap_replayer itself
    # propagates mcap's headers to its consumers but not its library, so this test links it.
    mcap
)
add_test(NAME test_mcap_replayer COMMAND test_mcap_replayer)
set_tests_properties(test_mcap_replayer PROPERTIES LABELS "cpp;common" TIMEOUT 300)

add_executable(test_pacing test_pacing.cc)
target_link_libraries(test_pacing
  PRIVATE
    GTest::gtest_main
    isaac::common::mcap_replayer
)
add_test(NAME test_pacing COMMAND test_pacing)
set_tests_properties(test_pacing PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_replay_buffer test_replay_buffer.cc)
target_link_libraries(test_replay_buffer
  PRIVATE
    GTest::gtest_main
    isaac::common::mcap_replayer
)
add_test(NAME test_replay_buffer COMMAND test_replay_buffer)
set_tests_properties(test_replay_buffer PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_replay_stats test_replay_stats.cc)
target_link_libraries(test_replay_stats
  PRIVATE
    GTest::gtest_main
    isaac::common::mcap_replayer
)
add_test(NAME test_replay_stats COMMAND test_replay_stats)
set_tests_properties(test_replay_stats PROPERTIES LABELS "cpp;common" TIMEOUT 30)

add_executable(test_tensor_decoder test_tensor_decoder.cc)
target_link_libraries(test_tensor_decoder
  PRIVATE
    GTest::gtest_main
    isaac::common::geometry_cpu_messages
    isaac::common::image_schema
    isaac::common::mcap_logger
    isaac::common::mcap_replayer
    # Constructing a McapReplayer instantiates McapFile::verify_topic<T>() here, a template that
    # calls the mcap reader's own code directly -- mcap_replayer propagates mcap's headers to its
    # consumers but not its library, so this test links it.
    mcap
)
add_test(NAME test_tensor_decoder COMMAND test_tensor_decoder)
set_tests_properties(test_tensor_decoder PROPERTIES LABELS "cpp;common" TIMEOUT 120)

if(ISAAC_ENABLE_NV_VIDEO_SDK)
  add_executable(test_compressed_video_replay test_compressed_video_replay.cc)
  target_link_libraries(test_compressed_video_replay
    PRIVATE
      GTest::gtest_main
      isaac::common::image_schema
      isaac::common::mcap_logger
      isaac::common::mcap_replayer
      isaac::common::nv_video_decoder
      mcap
  )
  add_test(NAME test_compressed_video_replay COMMAND test_compressed_video_replay)
  set_tests_properties(test_compressed_video_replay PROPERTIES
    LABELS "cpp;common;gpu;integration"
    TIMEOUT 120
  )
endif()
