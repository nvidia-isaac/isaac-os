# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_nv_video_decoder
  test_nv_video_decoder.cpp
)

include(isaac_deps/gtest)
target_link_libraries(test_nv_video_decoder
  PRIVATE
    GTest::gtest
    GTest::gtest_main
    isaac::common::nv_video_decoder
)

add_test(NAME test_nv_video_decoder COMMAND test_nv_video_decoder)
set_tests_properties(test_nv_video_decoder PROPERTIES LABELS "codec;cpp;gpu")
