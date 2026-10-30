# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_v4l2_camera test_v4l2_camera.cc)

include(isaac_deps/gtest)
set(_ctest_skip_return_code 77)
target_compile_definitions(test_v4l2_camera PRIVATE
  ISAAC_CTEST_SKIP_RETURN_CODE=${_ctest_skip_return_code}
)
target_link_libraries(test_v4l2_camera
  PRIVATE
    GTest::gtest
    holoscan::core
    isaac::operators::v4l2_camera
)

add_test(NAME test_v4l2_camera COMMAND test_v4l2_camera)
set_tests_properties(test_v4l2_camera PROPERTIES
  LABELS "camera;cpp;operators"
  SKIP_RETURN_CODE ${_ctest_skip_return_code}
  TIMEOUT 60
)
