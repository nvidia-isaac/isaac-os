# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GoogleTest dependency setup.

set(ISAAC_GTEST_VERSION 1.17.0)

if(TARGET GTest::gtest)
  return()
endif()

include(isaac_cpm_find_package)

isaac_cpm_find_package(
  NAME GTest
  VERSION ${ISAAC_GTEST_VERSION}
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v${ISAAC_GTEST_VERSION}
  OPTIONS
    "BUILD_GMOCK OFF"
    "INSTALL_GTEST OFF"
)
