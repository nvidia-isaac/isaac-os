# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# lz4 dependency setup (system package: liblz4-dev).

# Already exists?
if(TARGET LZ4::lz4)
  return()
endif()

# Resolved by cmake/Findlz4.cmake, which also defines the LZ4::lz4 imported target.
find_package(lz4 REQUIRED)
