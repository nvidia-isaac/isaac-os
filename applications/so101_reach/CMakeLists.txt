# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

add_subdirectory(mock)
add_subdirectory(real)
add_subdirectory(sim)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
