# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)

add_executable(mock_build main.cc)
target_link_libraries(mock_build PRIVATE holoscan::core)
