/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal test to validate the CMake + GTest build infrastructure.

#include <gtest/gtest.h>

TEST(BuildInfra, CompileAndLink) { EXPECT_TRUE(true); }
