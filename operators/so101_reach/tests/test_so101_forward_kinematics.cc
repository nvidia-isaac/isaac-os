/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>

#include <gtest/gtest.h>

#include "operators/so101_reach/so101_forward_kinematics.h"

namespace isaac::os::ops {
namespace {

TEST(So101ForwardKinematics, MatchesUrdfNominalPose) {
  constexpr std::array<float, 6U> kJointPosition{0.0F, 0.0F, 0.0F, 1.57000005F, 0.0F, 0.0F};

  const std::array<float, 3U> fingertip = so101_fingertip_position(kJointPosition);

  EXPECT_NEAR(fingertip[0U], 0.22436032F, 1.0e-5F);
  EXPECT_NEAR(fingertip[1U], -0.00000960F, 1.0e-5F);
  EXPECT_NEAR(fingertip[2U], 0.07516410F, 1.0e-5F);
}

TEST(So101ForwardKinematics, MatchesUrdfArticulatedPose) {
  constexpr std::array<float, 6U> kJointPosition{0.2F, -0.4F, 0.3F, 1.1F, -0.2F, 0.0F};

  const std::array<float, 3U> fingertip = so101_fingertip_position(kJointPosition);

  EXPECT_NEAR(fingertip[0U], 0.26023483F, 1.0e-5F);
  EXPECT_NEAR(fingertip[1U], -0.04329125F, 1.0e-5F);
  EXPECT_NEAR(fingertip[2U], 0.11170135F, 1.0e-5F);
}

}  // namespace
}  // namespace isaac::os::ops
