/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "operators/cuvslam/calibration_mapping.h"

namespace isaac::os::ops {
namespace {

TEST(BrownFromPlumbBob, MovesTheThirdRadialTermAheadOfTheTangentialPair) {
  // plumb_bob states (k1, k2, p1, p2, k3); values chosen so each is identifiable.
  const std::vector<double> plumb_bob{1.0, 2.0, 3.0, 4.0, 5.0};  // k1 k2 p1 p2 k3
  const auto brown = brown_from_plumb_bob(plumb_bob);
  ASSERT_TRUE(brown.has_value());

  // Brown takes (k1, k2, k3, p1, p2).
  EXPECT_FLOAT_EQ((*brown)[0], 1.0F);  // k1
  EXPECT_FLOAT_EQ((*brown)[1], 2.0F);  // k2
  EXPECT_FLOAT_EQ((*brown)[2], 5.0F);  // k3, from the last plumb_bob slot
  EXPECT_FLOAT_EQ((*brown)[3], 3.0F);  // p1
  EXPECT_FLOAT_EQ((*brown)[4], 4.0F);  // p2
}

TEST(BrownFromPlumbBob, RejectsAnythingOtherThanFiveCoefficients) {
  const std::vector<double> four{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> eight{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::vector<double> none{};
  EXPECT_FALSE(brown_from_plumb_bob(four).has_value());
  EXPECT_FALSE(brown_from_plumb_bob(eight).has_value());
  EXPECT_FALSE(brown_from_plumb_bob(none).has_value());
}

TEST(Narrow, LeavesModelsThatAlreadyAgreeAlone) {
  // rational_polynomial is the first eight OpenCV coefficients, which is the order cuVSLAM's
  // Polynomial model documents, so nothing moves.
  const std::vector<double> polynomial{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const std::vector<float> narrowed = narrow(polynomial);
  ASSERT_EQ(narrowed.size(), 8U);
  for (std::size_t i = 0; i < narrowed.size(); ++i) {
    EXPECT_FLOAT_EQ(narrowed[i], static_cast<float>(polynomial[i]));
  }
}

}  // namespace
}  // namespace isaac::os::ops
