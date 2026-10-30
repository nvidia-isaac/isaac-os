/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/robot_control_benchmark/arm_kinematics.hpp"

#include <array>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

namespace isaac::os::benchmark {
namespace {

/// Verify that frame contents never depend on caller-provided storage values.
template <typename T>
void check_forward_overwrites_frames() {
  const ArmModelT<T> model;
  for (int sample = 0; sample < 32; ++sample) {
    auto q = ArmModelT<T>::home();
    for (int joint = 0; joint < kNumJoints; ++joint) {
      q[joint] += static_cast<T>(0.2 * std::sin(sample + joint));
    }
    // The CUDA-compatible forward/jacobian API takes references to fixed C arrays.
    Mat4T<T> reference_frames[kNumJoints + 1]{};  // NOSONAR: fixed-array API.
    Mat4T<T> poisoned_frames[kNumJoints + 1];     // NOSONAR: fixed-array API.
    for (auto& frame : poisoned_frames) {
      for (auto& value : frame.m) {
        value = std::numeric_limits<T>::quiet_NaN();
      }
    }
    Mat4T<T> reference_ee{};
    Mat4T<T> poisoned_ee{};
    model.forward(q.data(), reference_ee, reference_frames);
    model.forward(q.data(), poisoned_ee, poisoned_frames);
    for (int frame = 0; frame <= kNumJoints; ++frame) {
      for (int element = 0; element < 16; ++element) {
        ASSERT_TRUE(std::isfinite(poisoned_frames[frame][element]))
            << "sample=" << sample << " frame=" << frame << " element=" << element;
        EXPECT_EQ(poisoned_frames[frame][element], reference_frames[frame][element]);
      }
    }
    std::array<T, 6 * kNumJoints> reference_jacobian;
    std::array<T, 6 * kNumJoints> poisoned_jacobian;
    model.jacobian(reference_frames, reference_ee, reference_jacobian.data());
    model.jacobian(poisoned_frames, poisoned_ee, poisoned_jacobian.data());
    for (int element = 0; element < 6 * kNumJoints; ++element) {
      EXPECT_EQ(poisoned_jacobian[element], reference_jacobian[element]);
    }
  }
}

TEST(ArmKinematics, ForwardOverwritesEveryFrameFloat) { check_forward_overwrites_frames<float>(); }

TEST(ArmKinematics, ForwardOverwritesEveryFrameDouble) {
  check_forward_overwrites_frames<double>();
}

}  // namespace
}  // namespace isaac::os::benchmark
