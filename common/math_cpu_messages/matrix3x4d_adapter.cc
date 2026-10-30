/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>

#include "Eigen/Dense"

#include "matrix3x4d_generated.h"

namespace flatbuffers {

using RowMajorMatrix3x4d = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

// isaac::Matrix3x4d is row-major on the wire, and RowMajorMatrix3x4d is explicitly declared
// Eigen::RowMajor (Eigen defaults to column-major) -- the two layouts match exactly, so
// Eigen::Map reinterprets the buffer directly instead of copying element-by-element.
RowMajorMatrix3x4d UnPack(const isaac::Matrix3x4d& v) {
  return Eigen::Map<const RowMajorMatrix3x4d>(v.m()->data());
}

isaac::Matrix3x4d Pack(const RowMajorMatrix3x4d& v) {
  std::array<double, 12> m;
  Eigen::Map<RowMajorMatrix3x4d>(m.data()) = v;
  return isaac::Matrix3x4d(m);
}

}  // namespace flatbuffers
