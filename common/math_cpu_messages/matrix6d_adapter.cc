/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>

#include "Eigen/Dense"

#include "matrix6d_generated.h"

namespace flatbuffers {

using RowMajorMatrix6d = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>;

// isaac::Matrix6d is row-major on the wire, and RowMajorMatrix6d is explicitly declared
// Eigen::RowMajor (Eigen defaults to column-major) -- the two layouts match exactly, so
// Eigen::Map reinterprets the buffer directly instead of copying element-by-element.
RowMajorMatrix6d UnPack(const isaac::Matrix6d& v) {
  return Eigen::Map<const RowMajorMatrix6d>(v.m()->data());
}

isaac::Matrix6d Pack(const RowMajorMatrix6d& v) {
  std::array<double, 36> m;
  Eigen::Map<RowMajorMatrix6d>(m.data()) = v;
  return isaac::Matrix6d(m);
}

}  // namespace flatbuffers
