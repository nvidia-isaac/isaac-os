/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>

#include <gtest/gtest.h>

#include "matrix3x4d_generated.h"
#include "matrix6d_generated.h"

namespace isaac {
namespace {

// Non-symmetric test data throughout: a row/column transposition bug in the adapter's row-major
// indexing would be invisible against a symmetric matrix but caught here.
//
// isaac::MatrixNd (the wire struct) has no (row, col) accessor of its own -- m is a flat
// [double:N] at the schema level, so flatc only generates .m()->Get(flat_index)/.data(). Each
// test below maps the wire buffer's raw pointer with its own local Eigen::Map to get real
// (row, col) access to what Pack wrote, independent of calling UnPack at all.

TEST(MathCpuMessages, Matrix3x4dPackWritesRowMajor) {
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> in;
  in << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

  const isaac::Matrix3x4d wire = flatbuffers::Pack(in);
  const Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> wire_view(wire.m()->data());
  EXPECT_EQ(wire_view(0, 1), in(0, 1));
  EXPECT_EQ(wire_view(1, 0), in(1, 0));
}

TEST(MathCpuMessages, Matrix3x4dUnPackReadsRowMajor) {
  // Constructed directly, bypassing Pack entirely -- Matrix3x4dPackWritesRowMajor above only
  // pins down Pack; a bug isolated to UnPack (e.g. both transposing, canceling out over a
  // round trip) needs its own check independent of Pack to be caught. This confirms UnPack's
  // *output* -- a real Eigen::Matrix, not the flat wire struct -- is properly formatted: its
  // (row, col) values line up with the row-major wire bytes it was built from, not transposed
  // or otherwise scrambled in the flat-buffer-to-2D-matrix conversion.
  const isaac::Matrix3x4d wire(std::array<double, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  const Eigen::Matrix<double, 3, 4, Eigen::RowMajor> out = flatbuffers::UnPack(wire);
  EXPECT_EQ(out(0, 1), 2);
  EXPECT_EQ(out(1, 0), 5);
}

TEST(MathCpuMessages, Matrix6dPackWritesRowMajor) {
  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> in;
  for (int row = 0; row < 6; ++row) {
    for (int col = 0; col < 6; ++col) {
      in(row, col) = row * 6 + col;
    }
  }

  const isaac::Matrix6d wire = flatbuffers::Pack(in);
  const Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> wire_view(wire.m()->data());
  EXPECT_EQ(wire_view(0, 1), in(0, 1));
  EXPECT_EQ(wire_view(1, 0), in(1, 0));
}

TEST(MathCpuMessages, Matrix6dUnPackReadsRowMajor) {
  // See Matrix3x4dUnPackReadsRowMajor: verifies UnPack's Eigen::Matrix output is properly
  // row-major-formatted, independent of Pack.
  std::array<double, 36> m{};
  for (std::size_t i = 0; i < m.size(); ++i) {
    m[i] = static_cast<double>(i);
  }
  const isaac::Matrix6d wire(m);

  const Eigen::Matrix<double, 6, 6, Eigen::RowMajor> out = flatbuffers::UnPack(wire);
  EXPECT_EQ(out(0, 1), 1);
  EXPECT_EQ(out(1, 0), 6);
}

}  // namespace
}  // namespace isaac
