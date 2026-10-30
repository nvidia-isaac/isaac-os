/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Eigen/Dense"

// Declarations only, referenced by matrix6d.fbs's native_include: flatc places this include
// immediately before isaac::Matrix6d's own definition in the generated header, so anything
// needing the complete type (the actual Pack/UnPack bodies) can't live here -- see
// matrix6d_adapter.cc. Ordinary (non-inline) linkage, so unlike a header-only definition,
// consumers just need this declaration in scope; the linker resolves the call to
// matrix6d_adapter.cc's definition as long as math_cpu_messages is linked in.
namespace isaac {
struct Matrix6d;
}  // namespace isaac

namespace flatbuffers {
Eigen::Matrix<double, 6, 6, Eigen::RowMajor> UnPack(const isaac::Matrix6d& v);
isaac::Matrix6d Pack(const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& v);
}  // namespace flatbuffers
