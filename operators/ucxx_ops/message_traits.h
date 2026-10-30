/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_UCXX_OPS_MESSAGE_TRAITS_H
#define ISAAC_OS_OPERATORS_UCXX_OPS_MESSAGE_TRAITS_H

#include <string_view>

#include "holoscan/schema/flatbuffer_schema_traits.hpp"

namespace isaac::os::ops {

/// Per-message-type binding for typed UCX operators.
///
/// Each specialization provides:
///   - `RootType`          : the FlatBuffer table type.
///   - `NativeType`        : the graph payload; may differ from the wire root's native type.
///   - `kSchemaName`       : legacy UCX wire schema name, independent of graph identity.
///   - `kHasSeparateTensor`: true if and only if a large device-memory tensor is sent on
///                           a second UCX tag (currently only `ImageT`).
///
/// Each specialization provides decode, encode, and timestamp functions.
/// Joint and IMU codecs convert wire layouts; Image and Time reuse SDK tables.
/// The graph schema identity belongs to NativeType; kSchemaName identifies
/// the distinct UCX wire protocol when a conversion is present.
template <typename T>
struct UcxxMessageTraits;

}  // namespace isaac::os::ops

#include "operators/ucxx_ops/message_traits_inl.h"

#endif  // ISAAC_OS_OPERATORS_UCXX_OPS_MESSAGE_TRAITS_H
