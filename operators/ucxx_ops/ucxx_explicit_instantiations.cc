/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Explicit template instantiations force compilation of every supported
// message type from a single translation unit. This catches type errors at
// library build (rather than at first consumer link) and keeps object-file
// bloat concentrated here rather than duplicated in every consuming subgraph.

#include <concepts>

#include "holoscan/core/sample.hpp"

#include "operators/ucxx_ops/ucxx_receiver_op.h"
#include "operators/ucxx_ops/ucxx_sender_op.h"

namespace isaac::os::ops {

// Verify the schema-identity chain: every native FBS type used as a port
// payload must satisfy HSDK5's PublishedPayload concept.
static_assert(::holoscan::PublishedPayload<isaac::JointStateCpuT>);
static_assert(::holoscan::PublishedPayload<isaac::JointCommandCpuT>);
static_assert(::holoscan::PublishedPayload<holoscan::schema::TimeT>);
static_assert(::holoscan::PublishedPayload<holoscan::schema::ImuT>);
static_assert(::holoscan::PublishedPayload<holoscan::schema::ImageT>);

template class UcxxSenderOp<isaac::JointStateCpuT>;
template class UcxxSenderOp<isaac::JointCommandCpuT>;
template class UcxxSenderOp<holoscan::schema::TimeT>;
template class UcxxSenderOp<holoscan::schema::ImuT>;
template class UcxxSenderOp<holoscan::schema::ImageT>;

template class UcxxReceiverOp<isaac::JointStateCpuT>;
template class UcxxReceiverOp<isaac::JointCommandCpuT>;
template class UcxxReceiverOp<holoscan::schema::TimeT>;
template class UcxxReceiverOp<holoscan::schema::ImuT>;
template class UcxxReceiverOp<holoscan::schema::ImageT>;

}  // namespace isaac::os::ops
