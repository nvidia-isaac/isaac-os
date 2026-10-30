/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_UCXX_OPS_UCXX_TYPE_ALIASES_H
#define ISAAC_OS_OPERATORS_UCXX_OPS_UCXX_TYPE_ALIASES_H

// Convenience aliases for the concrete message types shipped subgraphs
// currently exchange. Consumers can also use `UcxxSenderOp<Foo>` /
// `UcxxReceiverOp<Foo>` directly.

#include "operators/ucxx_ops/ucxx_receiver_op.h"
#include "operators/ucxx_ops/ucxx_sender_op.h"

namespace isaac::os::ops {

using JointStateUcxxSenderOp = UcxxSenderOp<isaac::JointStateCpuT>;
using JointStateUcxxReceiverOp = UcxxReceiverOp<isaac::JointStateCpuT>;

using JointCommandUcxxSenderOp = UcxxSenderOp<isaac::JointCommandCpuT>;
using JointCommandUcxxReceiverOp = UcxxReceiverOp<isaac::JointCommandCpuT>;

using TimeUcxxSenderOp = UcxxSenderOp<holoscan::schema::TimeT>;
using TimeUcxxReceiverOp = UcxxReceiverOp<holoscan::schema::TimeT>;

using ImuUcxxSenderOp = UcxxSenderOp<holoscan::schema::ImuT>;
using ImuUcxxReceiverOp = UcxxReceiverOp<holoscan::schema::ImuT>;

using ImageUcxxSenderOp = UcxxSenderOp<holoscan::schema::ImageT>;
using ImageUcxxReceiverOp = UcxxReceiverOp<holoscan::schema::ImageT>;

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_UCXX_OPS_UCXX_TYPE_ALIASES_H
