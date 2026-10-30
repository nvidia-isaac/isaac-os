/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/camera_info/camera_info_op.h"

int main() {
  isaac::os::ops::CameraInfoOp op;
  static_cast<void>(op.contract());
  return 0;
}
