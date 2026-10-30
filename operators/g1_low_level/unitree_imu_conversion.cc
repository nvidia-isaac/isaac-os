/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/g1_low_level/unitree_imu_conversion.h"

namespace isaac::os {

// These two constructor calls are the only positional writes to the schema's inline structs in
// this tree. The accompanying test asserts both the named read-back and the raw component layout,
// so a schema that reorders its members fails here rather than on the robot.
holoscan::schema::Quaternionf to_schema_orientation(const UnitreeOrientation& orientation) {
  return holoscan::schema::Quaternionf{orientation.x, orientation.y, orientation.z, orientation.w};
}

holoscan::schema::Quaternionf identity_orientation() {
  return to_schema_orientation({.w = 1.0F, .x = 0.0F, .y = 0.0F, .z = 0.0F});
}

holoscan::schema::Vector3f to_schema_vector(const UnitreeVector& value) {
  return holoscan::schema::Vector3f{value.x, value.y, value.z};
}

}  // namespace isaac::os
