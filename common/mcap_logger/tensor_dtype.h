/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlpack/dlpack.h>

#include <optional>

#include "tensor_frame_generated.h"

namespace isaac::os::mcap_logger {

// The two directions of the correspondence between a DLDataType and tensor_frame.fbs's
// TensorDType, kept together so the writer (ToTensorDType) and reader (FromTensorDType) cannot
// drift apart.

// nullopt for any DLDataType TensorDType has no matching value for (non-scalar lanes, or a
// code/bits combination outside the 8 numeric types tensor_frame.fbs's TensorDType enumerates).
[[nodiscard]] inline std::optional<::isaac::mcap_logger::TensorDType> ToTensorDType(
    DLDataType dtype) {
  if (dtype.lanes != 1) {
    return std::nullopt;
  }
  switch (dtype.code) {
    case kDLInt:
      switch (dtype.bits) {
        case 8:
          return ::isaac::mcap_logger::TensorDType_Int8;
        case 16:
          return ::isaac::mcap_logger::TensorDType_Int16;
        case 32:
          return ::isaac::mcap_logger::TensorDType_Int32;
        default:
          return std::nullopt;
      }
    case kDLUInt:
      switch (dtype.bits) {
        case 8:
          return ::isaac::mcap_logger::TensorDType_UInt8;
        case 16:
          return ::isaac::mcap_logger::TensorDType_UInt16;
        case 32:
          return ::isaac::mcap_logger::TensorDType_UInt32;
        default:
          return std::nullopt;
      }
    case kDLFloat:
      switch (dtype.bits) {
        case 32:
          return ::isaac::mcap_logger::TensorDType_Float32;
        case 64:
          return ::isaac::mcap_logger::TensorDType_Float64;
        default:
          return std::nullopt;
      }
    default:
      return std::nullopt;
  }
}

// nullopt for a recorded value outside TensorDType's enumeration. The FlatBuffers verifier checks
// only that the field is present and the right width, so a malformed recording can still reach
// here with a value ToTensorDType() could never have written.
[[nodiscard]] inline std::optional<DLDataType> FromTensorDType(
    ::isaac::mcap_logger::TensorDType dtype) {
  switch (dtype) {
    case ::isaac::mcap_logger::TensorDType_UInt8:
      return DLDataType{kDLUInt, 8U, 1U};
    case ::isaac::mcap_logger::TensorDType_Int8:
      return DLDataType{kDLInt, 8U, 1U};
    case ::isaac::mcap_logger::TensorDType_UInt16:
      return DLDataType{kDLUInt, 16U, 1U};
    case ::isaac::mcap_logger::TensorDType_Int16:
      return DLDataType{kDLInt, 16U, 1U};
    case ::isaac::mcap_logger::TensorDType_UInt32:
      return DLDataType{kDLUInt, 32U, 1U};
    case ::isaac::mcap_logger::TensorDType_Int32:
      return DLDataType{kDLInt, 32U, 1U};
    case ::isaac::mcap_logger::TensorDType_Float32:
      return DLDataType{kDLFloat, 32U, 1U};
    case ::isaac::mcap_logger::TensorDType_Float64:
      return DLDataType{kDLFloat, 64U, 1U};
  }
  return std::nullopt;
}

}  // namespace isaac::os::mcap_logger
