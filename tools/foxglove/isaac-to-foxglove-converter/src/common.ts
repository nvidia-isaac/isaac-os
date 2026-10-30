// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for the isaac.*Cpu and TensorFrame converters.

// Convert a raw nanosecond count into Foxglove's {sec, nsec} timestamp shape. bigint only --
// every producer this converts (Foxglove's own generic flatbuffer decode of an int64 field, and
// this file's own generated FlatBuffers bindings, e.g. Header.captureTimestampNs()) hands back a
// bigint, never a number or string; narrowing the parameter to what callers actually pass avoids
// a footgun like BigInt("") (== 0n, not an error) or BigInt(1.5) (throws a bare RangeError,
// preempting this function's own diagnostics) sneaking in through a wider, unused type.
//
// Logs and returns undefined on a missing value, matching every other soft-failure in this file
// (missing position, missing tensor data, unsupported encoding, ...) -- one message with a bad
// timestamp should drop that message, not throw out of the converter and take the rest of the
// panel down with it (a converter registered via registerMessageConverter is not wrapped in a
// try/catch by this extension; Foxglove's host is what would ultimately catch a throw here, and
// its behavior on that path isn't something this codebase controls).
export function convertTimestamp(
  timeNsValue: bigint | undefined,
): { sec: number; nsec: number } | undefined {
  if (timeNsValue == null) {
    console.debug("convertTimestamp: timeNsValue is missing (null or undefined).");
    return undefined;
  }
  const nsPerSec = BigInt(1_000_000_000);
  // BigInt %/ truncate toward zero, not floor -- for a negative timeNsValue that leaves a
  // negative remainder (e.g. -1n % 1e9n === -1n), violating {sec, nsec}'s POSIX-timespec-style
  // contract that nsec always lands in [0, 1e9) regardless of sec's sign. Normalize to floor
  // division so nsec stays non-negative, borrowing a second from sec when the truncated
  // remainder is negative.
  let sec = timeNsValue / nsPerSec;
  let nsec = timeNsValue % nsPerSec;
  if (nsec < BigInt(0)) {
    nsec += nsPerSec;
    sec -= BigInt(1);
  }
  // Safe even at the full int64 range (+-2^63 ns): sec tops out around 9.22e9, nsec is always in
  // [0, 1e9), and both stay far inside Number.MAX_SAFE_INTEGER (~9.007e15) -- no clamping needed.
  return { sec: Number(sec), nsec: Number(nsec) };
}

// Calculate a quaternion that rotates the +X axis to point along a target direction vector, used
// to orient a foxglove.SceneUpdate arrow along a velocity vector.
export function quaternionFromXAxisToDirection(
  dx: number,
  dy: number,
  dz: number,
): { x: number; y: number; z: number; w: number } {
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (len < 0.0001) {
    return { x: 0, y: 0, z: 0, w: 1 };
  }

  const nx = dx / len;
  const ny = dy / len;
  const nz = dz / len;

  const x_axis = { x: 1, y: 0, z: 0 };

  const axis_x = x_axis.y * nz - x_axis.z * ny;
  const axis_y = x_axis.z * nx - x_axis.x * nz;
  const axis_z = x_axis.x * ny - x_axis.y * nx;

  const dot = x_axis.x * nx + x_axis.y * ny + x_axis.z * nz;

  if (Math.abs(dot - 1) < 0.0001) {
    return { x: 0, y: 0, z: 0, w: 1 };
  } else if (Math.abs(dot + 1) < 0.0001) {
    return { x: 0, y: 0, z: 1, w: 0 };
  }

  const angle = Math.acos(dot);
  const sin_half = Math.sin(angle / 2);
  const cos_half = Math.cos(angle / 2);

  const axis_len = Math.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);

  return {
    x: (axis_x / axis_len) * sin_half,
    y: (axis_y / axis_len) * sin_half,
    z: (axis_z / axis_len) * sin_half,
    w: cos_half,
  };
}

export interface ImageEncodingInfo {
  format: string;
  bytesPerPixel: number;
}

// Maps holoscan.schema.ImageEncoding values to Foxglove RawImage encoding strings.
export const imageEncodingMap: { [key: number]: ImageEncodingInfo } = {
  1: { format: "rgb8", bytesPerPixel: 3 }, // RGB8
  2: { format: "rgba8", bytesPerPixel: 4 }, // RGBA8
  3: { format: "bgr8", bytesPerPixel: 3 }, // BGR8
  4: { format: "bgra8", bytesPerPixel: 4 }, // BGRA8
  7: { format: "mono8", bytesPerPixel: 1 }, // MONO8
  8: { format: "mono16", bytesPerPixel: 2 }, // MONO16
  // Add more mappings as needed
};
