// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { convertTimestamp } from "./common";

describe("convertTimestamp", () => {
  it("logs and returns undefined when the timestamp is undefined", () => {
    const debugSpy = jest.spyOn(console, "debug").mockImplementation(() => undefined);
    expect(convertTimestamp(undefined)).toBeUndefined();
    expect(debugSpy).toHaveBeenCalledWith(expect.stringMatching(/missing/));
    debugSpy.mockRestore();
  });

  it("logs and returns undefined when the timestamp is null", () => {
    // Runtime-only case: TS's declared type is bigint | undefined, but callers read this off
    // untyped/external message data, so a literal null can still reach here at runtime --
    // convertTimestamp's own `== null` check must catch it too.
    const debugSpy = jest.spyOn(console, "debug").mockImplementation(() => undefined);
    expect(convertTimestamp(null as unknown as undefined)).toBeUndefined();
    expect(debugSpy).toHaveBeenCalledWith(expect.stringMatching(/missing/));
    debugSpy.mockRestore();
  });

  it("converts a bigint nanosecond count into {sec, nsec}", () => {
    expect(convertTimestamp(BigInt(5_000_000_123))).toEqual({ sec: 5, nsec: 123 });
  });

  it("handles zero without treating it as a missing value", () => {
    expect(convertTimestamp(BigInt(0))).toEqual({ sec: 0, nsec: 0 });
  });

  it("stays within Number.MAX_SAFE_INTEGER at the full int64 nanosecond range", () => {
    const maxInt64Ns = (BigInt(1) << BigInt(63)) - BigInt(1); // 9223372036854775807n
    const result = convertTimestamp(maxInt64Ns);
    expect(result).toBeDefined();
    expect(Number.isSafeInteger(result?.sec)).toBe(true);
    expect(Number.isSafeInteger(result?.nsec)).toBe(true);
    expect(result).toEqual({ sec: 9223372036, nsec: 854775807 });
  });

  it("floor-normalizes negative timestamps so nsec stays non-negative", () => {
    // BigInt's %//truncate toward zero: -1n % 1e9n === -1n, not the 999_999_999n a
    // POSIX-timespec-style {sec, nsec} contract requires. -1ns is one nanosecond before the
    // epoch, i.e. sec=-1, nsec=999_999_999 -- not sec=0, nsec=-1.
    expect(convertTimestamp(BigInt(-1))).toEqual({ sec: -1, nsec: 999_999_999 });
  });

  it("floor-normalizes a negative timestamp that lands exactly on a second boundary", () => {
    expect(convertTimestamp(BigInt(-5_000_000_000))).toEqual({ sec: -5, nsec: 0 });
  });

  it("floor-normalizes a negative timestamp with a non-zero sub-second remainder", () => {
    // -4_999_999_877ns is 123ns after -5s (-5_000_000_000ns + 123ns), i.e. sec=-5, nsec=123.
    expect(convertTimestamp(BigInt(-4_999_999_877))).toEqual({ sec: -5, nsec: 123 });
  });
});
