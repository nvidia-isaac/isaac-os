/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>

#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include <fmt/format.h>

namespace isaac::os::ros2_bridge {

// Shared by every hsdk-to-ROS2 conversion that splits a nanosecond timestamp against ROS2's
// builtin_interfaces/Time sec+nanosec pair -- converter.h's stamped message converters and
// tensor_converter.h's TensorConverter<sensor_msgs::msg::Image, holoscan::schema::ImageT> both
// use this.
inline constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000;

// builtin_interfaces/Time's own field pair: seconds and the sub-second remainder in
// nanoseconds.
struct Ros2Timestamp {
  std::int32_t sec;
  std::uint32_t nanosec;
};

// Splits a nanosecond timestamp into ROS2's (int32 sec, uint32 nanosec) pair. capture_timestamp_ns
// is int64, wider than builtin_interfaces/Time's int32 sec field -- rejects a timestamp that
// doesn't fit rather than silently truncating or wrapping it into a wrong-but-plausible value.
[[nodiscard]] inline holoscan::expected<Ros2Timestamp, holoscan::Error> to_ros2_timestamp(
    std::int64_t capture_timestamp_ns) {
  std::int64_t sec = capture_timestamp_ns / kNanosecondsPerSecond;
  std::int64_t nanosec = capture_timestamp_ns % kNanosecondsPerSecond;
  // Normalize to floor-division so nanosec stays in builtin_interfaces/Time's required [0, 1e9)
  // range even for a negative capture_timestamp_ns (operator% alone would leave a negative
  // remainder, which wraps to a huge value once cast to uint32_t).
  if (nanosec < 0) {
    nanosec += kNanosecondsPerSecond;
    sec -= 1;
  }
  if (sec < std::numeric_limits<std::int32_t>::min() ||
      sec > std::numeric_limits<std::int32_t>::max()) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument,
        fmt::format("capture_timestamp_ns ({}) does not fit in builtin_interfaces/Time's int32 "
                    "seconds field.",
                    capture_timestamp_ns)});
  }
  return Ros2Timestamp{.sec = static_cast<std::int32_t>(sec),
                       .nanosec = static_cast<std::uint32_t>(nanosec)};
}

// Combines ROS2's (int32 sec, uint32 nanosec) pair into a nanosecond timestamp. Rejects a pair that
// builtin_interfaces/Time does not allow -- nanosec at or above one second, or negative seconds --
// rather than producing a time that is plausible but wrong. The int64 result cannot overflow:
// int32 seconds in nanoseconds stays below 2.2e18.
[[nodiscard]] inline holoscan::expected<std::int64_t, holoscan::Error> from_ros2_timestamp(
    std::int32_t sec, std::uint32_t nanosec) {
  if (sec < 0 || nanosec >= static_cast<std::uint64_t>(kNanosecondsPerSecond)) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument,
        fmt::format("header.stamp (sec={}, nanosec={}) is not a valid capture time: seconds must "
                    "be non-negative and nanosec below one second.",
                    sec, nanosec)});
  }
  return static_cast<std::int64_t>(sec) * kNanosecondsPerSecond +
         static_cast<std::int64_t>(nanosec);
}

}  // namespace isaac::os::ros2_bridge
