# camera_info_cpu_messages

CPU-resident flatbuffer type for calibrated camera intrinsics, extrinsics, and distortion.

| Table | Mirrors | Notes |
| --- | --- | --- |
| `CameraInfoCpu` | `sensor_msgs/CameraInfo` | `k`/`r` are `holoscan.schema.Matrix3d`; `p` is `isaac.Matrix3x4d` (from `math_cpu_messages` -- no 3x4 shape exists upstream); `d` (distortion coefficients) stays a variable-length `[double]`, since different distortion models use different coefficient counts |

See `common/ros2_bridge/converter.h` for the ROS2 <-> hsdk conversion that uses this type.
