# geometry_cpu_messages

CPU-resident flatbuffer geometry types, built from `holoscan.schema` primitives
(`Vector3d`/`Quaterniond` -- double-precision, matching `geometry_msgs`' own `float64` fields
exactly, unlike the SDK's older `Vector3f`/`Quaternionf`). Each mirrors a pair of ROS2
`geometry_msgs` types -- the plain and `Stamped` variants convert to/from the same table, since
the only difference between them is whether `header` (and, for transforms, `child_frame_id`) is
populated.

| Table | Mirrors | Notes |
| --- | --- | --- |
| `PoseCpu` | `geometry_msgs/Pose`, `geometry_msgs/PoseStamped` | position + orientation |
| `TwistCpu` | `geometry_msgs/Twist`, `geometry_msgs/TwistStamped` | linear + angular velocity |
| `TransformCpu` | `geometry_msgs/Transform`, `geometry_msgs/TransformStamped` | translation + rotation |
| `OdometryCpu` | `nav_msgs/Odometry` | embeds `PoseCpu`/`TwistCpu` (unstamped -- this table's own `header` is the single source of timestamp/frame) plus `Matrix6d` pose/twist covariances |

See `common/ros2_bridge/converter.h` for the ROS2 <-> hsdk conversions that use these
types.
