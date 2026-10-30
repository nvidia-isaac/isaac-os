# robot_control_benchmark common core

Runtime-framework-independent robot-control workload. Do not add Holoscan or ROS types here.

| Header | Role |
| --- | --- |
| `payloads.hpp` | POD tick messages (padding-free) |
| `control_loop.hpp` | `steady_now_ns` and simulated joint snapshot |
| `runtime_support.hpp` | `SimulatedRobot` |
| `latency_stats.hpp` | control-loop timing distributions, correctness counters, and the result row |
| `arm_kinematics.hpp` | `ArmModelT<T>`: FK, the geometric Jacobian, the DLS IK step, and the FK finite-gate |
| `trajectory.hpp` | independent synthetic 6-DoF target |
| `joint_safety.hpp` | range/step clamps |
| `motor_protocol.hpp` | simulated motor batch encoding |
| `motion_backend.hpp` / `.cc` | the `MotionBackend` interface, `cpu-ik`, and the factory |
| `gpu_ik_workload.hpp` | deterministic local seeds, command bounds, and candidate scoring |
| `gpu_motion_ik.hpp` / `.cu` | `gpu-ik` backend owning CUDA resources, 1,024 parallel IK solves, and GPU command selection |

## Motion workloads

- **CPU:** four damped least-squares (DLS) IK iterations from one starting configuration,
  with fixed damping and a fixed per-iteration joint step limit.
- **GPU:** a parallel search over 1,024 nearby starting configurations, each refined with
  four DLS iterations. It selects the best bounded command using target pose error and joint
  movement, with a bounded hold command as the fallback.

Both use `ArmModelT<T>` from `arm_kinematics.hpp`: double precision on the CPU and single
precision for GPU candidates. The shared implementation provides forward kinematics, the
Jacobian, and the DLS update. GPU tests compare selection with an exhaustive host search and
check joint bounds and trajectory tracking in the simulated control loop.

Functions called on the GPU use `RCB_HD` to compile for both host and device. The shared math
uses `Mat4T` and `kin_clamp` to avoid host-only standard-library calls in device code.

## Build notes for other runtimes

This directory defines no CMake library on purpose. Each runtime compiles the sources it needs
by path, which is what lets a runtime outside the Isaac OS CMake graph share the workload.

The `CpuDlsIkBackend` header needs no CUDA. The backend factory in `motion_backend.cc`
provides both `cpu-ik` and the fixed 1,024-seed `gpu-ik` workload; link it with
`gpu_motion_ik.cu`, enable the CUDA language, and set `CUDA_ARCHITECTURES`.
