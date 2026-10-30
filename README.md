# Isaac OS

<div align="center">

[![Docs](https://img.shields.io/badge/docs-docs.nvidia.com%2Fisaac--os-76B900?logo=nvidia&logoColor=white)](https://docs.nvidia.com/isaac-os/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE.txt)
[![Validated on Jetson AGX Thor](https://img.shields.io/badge/validated_on-Jetson_AGX_Thor-76B900)](https://docs.nvidia.com/isaac-os/reference/platforms)
[![Built on Holoscan, supported by NVIDIA AI Enterprise](https://img.shields.io/badge/NVIDIA_AI_Enterprise-Holoscan-76B900)](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/)

[![Agentic ready: skills, MCP and prompts](https://img.shields.io/badge/%E2%9C%A8_agentic_ready-skills_%C2%B7_MCP_%C2%B7_prompts-76B900?style=for-the-badge&labelColor=000000)](https://docs.nvidia.com/isaac-os/for-agents)

</div>

**Enable robotics developers to build and deploy enterprise-grade physical AI robot
applications with speed-of-light performance on NVIDIA Edge hardware.**

| ⚡ Speed of light | 🛡️ Enterprise readiness | ↔️ ROS 2 interop |
| :--- | :--- | :--- |
| Optimized end to end for NVIDIA hardware, for deterministic real-time control next to AI inference. [See the numbers](#performance). | Built for production robots on the enterprise-supported [Holoscan platform](https://holoscan.org). [Ask about NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/contact-sales/). | Connect your ROS 2 node with the [ROS 2 Bridge](https://docs.nvidia.com/isaac-os/how-to/connect-a-ros-2-graph). See how ROS 2 can map to Isaac OS [here](https://docs.nvidia.com/isaac-os/explanation/isaac-os-and-ros-2). |

## Full robotics stack

Isaac OS is the NVIDIA-native robotics stack for deploying humanoids, mobile robots
and manipulators. It brings together:

- **NVIDIA Edge hardware:** AGX and IGX [platforms](https://docs.nvidia.com/isaac-os/reference/platforms).
- **NVIDIA system software:** JetPack, IGX OS, CUDA and TensorRT.
- **Holoscan-based middleware and reusable components:** compute graphs, message
  passing and GPU resource management for low-latency control alongside AI inference.
- **Developer resources:** reference applications, performance benchmarks, tutorials
  and agent skills.

This repository provides the reference applications and developer resources.

## Performance

**Real-time control and AI inference on the same chip.** Isaac OS partitions
CPU and GPU resources so control loops keep their deadlines and determinism while AI
policies and other heavy workloads share the hardware. Measured on Jetson AGX Thor with about
one hour of samples per condition:

| <div align="center">🚶 7× lower WBC tail latency, 13× less jitter</div> | <div align="center">🖥️ 25× lower GPU control latency</div> | <div align="center">⏱️ Under 19 µs on the CPU</div> |
| :---: | :---: | :---: |
| GR00T N1.7 runs at 5 Hz on the same GPU. With Green Context, WBC p99.9 latency: **2.57 → 0.37 ms**, jitter **3.00 → 0.23 ms**. | 1 kHz GPU IK, 1,024 candidates in parallel, under heavy GPU load. With Green Context isolation: p99.99 **6.60 → 0.27 ms**, late commands **3.1 million → 0**. | 1 kHz CPU IK, one solve per cycle, on isolated PREEMPT_RT cores: p99.99 **under 19 µs** with or without stress-ng, **0** late commands. |
| [Full benchmark](applications/gr00t_n1_7_wbc_g1/BENCHMARK.md) | [Full benchmark](applications/robot_control_benchmark/README.md) | [Full benchmark](applications/robot_control_benchmark/README.md) |

## Try it now

**Run GR00T N1.7 and whole-body control on a simulated G1 in two commands.**
Start with
[Getting started](https://docs.nvidia.com/isaac-os/tutorials/getting-started)
for host setup and checkout. Run commands from the repository root; the
simulation examples also use
[Isaac Sim setup](https://docs.nvidia.com/isaac-os/how-to/set-up-isaac-sim).
These commands run on x86_64 only.

```bash
# Terminal 1: simulator. Start the two terminals in either order.
./isaac_sim run --scene operators/unitree_g1/assets/g1_static_apple_ucx.usda \
  -- --substeps 4

# Terminal 2: controller. The first run builds five TensorRT engines (several minutes).
./isaac_os run gr00t_n1_7_wbc_g1 sim
```

Set the same `DISPLAY` in both terminals to see the robot. See the
[Isaac Sim quick start](applications/gr00t_n1_7_wbc_g1/README.md#isaac-sim-quick-start)
for details. For a run without a simulator, try the
[SO-101 mock in Getting started](https://docs.nvidia.com/isaac-os/tutorials/getting-started).

## Go further

- **Evaluate:** explore [application results](applications/gr00t_n1_7_wbc_g1/BENCHMARK.md#results)
  and [platform support](https://docs.nvidia.com/isaac-os/reference/platforms).
- **Try:** follow
  [SO-101 reach deployment](https://docs.nvidia.com/isaac-os/tutorials/so101-reach/deployment)
  or
  [GR00T and whole-body control tutorial](https://docs.nvidia.com/isaac-os/tutorials/gr00t-wbc).
- **Adapt:** [integrate components](https://docs.nvidia.com/isaac-os/how-to/integrate-in-your-project),
  [add a robot embodiment](https://docs.nvidia.com/isaac-os/how-to/add-a-robot-embodiment),
  or
  [exchange messages with ROS 2](https://docs.nvidia.com/isaac-os/how-to/connect-a-ros-2-graph).

This project is currently not accepting contributions.

## Repository

- `isaac_os` builds, tests, and runs applications; `isaac_sim` launches Isaac Sim.
- [Dockerfile](Dockerfile): reference toolchain and container environment for builds and runs.
- [applications/](applications/): application graphs, configs, and `metadata.json` run modes.
- [operators/](operators/): reusable Holoscan operators and subgraphs.
- [common/](common/): shared libraries and utilities.
- [messages/](messages/): FlatBuffers message schemas.
