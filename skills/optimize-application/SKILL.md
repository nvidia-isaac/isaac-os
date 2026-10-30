---
name: optimize-application
description: >
  Optimize Isaac OS / Holoscan application performance and determinism. Use for
  latency or throughput bottlenecks, tail-latency outliers, interference between
  mixed-rate workloads, CPU placement, CUDA Green Contexts, buffer reuse, and
  CUDA Graphs. For profiling or reporting alone, use data-flow-profiling.
license: Apache-2.0
metadata:
  author: "Connor Smith <cosmith@nvidia.com>"
---

# Optimize an Application

Read [Optimize an Application](../../docs/how-to/optimize_an_application.mdx)
for the optimization techniques and Holoscan examples. Keep that guide as the
canonical source; check the application's pinned SDK version before adapting APIs.

Identify the application, run mode, critical workloads, and performance target
from the request and available configuration. Establish which output rates,
sample-age limits, and numerical behavior must be preserved before changing
queueing, scheduling, or computation.

For pipeline timing and message rates, use the
[data-flow-profiling skill](../data-flow-profiling/SKILL.md) when available,
or follow [Profile an Application](../../docs/how-to/profile_an_application.mdx).
Follow the guide's Nsight workflow when diagnosis needs CUDA activity or CPU
scheduling detail. For CUDA SQLite analysis, read
[Nsight diagnostic queries](../../tools/profiling/nsight_queries.md).

Choose the guide's technique that addresses the observed delay. Prefer Holoscan's
placement, Tensor storage, and stream contracts over custom runtime machinery.
Do not remove synchronization without preserving the dependency it enforces, or
infer a specific GPU contention mechanism from a launch gap alone.

After a change, compare the affected workload under comparable conditions,
including competing work. Check the requested timing metric alongside output
rates, freshness, and correctness. Report the evidence, configuration, and any
validation that could not be performed; do not present an observed maximum as a
worst-case guarantee.

## Links

The links in this skill are paths in an Isaac OS checkout. Without a checkout, find
docs pages in [llms.txt](https://docs.nvidia.com/isaac-os/llms.txt) for your Isaac OS
version, and other files in the
[source repository](https://github.com/nvidia-isaac/isaac-os) at your tag or commit.
If your version is not in `llms.txt`, read `docs/` in the source repository.
