---
name: isaac-os-build-and-run
description: >
  Use when building, testing or running Isaac OS applications, components or runtime
  checks in this repository, including MR verification. Not for integrating Isaac OS
  into another project, documentation checks, formatting or host tooling tests.
license: Apache-2.0
metadata:
  author: "Alexis Girault <agirault@nvidia.com>"
  tags:
    - isaac-os
---

# Build, test and run Isaac OS

Read [Build and run](../../../docs/how-to/build_and_run.mdx) for the workflow and
[CLI reference](../../../docs/reference/cli.mdx) for commands and options.

Use `./isaac_os` from the checkout being tested to build, test and run Isaac OS
applications and components. Use its `run-container` command for individual CMake
targets and custom checks in the development environment, as described in the guide.

If the CLI blocks the task, report the failure and explain the alternative before
proceeding.

Check GPU access inside the CLI container before reporting that it is unavailable.
A failed check in the agent's shell is not sufficient evidence.
