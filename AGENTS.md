# Isaac OS Agent Instructions

## Project Overview

Isaac OS is a **Physical AI robotics platform** built on **NVIDIA Holoscan SDK**, written in
**C++ and Python with CUDA**, built with **CMake**.

For the repository layout, see [Repository](README.md#repository). For applications,
operators, subgraphs and messages, see [Isaac OS concepts](docs/explanation/isaac_os_basics.mdx).

## Rules



### Skill Authoring

When adding or modifying `skills/**`, `.agents/skills/**`, any `SKILL.md`, or
`docs/for_agents.mdx`, follow `.agents/rules/skill-authoring.md`.

### CMake

When modifying `**/CMakeLists.txt`, `**/*.cmake`, or `cmake/**`, follow `.agents/rules/cmake.md`.

### C++ Style

When modifying `**/*.cc`, `**/*.cpp`, `**/*.cuh`, `**/*.cu`, `**/*.cxx`, `**/*.h`, or
`**/*.hpp`, follow `.agents/rules/cpp-style.md`.

### Docker

When modifying `**/Dockerfile*`, follow `.agents/rules/docker.md`.

### Logging

When adding or modifying terminal log or exception messages in source code, follow
`.agents/rules/logging.md`.

### README and overview

When modifying `README.md` or `docs/overview.mdx`, keep their shared sections in
sync. Follow [Edit documentation](docs/README.md#edit-documentation).

### Spelling

Use American English spelling in documentation, comments, and log messages, for
example "meter", "color", "behavior", and "optimize".

### Prose

Use plain language (ASD-STE100) in documentation, skills, agent prompts, and comments:
short sentences, common words, and one instruction per sentence. Stick to precise
semantics to ensure you do not sacrifice accuracy.

### Shared CMake caches across worktrees

For local `./isaac_os` builds with multiple worktrees:

- Find and reuse an existing Isaac OS cache under `$HOME` or in the main checkout
  (not a worktree). Warn if more than one exists. If none exists, use
  `$HOME/.cache/isaac-os`.
- Use the same host cache and container paths for every `./isaac_os` invocation
  that reuses a build directory. CMake stores these paths in
  `build/<app>/CMakeCache.txt`, so the variable must be present for every
  invocation:

  ```bash
  export ISAAC_OS_EXTRA_DOCKER_RUN_ARGS='-v <host-cache>:/cache/isaac-os -e CPM_SOURCE_CACHE=/cache/isaac-os/cpm -e ISAAC_RESOURCE_CACHE=/cache/isaac-os/resources'
  ```

- If a build directory was configured with different cache paths, clear its
  CMake cache or reconfigure it once. Pass each definition as a separate
  `--configure-args` value:

  ```bash
  ./isaac_os build <app> \
    --configure-args='-DCPM_SOURCE_CACHE=/cache/isaac-os/cpm' \
    --configure-args='-DISAAC_RESOURCE_CACHE=/cache/isaac-os/resources'
  ```

- The generated `data/` symlinks are relative, but targets under
  `/cache/isaac-os` resolve only inside a container using this mapping.

A missing mount surfaces as `creation failed (check permissions)` under
`/cache/isaac-os`. `/cache` does not exist in the image, so this is an absent
mount rather than a permission problem.

## Agent Skills

Product skills live in `skills/`. See [For agents](docs/for_agents.mdx) for
the Isaac OS task workflow.


