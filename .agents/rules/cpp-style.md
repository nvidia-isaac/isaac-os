---
paths:
  - "**/*.cc"
  - "**/*.cpp"
  - "**/*.cuh"
  - "**/*.cu"
  - "**/*.cxx"
  - "**/*.h"
  - "**/*.hpp"
---

# C++ Style Rules

## Namespaces

- Do not qualify a name with the namespace that already encloses it. For example, inside
  `namespace isaac::os::ops`, write `McapReplayerOp::TopicMapping`, not
  `isaac::os::ops::McapReplayerOp::TopicMapping`.
- Keep qualifiers that identify a nested namespace or an external type, such as `ops::Type` inside
  `namespace isaac::os` or `holoscan::Tensor`.
- Put operator classes in `isaac::os::ops` and suffix their class names with `Op`.
- Do not repeat a Holoscan SDK major version in C++ namespaces or symbols when the source tree and
  separate build graph already identify that version.
