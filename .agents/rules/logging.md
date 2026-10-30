---
paths:
  - "**/*.cc"
  - "**/*.cpp"
  - "**/*.cuh"
  - "**/*.cu"
  - "**/*.cxx"
  - "**/*.h"
  - "**/*.hpp"
  - "**/*.py"
---

# Logging Rules

Terminal logs and exception messages are an operator-facing interface. They must let a reader
understand system state during setup, normal operation, and incident response without inspecting
the source code.

## Write for the reader

- State the observable outcome first, then its impact or next step when useful.
- Use plain language and complete sentences.
- Preserve actionable details: user-facing names, paths, counts, timestamps, device information,
  and the underlying error text.
- Describe configuration by its effect. For example, say that visualization is disabled instead of
  naming the configuration field that controls it.
- Apply the same standard to exception text that can reach the terminal.

## Do not expose implementation details

Do not use class, method, variable, configuration-field, or API names as message labels. Avoid
code fragments, assignment-style output, and internal identifiers that a reader would need source
code to interpret.

| Avoid | Prefer |
| --- | --- |
| `VideoEncoder::encode: failed` | `Unable to encode the camera frame: {error}.` |
| `enable_visualization = false` | `Visualization is disabled.` |
| `buffer_size_ = 0` | `No buffer space is available.` |

Keep a technical identifier when it is meaningful to an operator, such as a file path, camera
name, topic, connection number, standard scheduling policy, or error returned by a dependency.
