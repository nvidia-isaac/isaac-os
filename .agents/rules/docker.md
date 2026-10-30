---
paths:
  - "**/Dockerfile*"
---

# Docker Rules

- Use multi-stage builds to reduce the size of the final image.
- Prefer bind mounts over copies if what's needed is only temporary.
- Use mount cache for apt and pip (or package managers) instead of deleting the cache in the layer.
- Keep `.dockerignore` updated for new build-context paths to avoid sending unnecessary files and speed builds.
