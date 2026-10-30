---
paths:
  - "**/CMakeLists.txt"
  - "**/*.cmake"
  - "cmake/**"
---

# CMake Rules

## General

- Use `PROJECT_SOURCE_DIR` and `PROJECT_BINARY_DIR` for project-owned paths; reserve
  `CMAKE_SOURCE_DIR` and `CMAKE_BINARY_DIR` for paths intentionally tied to the entire build.
- Any list of items should be sorted alphabetically unless order matters. For example:
  - consecutive `add_subdirectory` calls
  - targets in `target_link_libraries`
  - source fields in `add_library`, `add_executable`, `target_sources`
  - directories passed to `target_include_directories`

## Tests

- Gate test-only subdirectories, targets, and registration with `ISAAC_BUILD_TESTS`, not CMake's
  `BUILD_TESTING`.

## Targets

- Use modern CMake (target-based) syntax, don't link and include dirs at the directory level.
- Stick to `SHARED` libraries. Create `INTERFACE` libraries for header-only targets.
- Name concrete operator implementation targets with an `_op` suffix. Omit `_op` from their public
  aliases; the `::operators::` segment already provides that context.
- For library targets, add alias based on the relative path to the repository root, each nesting adding a new `::` namespace. E.g. `isaac::<dir>::<subdir>::<name>`.
- Do not include a Holoscan SDK major version in target or alias names.
- Do not repeat a source-tree discriminator in names inside that tree.
- Be diligent about the link libraries:
  - If we `#include foo` in our target headers (bar.h), list foo under `PUBLIC` (or `INTERFACE` if header-only lib).
  - If we `#include foo` in our target sources only (bar.cc), list foo under `PRIVATE`.
  - Do not list transitive dependencies to a target's linked libraries if they're not `#include`d directly in the target headers: it is the responsibility of the direct dependency that is included/linked against to expose that transitive dependency as `PUBLIC` (or `INTERFACE`).
    - However, make sure to `#include` what you call, don't depend on transitive includes to expose symbols you use explicitly.
    - Vice-versa, make sure you don't `#include` what you don't need, particularly consider forward declarations or Pimpl idiom to keep your interface headers less tied to your dependencies.

## Third-party dependencies & data

- All dependencies pulled with CMake should have their `cmake/isaac_deps/<name>` file that gets included with `include(isaac_deps/<name>)` exactly where it is needed.
- Dependencies built from source: use `isaac_cpm_find_package`.
- Pre-built dependencies (config-time): use either:
  - `isaac_add_resource(... CONFIG_TIME)` - sets `<name>_CONTENT_DIR`.
    Prefer when you need pattern-based extraction or prefix stripping.
  - `isaac_cpm_find_package(... DOWNLOAD_ONLY TRUE)` - sets `<name>_SOURCE_DIR`.
    Prefer for simpler archives without extraction filtering.
  - Either must be followed by `find_package` or `find_path`/`find_library` depending on
    whether the dependency ships a CMake config.
- Runtime dependencies (build-time data, models, datasets): use `isaac_add_resource(<name> ...)`. Output at `data/<name>/`.

Note: Patches should be kept under `third_party/<name>/` and applied with
- source: `git apply --check --reverse <patch1> ... <patchN> || git apply <patch1> ... <patchN>`.
- resource: pass the patch file to the `PATCH` argument.
