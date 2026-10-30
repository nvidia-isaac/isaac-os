# Resource Requirements Test Project

This directory contains a standalone CMake test project for validating resource behavior
requirements documented in [`resource_management_requirements.md`][1].

## Scope

The suite runs scenario-driven tests against [`isaac_add_resource()`][2] behavior using local
`file://` artifacts and a small loopback HTTP fixture (no external downloads).

Test suite organization is scenario-first: one test per scenario, listed in [`resource_management_requirements.md`][1].

The suite also covers `isaac_add_resource_bundle()`, which composes independently cached resources
under one runtime directory:

```cmake
isaac_add_resource_bundle(models
  LINK_DIR "${CMAKE_SOURCE_DIR}/data/models"

  RESOURCE encoder
    URL "${MODEL_URL}/encoder.onnx"
    EXPECTED_HASH SHA256=<hash>

  RESOURCE encoder_data
    URL "${MODEL_URL}/encoder.onnx.data"
    EXPECTED_HASH SHA256=<hash>
)
```

`RESOURCE` is a reserved delimiter. Each member accepts the ordinary resource content arguments,
but not `LINK_DIR` or `CONFIG_TIME`; optional empty values should be omitted. File members appear at
the bundle root by URL basename. Archive members appear as directories named after their member
targets and link to their resolved archive views.

Members keep independent download caches. The bundle cache stores only relative member symlinks and
generated files written beside root members. Its order-independent identity changes when any member
view changes, so root-level generated engines are reused for the same complete composition and
invalidated for a changed composition. Files generated within an archive mount live in that archive
member's cache and remain independently reusable.

Assembly preserves undeclared generated files, repairs missing or stale member symlinks, and refuses
to replace a real object at a declared member destination. The public `LINK_DIR` is managed like an
ordinary resource link and defaults to `data/<bundle-name>`. Old SHA-versioned bundle directories
are retained and require the same external cache-pruning policy as ordinary resources.

[1]: ./resource_management_requirements.md
[2]: ../../isaac_add_resource.cmake

## HTTP download coverage

The HTTP fixture covers interruption and restart (SIGINT and SIGKILL), connection
loss, exhausted retries, range fallbacks, changed identity and content, checksum
rejection, concurrent and orphaned transport writers, live Ninja progress, and
the explicit download phase. SIGKILL exercises abrupt process termination; the
tests do not reboot the host or download external models.

## Run

Python 3, curl, CMake, Make, and Ninja must be available. From repository root:

```bash
cmake -S cmake/tests/resource -B cmake/tests/resource/build -DISAAC_BUILD_TESTS=ON
ctest --test-dir cmake/tests/resource/build --output-on-failure
```

Or run the Python suite directly:

```bash
python3 cmake/tests/resource/test_resource_requirements.py
```

For model download and recovery instructions, see
[Managing model downloads](../../../docs/how-to/model_downloads.mdx).
