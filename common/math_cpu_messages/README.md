# math_cpu_messages

Reusable fixed-size double-precision matrix types, `native_type`-backed by Eigen so their native
(`...T`) representation is a real `Eigen::Matrix`, not a hand-rolled wrapper. Each is explicitly
`Eigen::RowMajor` (Eigen defaults to column-major) so its in-memory layout matches the row-major
wire layout exactly, letting `Pack`/`UnPack` reinterpret the buffer directly via `Eigen::Map`
instead of copying element-by-element.

| Struct | Native type | Notes |
| --- | --- | --- |
| `Matrix3x4d` | `Eigen::Matrix<double, 3, 4, Eigen::RowMajor>` | e.g. a camera projection matrix |
| `Matrix6d` | `Eigen::Matrix<double, 6, 6, Eigen::RowMajor>` | e.g. a pose/twist covariance matrix |

Neither shape exists in `holoscan.schema` (the SDK's `geometry.fbs`). Embeddable structs, not
standalone messages -- no `root_type`, consumed via `include` from `camera_info_cpu_messages` and
`geometry_cpu_messages`.

A double-precision 3x3 (`camera_info_cpu_messages`' `k`/`r`) doesn't need a type here:
`holoscan.schema.Matrix3d` (added alongside `Vector3d`/`Quaterniond` in `geometry.fbs`) already
covers that shape -- plain flatc struct, not `native_type`-backed, so it's `.m()->Get(i)`
element access rather than `Eigen::Matrix`'s `(row, col)`, same as `Matrix3f` elsewhere in the
SDK. Prefer it over adding another local 3x3 type once a shape is already covered upstream.

Each type's `Pack`/`UnPack` is compiled (not header-only, unlike its sibling `*_cpu_messages`
libs): flatc places a `.fbs`'s `native_include` before the struct's own definition in the
generated header, so the actual function bodies can't live in the `native_include` target
itself. `matrix*d_adapter.h` (referenced by `native_include`) has declarations only;
`matrix*d_adapter.cc` has the ordinary (non-inline) definitions, so consumers just need the
declaration in scope -- the linker resolves the rest as long as `math_cpu_messages` is linked
in.
