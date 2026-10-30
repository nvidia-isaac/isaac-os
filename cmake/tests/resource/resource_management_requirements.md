# Resource Management Requirements (Testable)

This document defines testable, implementation-agnostic requirements for resource behavior.
Each requirement is intended to be independently testable and maps to one or more scenarios.

Implemented in:

- `cmake/isaac_resource_helpers.cmake` - core operations (download, verify, extract, symlink)
- `cmake/isaac_add_resource.cmake` - project wrapper (ISAAC_RESOURCE_CACHE, LINK_DIR defaults)
- `cmake/isaac_resource_ensure_script.cmake` - build-time entry point

Cache layout (one canonical extraction per sha8; recipes are tracked in a sibling stamp):

```log
${ISAAC_RESOURCE_CACHE}/<name>/<sha8>/
  .lock                        # advisory lock for this sha8
  <archive>                    # archive file (kept only with KEEP_ARCHIVE)
  <archive>.hash.verified      # per-sha8 hash verification stamp
  content/                     # single canonical extraction tree
  content.recipes              # one line per recipe already extracted
                               #   "*" = full extraction
                               #   <8-hex> = canonicalized ARCHIVE_PATTERNS hash
```

Recipes are additive: each new ARCHIVE_PATTERNS call extracts matching files
into the same `content/` tree and appends its recipe token to the stamp.
Files that already exist are overwritten with identical content (no-op).
A `*` token covers every possible pattern request, so any later call skips.

---

## Requirements

### 1) Correctness Requirements (Staleness and Recovery)

#### R1.1 Missing non-archive output must be repaired

- **Given:** A non-archive resource was built successfully.
- **When:** The output file is deleted and `build` is run.
- **Then:** The output file is recreated and target build succeeds.

#### R1.2a Missing archive extracted output must be repaired (build_only)

- **Given:** An archive resource was built successfully.
- **When:** The extracted output is deleted and `build` is run.
- **Then:** Extracted output is restored and target build succeeds.

#### R1.2b Missing archive extracted output must be repaired (reconfigure_then_build)

- **Given:** An archive resource was built successfully.
- **When:** The extracted output is deleted, then `configure` and `build` are run.
- **Then:** Extracted output is restored and target build succeeds.

#### R1.3 Expected identity change must force refresh

- **Given:** A resource is currently valid for expected identity `A`.
- **When:** Expected identity is changed to `B` and `build` is run.
- **Then:** The resource is revalidated/rebuilt to `B`; stale artifacts from `A` are not accepted.

#### R1.4a Archive prerequisites must be recoverable when extraction is required (build_only)

- **Given:** An archive resource's extracted output is missing or invalid.
- **When:** The required archive/intermediate artifacts are missing or invalid and `build` is run.
- **Then:** Build recreates prerequisites as needed and successfully restores valid extracted output.

#### R1.4b Archive prerequisites must be recoverable when extraction is required (reconfigure_then_build)

- **Given:** An archive resource's extracted output is missing or invalid.
- **When:** The required archive/intermediate artifacts are missing or invalid, then `configure`
and `build` are run.
- **Then:** Build recreates prerequisites as needed and successfully restores valid extracted output.

#### R1.5 Verification stamps must preserve expected-hash spelling

- **Given:** EXPECTED_HASH has surrounding whitespace and mixed-case algorithm or hex characters.
- **When:** The resource is verified.
- **Then:** Its stamp contains the whitespace-stripped original spelling so existing stamp matching
  remains compatible.

---

### 2) Efficiency Requirements (Skip Unnecessary Work)

For this document, **expensive work** means any unnecessary resource pipeline step:

- redownload,
- reverify/rehash,
- reextract.

#### R2.1 Valid extracted archive output should bypass intermediates

- **Given:** Extracted content is valid for expected identity (archive absent by default).
- **When:** `build` is run.
- **Then:** No unnecessary redownload/reverify/reextract is performed.

#### R2.2 Valid non-archive output should skip verification work

- **Given:** Non-archive output is already valid for expected identity.
- **When:** `build` is run without semantic changes.
- **Then:** Verification work is skipped.

#### R2.3 Reconfigure with unchanged identity should avoid expensive rebuild work

- **Given:** Resource is valid and expected identity is unchanged.
- **When:** `configure` is rerun, then `build`.
- **Then:** Expensive work is not repeated.

#### R2.4a Content stability should dominate timestamp churn (build_only)

- **Given:** No semantic change to expected identity or output validity.
- **When:** Only metadata timestamps change and only `build` runs.
- **Then:** Expensive resource processing should not rerun.

#### R2.4b Content stability should dominate timestamp churn (reconfigure_then_build)

- **Given:** No semantic change to expected identity or output validity.
- **When:** Only metadata timestamps change, then `configure` and `build` run.
- **Then:** Expensive resource processing should not rerun.

---

### 3) Configure-vs-Build Safety Requirements

#### R3.1 Build must recover from post-configure deletions (build_only)

- **Given:** `configure` has completed.
- **When:** Required resource files/stamps are deleted before `build`.
- **Then:** Plain `build` (without reconfigure) must recover or regenerate as needed.

#### R3.2a Configure-time optimization must not hide build-time invalid state (build_only)

- **Given:** Configure-time shortcut/optimization decisions were made.
- **When:** Resource state changes before build and only `build` runs.
- **Then:** `build` must still detect invalid state and not report false success.

#### R3.2b Configure-time optimization must not hide build-time invalid state (reconfigure_then_build)

- **Given:** Configure-time shortcut/optimization decisions were made.
- **When:** Resource state changes before build, then `configure` and `build` run.
- **Then:** `build` must still detect invalid state and not report false success.

#### R3.3 Reconfigure must not be mandatory for correctness recovery (build_only)

- **Given:** Resource state is broken.
- **When:** User runs `build` only.
- **Then:** Correctness should be restorable without requiring `configure`.

---

### 4) Acceptable vs Unacceptable Missing State

#### R4.1 Acceptable: archive absent, extracted output valid

- **Given:** Valid extracted output exists. Archive is absent
  (the default post-extraction state per R8.1).
- **When:** `build` runs.
- **Then:** Target is accepted as up to date; no re-download triggered.

#### R4.2a Unacceptable: extraction stamp present, extracted output missing (build_only)

- **Given:** Extraction stamp exists but extracted output is missing/empty.
- **When:** `build` runs.
- **Then:** State is treated as stale and repaired or clearly failed.

#### R4.2b Unacceptable: extraction stamp present, extracted output missing (reconfigure_then_build)

- **Given:** Extraction stamp exists but extracted output is missing/empty.
- **When:** `configure` is rerun, then `build` runs.
- **Then:** State is treated as stale and repaired or clearly failed.

#### R4.3 Unacceptable: verification stamp present, non-archive output missing

- **Given:** Verification stamp exists but non-archive output file is missing.
- **When:** `build` runs.
- **Then:** State is treated as stale and repaired.

#### R4.4 Missing stamps with valid output should self-heal

- **Given:** Output is valid, but one or more stamps are missing.
- **When:** `build` runs.
- **Then:** Stamps are restored with minimal unnecessary heavy work.

#### R4.5a Missing stamps and missing output should fully recover (build_only)

- **Given:** Both output and metadata are missing.
- **When:** `build` runs.
- **Then:** Full resource pipeline runs successfully.

#### R4.5b Missing stamps and missing output should fully recover (reconfigure_then_build)

- **Given:** Both output and metadata are missing.
- **When:** `configure` is rerun, then `build` runs.
- **Then:** Full resource pipeline runs successfully.

---

### 5) Filtered Extraction Requirements (ARCHIVE_PATTERNS)

#### R5.1 Filtered extraction must only extract matching patterns

- **Given:** An archive with multiple directories/files and an `ARCHIVE_PATTERNS` filter.
- **When:** `build` runs.
- **Then:** Only files matching the patterns are present in the output; non-matching files are absent.

#### R5.2 Filtered extraction must be idempotent

- **Given:** A filtered archive resource was built successfully.
- **When:** `build` is run again without changes.
- **Then:** No re-extraction occurs.

#### R5.3 Filtered extraction must recover after output deletion

- **Given:** A filtered archive resource was built successfully.
- **When:** Extracted output is deleted and `build` is run.
- **Then:** Output is restored with only matching patterns extracted.

---

### 6) Cross-Build Reuse Requirements

#### R6.1 Shared metadata/cache must be reusable across build directories

- **Given:** Build directory A has already prepared resource state.
- **When:** Build directory B builds the same target against same source tree.
- **Then:** Resource state is reused rather than redone from scratch.

#### R6.2a Shared metadata must not produce false-valid outcomes (build_only)

- **Given:** Shared metadata exists across builds.
- **When:** Corresponding output is stale/missing in the active build context and only `build` runs.
- **Then:** Build must not report success incorrectly.

#### R6.2b Shared metadata must not produce false-valid outcomes (reconfigure_then_build)

- **Given:** Shared metadata exists across builds.
- **When:** Corresponding output is stale/missing in the active build context, then `configure`
and `build` run.
- **Then:** Build must not report success incorrectly.

---

### 7) Symlink (LINK_DIR) Requirements

#### R7.1 LINK_DIR symlink must point to content after build

- **Given:** A resource with a LINK_DIR is built.
- **When:** `build` completes.
- **Then:** LINK_DIR is a symlink to the resolved view
  (`CACHE_DIR/<sha8>/content/<STRIP_PREFIX>/`, or just `content/` when
  STRIP_PREFIX is unset).

#### R7.2 LINK_DIR symlink must be recreated after deletion

- **Given:** A resource was built and its LINK_DIR symlink was deleted.
- **When:** `build` runs again.
- **Then:** The symlink is recreated; no re-extraction or re-download occurs.

#### R7.3 No LINK_DIR means no symlink, content in cache only

- **Given:** A config-time resource is built without a LINK_DIR.
- **When:** `build` or configure completes.
- **Then:** No symlink is created; content is accessible directly via
  `CACHE_DIR/<sha8>/content/`.

#### R7.4 Generated sibling files must remain in the resource cache

- **Given:** A non-archive resource is exposed through its LINK_DIR directory.
- **When:** A consumer writes another file beside the downloaded file.
- **Then:** The generated sibling is stored under `CACHE_DIR/<sha8>/content/`.

### 8) Archive Lifecycle Requirements

#### R8.1 Archive must be deleted after extraction by default

- **Given:** An archive resource is extracted successfully.
- **When:** `build` completes.
- **Then:** The downloaded archive file is absent.

#### R8.2 KEEP_ARCHIVE must preserve archive after extraction

- **Given:** An archive resource is built with `KEEP_ARCHIVE` set.
- **When:** `build` completes.
- **Then:** The downloaded archive file is still present.

---

### 9) SHA-8 Cache Versioning Requirements

#### R9.1 Different hash must produce a separate cache subdir

- **Given:** A resource is built, then its hash changes.
- **When:** `build` runs with the new hash.
- **Then:** A new SHA-8 subdir is created; the old subdir is preserved.

#### R9.2 SHA-8 subdir must match the first 8 hex chars of the normalized hash

- **Given:** A resource with a known hash is ensured.
- **When:** The cache is inspected.
- **Then:** Content lives under a subdir equal to the first 8 lowercase hex chars of the hash.

---

### 10) Concurrency Safety Requirements

Note: advisory locks (`file(LOCK)`) are released by the OS on process exit/crash, so
stale locks are not possible. `file(LOCK)` is local-filesystem only; NFS advisory
locks are unreliable.

#### R10.1 Concurrent processes ensuring the same resource must both succeed without corruption

- **Given:** Two cmake processes share the same cache and target the same resource+hash.
- **When:** Both run `isaac_resource_ensure` simultaneously.
- **Then:** Both complete successfully with correct, uncorrupted content.

#### R10.2 Concurrent processes ensuring different versions must not block each other

- **Given:** Two cmake processes share the same cache but target different hashes of the same resource.
- **When:** Both run `isaac_resource_ensure` simultaneously.
- **Then:** Both complete successfully; neither waits for the other.

---

### 11) Variable Resolution Requirements

#### R11.1 ISAAC_RESOURCE_CACHE must resolve from ENV when -D is not set

- **Given:** ISAAC_RESOURCE_CACHE is set only via environment variable.
- **When:** cmake configures.
- **Then:** Content lands in the ENV-specified path.

#### R11.2 ISAAC_RESOURCE_CACHE must default when neither -D nor ENV is set

- **Given:** ISAAC_RESOURCE_CACHE is not set anywhere.
- **When:** cmake configures.
- **Then:** Content lands in `${CMAKE_SOURCE_DIR}/.cache/resources/`.

#### R11.3 -D must take priority over ENV for ISAAC_RESOURCE_CACHE

- **Given:** ISAAC_RESOURCE_CACHE is set via both -D and ENV to different paths.
- **When:** cmake configures.
- **Then:** Content lands in the -D path, not the ENV path.

---

### 12) Extraction Recipe Isolation

#### R12.1 STRIP_PREFIX is a consumer-side view over one shared extraction

- **Given:** Two callers ensure the same archive+hash with different
  `STRIP_PREFIX` values.
- **When:** Both configure against the same `ISAAC_RESOURCE_CACHE`.
- **Then:** Each caller's CONTENT_DIR resolves to the correct sub-path of a
  single canonical `content/` tree (`content/<STRIP_PREFIX>/`); no
  duplicate extraction is triggered.

#### R12.2 Different ARCHIVE_PATTERNS coexist additively

- **Given:** Two callers ensure the same archive+hash with different
  canonicalized pattern sets.
- **When:** Both configure against the same cache.
- **Then:** Both extractions land in the single `content/` tree; each
  caller's CONTENT_DIR view surfaces its requested files (and may also see
  files contributed by earlier recipe extractions). `content.recipes`
  records one token per extracted recipe.

#### R12.3 Equivalent ARCHIVE_PATTERNS are a cache hit

- **Given:** A recipe with a given canonicalized hash already appears in
  `content.recipes`.
- **When:** A later caller ensures the same archive with patterns that
  canonicalize to the same hash (for example a reordered list).
- **Then:** No new extraction runs and `content.recipes` is not rewritten.

#### R12.4 `*` recipe covers every subsequent patterns request

- **Given:** `content.recipes` contains `*` (a full extraction has been
  recorded).
- **When:** A later caller ensures the same archive with any ARCHIVE_PATTERNS.
- **Then:** Extraction is skipped (the full tree satisfies any subset request)
  and no new recipe token is appended.

---

### 13) Content View Description Requirements

#### R13.1 Content identity must represent the resolved resource view

- **Given:** Resource declarations with the same downloaded bytes.
- **When:** Their pattern order is equivalent, their view recipe changes, or their
  non-archive basename changes.
- **Then:** Equivalent views have equal content IDs and different views have different IDs.

#### R13.2 Build-time targets must publish their content description

- **Given:** A build-time file or archive resource target.
- **When:** CMake configures the target.
- **Then:** Target properties expose its content directory, full content ID,
  bundle destination, and archive kind without downloading it.

---

### 14) Resource Bundle Parser Requirements

#### R14.1 Repeated resource clauses must create unlinked member targets

- **Given:** A bundle declares multiple `RESOURCE <name> ...` clauses.
- **When:** CMake configures and builds the bundle target.
- **Then:** Each member is an ordinary cache-backed resource target and a dependency of the bundle.
- **And:** Members do not create their ordinary default `data/<member>` links.

#### R14.2 Ambiguous bundle syntax must fail at configure time

- **Given:** A bundle declaration is missing required clauses, repeats targets or destinations,
  contains member-level linking/configuration-time options, or uses the reserved `RESOURCE` token
  as data.
- **When:** CMake configures the bundle.
- **Then:** Configuration fails with a diagnostic naming the bundle and offending member or token.

---

### 15) Resource Bundle Assembly Requirements

#### R15.1 Bundle content must compose member views with symlinks

- **Given:** A bundle contains file and archive members.
- **When:** The bundle target builds.
- **Then:** File members are linked at the bundle root and archive views are linked under their
  member target names.
- **And:** The public LINK_DIR points to the bundle's SHA-versioned content directory.

#### R15.2 Bundle identity must be canonical and preserve independent member caches

- **Given:** Equivalent bundles declared in different orders, then one bundle with one changed
  member hash.
- **When:** The bundle targets build against a shared cache.
- **Then:** Equivalent bundles share an identity and generated root artifacts.
- **And:** The changed member selects a new bundle identity without modifying unchanged member
  verification state.

#### R15.3 Assembly must preserve generated artifacts and reject real collisions

- **Given:** Generated files exist beside root files or within an archive mount.
- **When:** A missing member link or public link is rebuilt.
- **Then:** Generated files remain in their respective bundle or archive-member cache.
- **And:** A real object at a declared member destination causes an error and is not replaced.

---

## Scenarios

### Scenario IDs for Tests

- **S1:** Clean checkout, first build.
- **S2:** Rebuild without changes (build_only).
- **S3:** Reconfigure without semantic changes, then build.
- **S4:** Delete non-archive output only (build_only).
- **S5:** Delete archive extracted output only, keep extraction stamp (build_only).
- **S6:** Delete archive extracted output only, keep extraction stamp (reconfigure_then_build).
- **S7:** Archive absent (deleted by default), content valid - second build is a no-op.
- **S8:** Change expected identity/hash.
- **S9:** Delete stamps only (outputs intact, build_only).
- **S10:** Delete all resource state (symlinks, cache content, stamps) - build-only recovery.
- **S11:** Delete all resource state (symlinks, cache content, stamps) - reconfigure+build recovery.
- **S12:** Delete outputs/stamps between configure and build (no reconfigure).
- **S13:** Two build directories share one source/data tree (reuse).
- **S14:** Shared metadata with stale output (build_only).
- **S15:** Shared metadata with stale output (reconfigure_then_build).
- **S16:** Metadata timestamp-only change (build_only).
- **S17:** Reconfigure unchanged archive identity, then build.
- **S18:** Output missing + extraction stamp present (archive absent by default) (build_only).
- **S19:** Archive absent (default), content valid - rebuild skips all heavy work (build_only).
- **S20:** Filtered extraction only extracts matching patterns.
- **S21:** Filtered extraction is idempotent on rebuild.
- **S22:** Filtered extraction recovers after output deletion.
- **S23:** Symlink recreated after deletion (content stays in cache).
- **S24:** No LINK_DIR set → content accessible in cache only, no symlink created.
- **S25:** Archive deleted by default after extraction.
- **S26:** KEEP_ARCHIVE preserves archive after extraction.
- **S27:** Hash change creates new SHA-8 subdir, old subdir preserved, symlink updated.
- **S28:** Two concurrent processes ensuring the same resource both complete without corruption.
- **S29:** Two concurrent processes ensuring different versions of the same resource
  complete without blocking each other.
- **S30:** ISAAC_RESOURCE_CACHE resolved from ENV when -D is not set.
- **S31:** ISAAC_RESOURCE_CACHE defaults to source-tree .cache/resources when neither set.
- **S32:** -D takes priority over ENV for ISAAC_RESOURCE_CACHE.
- **S33:** Same hash, different STRIP_PREFIX → independent sub-path views
  over a shared `content/` tree (single extraction on disk).
- **S34:** Same hash, different ARCHIVE_PATTERNS → both recipes extract
  additively into the one `content/` tree; `content.recipes` lists both
  recipe tokens.
- **S35:** Same hash, equivalent ARCHIVE_PATTERNS (reordered) → second call
  is a cache hit via canonicalized recipe; no re-extraction, recipes stamp
  unchanged.
- **S36:** `*` already in `content.recipes` → subsequent patterns caller
  skips extraction and does not add a new recipe token.
- **S37:** A generated sibling written through a non-archive resource directory
  is stored in the SHA-versioned resource cache.
- **S38:** Explicit LINK_DIR points to the SHA-versioned content directory.
- **S39:** A non-empty LINK_DIR ending in a CMake false-constant spelling is used literally.
- **S40:** Resource content IDs are canonical for equivalent views and differ for
  changed view recipes or file basenames.
- **S41:** Build-time resource targets publish their resolved content description.
- **S42:** Repeated RESOURCE clauses create unlinked member targets and bundle dependencies.
- **S43:** Bundle declarations reject ambiguous or unsupported syntax at configure time.
- **S44:** Bundle assembly exposes file and archive views and preserves generated artifacts.
- **S45:** Bundle identity ignores declaration order and independently reuses unchanged members.
- **S46:** Bundle assembly refuses to replace a real declared destination.
- **S47:** Verification stamps preserve the stripped spelling of EXPECTED_HASH.

---

### Scenario to Test Mapping (1:1)

- **S1** -> `test_s1_clean_checkout_first_build`
- **S2** -> `test_s2_rebuild_without_changes_build_only`
- **S3** -> `test_s3_reconfigure_unchanged_identity_then_build`
- **S4** -> `test_s4_delete_non_archive_output_only_build_only`
- **S5** -> `test_s5_delete_archive_output_only_keep_extraction_stamp_build_only`
- **S6** -> `test_s6_delete_archive_output_only_keep_extraction_stamp_reconfigure_then_build`
- **S7** -> `test_s7_delete_archive_artifact_only_keep_valid_extracted_output`
- **S8** -> `test_s8_change_expected_identity_hash`
- **S9** -> `test_s9_delete_stamps_only_outputs_intact_build_only`
- **S10** -> `test_s10_delete_outputs_and_stamps_build_only`
- **S11** -> `test_s11_delete_outputs_and_stamps_reconfigure_then_build`
- **S12** -> `test_s12_delete_between_configure_and_build_build_only`
- **S13** -> `test_s13_two_build_dirs_shared_data_tree_reuse`
- **S14** -> `test_s14_shared_metadata_stale_output_build_only`
- **S15** -> `test_s15_shared_metadata_stale_output_reconfigure_then_build`
- **S16** -> `test_s16_metadata_timestamp_only_change_build_only`
- **S17** -> `test_s17_reconfigure_unchanged_archive_identity_then_build`
- **S18** -> `test_s18_archive_missing_output_missing_stamp_present_build_only`
- **S19** -> `test_s19_archive_missing_extracted_valid_build_only_skip_intermediates`
- **S20** -> `test_s20_filtered_extraction_only_extracts_matching_patterns`
- **S21** -> `test_s21_filtered_extraction_idempotent_rebuild`
- **S22** -> `test_s22_filtered_extraction_recovers_after_delete`
- **S23** -> `test_s23_symlink_recreated_after_deletion`
- **S24** -> `test_s24_no_link_dir_content_in_cache_no_symlink`
- **S25** -> `test_s25_archive_deleted_by_default_after_extraction`
- **S26** -> `test_s26_keep_archive_preserves_archive_after_extraction`
- **S27** -> `test_s27_hash_change_creates_new_sha8_dir_old_preserved`
- **S28** -> `test_s28_concurrent_same_resource_both_succeed`
- **S29** -> `test_s29_concurrent_different_versions_no_blocking`
- **S30** -> `test_s30_cache_dir_resolved_from_env`
- **S31** -> `test_s31_cache_dir_defaults_when_unset`
- **S32** -> `test_s32_cache_dir_d_overrides_env`
- **S33** -> `test_s33_strip_prefix_is_view_over_shared_extraction`
- **S34** -> `test_s34_patterns_coexist_in_shared_extraction`
- **S35** -> `test_s35_patterns_canonicalization_is_cache_hit`
- **S36** -> `test_s36_star_recipe_covers_subsequent_patterns`
- **S37** -> `test_s37_file_resource_keeps_generated_siblings_in_cache`
- **S38** -> `test_s38_explicit_link_dir_points_to_content`
- **S39** -> `test_s39_link_dir_cmake_false_constant_is_not_treated_as_unset`
- **S40** -> `test_s40_resource_content_view_identity_is_canonical`
- **S41** -> `test_s41_resource_targets_publish_content_view_properties`
- **S42** -> `test_s42_bundle_parser_creates_member_targets`
- **S43** -> `test_s43_bundle_parser_rejects_invalid_clauses`
- **S44** -> `test_s44_bundle_assembly_layout_and_generated_artifacts`
- **S45** -> `test_s45_bundle_identity_and_independent_member_reuse`
- **S46** -> `test_s46_bundle_assembly_rejects_real_destination_collision`
- **S47** -> `test_s47_expected_hash_stamp_preserves_stripped_input`

---

### Traceability (Requirement -> Scenarios)

- **R1.1:** S4, S11
- **R1.2a:** S5, S10, S12
- **R1.2b:** S6, S11
- **R1.3:** S8
- **R1.4a:** S10, S12, S18
- **R1.4b:** S11
- **R1.5:** S47
- **R2.1:** S7, S13, S19
- **R2.2:** S2, S3, S17
- **R2.3:** S3, S17
- **R2.4a:** S16
- **R2.4b:** S3, S17
- **R3.1:** S12, S18
- **R3.2a:** S5, S9, S12, S14
- **R3.2b:** S6, S11, S15
- **R3.3:** S10, S12, S18
- **R4.1:** S7, S19
- **R4.2a:** S5, S12, S18
- **R4.2b:** S6
- **R4.3:** S4, S12
- **R4.4:** S9
- **R4.5a:** S10
- **R4.5b:** S11
- **R5.1:** S20
- **R5.2:** S21
- **R5.3:** S22
- **R6.1:** S13
- **R6.2a:** S14
- **R6.2b:** S15
- **R7.1:** S23, S37, S38, S39
- **R7.2:** S23
- **R7.3:** S24
- **R7.4:** S37
- **R8.1:** S25
- **R8.2:** S26
- **R9.1:** S27
- **R9.2:** S27
- **R10.1:** S28
- **R10.2:** S29
- **R11.1:** S30
- **R11.2:** S31
- **R11.3:** S32
- **R12.1:** S33
- **R12.2:** S34
- **R12.3:** S35
- **R12.4:** S36
- **R13.1:** S40
- **R13.2:** S41
- **R14.1:** S42
- **R14.2:** S43
- **R15.1:** S44
- **R15.2:** S45
- **R15.3:** S44, S46

---
