# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import unittest


class ResourceRequirementsTest(unittest.TestCase):
    """Scenario-driven tests mapped to resource requirements."""

    maxDiff = None
    REPO_ROOT = Path(__file__).resolve().parents[3]
    SHA_PREFIX_LEN = 8  # must match isaac_resource_ensure SHA subdir length

    # -------------------------------------------------------------------------
    # Setup / teardown
    # -------------------------------------------------------------------------

    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix='resource-req-'))
        self.sources_dir = self.workspace / 'sources'
        self.project_dir = self.workspace / 'fixture'
        self.data_dir = self.project_dir / 'data'
        self.build_dir = self.workspace / 'build'
        self.build2_dir = self.workspace / 'build2'

        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.resource_cache_dir = self.workspace / 'resource_cache'
        self.resource_cache_dir.mkdir(parents=True, exist_ok=True)

        # Single plain file resource.
        self.source_file = self.sources_dir / 'sample_data.txt'
        self.source_file.write_text('payload-v1\n', encoding='utf-8')
        self.renamed_source_file = self.sources_dir / 'renamed_data.txt'
        shutil.copyfile(self.source_file, self.renamed_source_file)
        self.identity_patch = self.sources_dir / 'identity.patch'
        self.identity_patch.write_text('identity-patch-v1\n', encoding='utf-8')

        # Archive resource (single inner file after strip-prefix).
        archive_root = self.sources_dir / 'archive-tree'
        (archive_root / 'payload-root').mkdir(parents=True, exist_ok=True)
        (archive_root / 'payload-root' / 'inner.txt').write_text('archive-v1\n', encoding='utf-8')
        self.archive_file = self.sources_dir / 'sample_archive.tar.gz'
        with tarfile.open(self.archive_file, 'w:gz') as tar:
            tar.add(archive_root / 'payload-root', arcname='payload-root')

        # Filtered archive: multiple dirs, only some should be extracted.
        filtered_root = self.sources_dir / 'filtered-tree'
        (filtered_root / 'pkg' / 'include' / 'api').mkdir(parents=True, exist_ok=True)
        (filtered_root / 'pkg' / 'include' / 'api' / 'header.h').write_text(
            '#pragma once\n', encoding='utf-8'
        )
        (filtered_root / 'pkg' / 'lib' / 'x86_64').mkdir(parents=True, exist_ok=True)
        (filtered_root / 'pkg' / 'lib' / 'x86_64' / 'libfoo.so').write_text(
            'ELF-fake\n', encoding='utf-8'
        )
        (filtered_root / 'pkg' / 'lib' / 'aarch64').mkdir(parents=True, exist_ok=True)
        (filtered_root / 'pkg' / 'lib' / 'aarch64' / 'libfoo.so').write_text(
            'ELF-fake-arm\n', encoding='utf-8'
        )
        (filtered_root / 'pkg' / 'docs').mkdir(parents=True, exist_ok=True)
        (filtered_root / 'pkg' / 'docs' / 'big_file.html').write_text(
            '<html>docs</html>\n', encoding='utf-8'
        )
        self.filtered_archive_file = self.sources_dir / 'filtered_archive.tar.gz'
        with tarfile.open(self.filtered_archive_file, 'w:gz') as tar:
            tar.add(filtered_root / 'pkg', arcname='pkg')
        self.archive_as_file = self.sources_dir / 'filtered_archive.bin'
        shutil.copyfile(self.filtered_archive_file, self.archive_as_file)

        self.file_hash = self._sha256(self.source_file)
        self.archive_hash = self._sha256(self.archive_file)
        self.filtered_archive_hash = self._sha256(self.filtered_archive_file)

        self._write_fixture_project()

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)

    def _write_fixture_project(self) -> None:
        fixture_src = Path(__file__).parent / 'fixture_CMakeLists.txt'
        shutil.copy(fixture_src, self.project_dir / 'CMakeLists.txt')

    def _configure_bundle_project(
        self, body: str, *, build_dir: Path | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        build_dir = build_dir or self.build_dir
        (self.project_dir / 'CMakeLists.txt').write_text(
            '\n'.join(
                [
                    'cmake_minimum_required(VERSION 3.20)',
                    'project(resource_bundle_fixture NONE)',
                    'include(isaac_add_resource)',
                    body,
                    '',
                ]
            ),
            encoding='utf-8',
        )
        cmake_module_path = self.REPO_ROOT / 'cmake'
        return self._run(
            [
                'cmake',
                '-S',
                str(self.project_dir),
                '-B',
                str(build_dir),
                f'-DCMAKE_MODULE_PATH={cmake_module_path}',
                f'-DISAAC_RESOURCE_CACHE={self.resource_cache_dir}',
            ],
            cwd=self.workspace,
            check=check,
        )

    # -------------------------------------------------------------------------
    # Infrastructure helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open('rb') as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _configure(
        self,
        build_dir: Path,
        file_hash: str | None = None,
        archive_hash: str | None = None,
        archive_url: Path | None = None,
        *,
        with_filtered: bool = False,
        filtered_patterns: list[str] | None = None,
        keep_archive: bool = False,
        cache_via_env: bool = False,
        check: bool = True,
        explicit_link_mode: str = 'LINK_DIR',
        extra_args: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        file_hash = file_hash or self.file_hash
        archive_hash = archive_hash or self.archive_hash
        archive_url = archive_url or self.archive_file
        cmake_module_path = self.REPO_ROOT / 'cmake'
        cmd = [
            'cmake',
            '-S',
            str(self.project_dir),
            '-B',
            str(build_dir),
            f'-DCMAKE_MODULE_PATH={cmake_module_path}',
        ]
        if not cache_via_env:
            cmd.append(f'-DISAAC_RESOURCE_CACHE={self.resource_cache_dir}')
        cmd += [
            f'-DTEST_FILE_URL=file://{self.source_file}',
            f'-DTEST_FILE_HASH=SHA256={file_hash}',
            f'-DTEST_IDENTITY_PATCH={self.identity_patch}',
            f'-DTEST_RENAMED_FILE_URL=file://{self.renamed_source_file}',
            f'-DTEST_ARCHIVE_URL=file://{archive_url}',
            f'-DTEST_ARCHIVE_HASH=SHA256={archive_hash}',
            f'-DTEST_ARCHIVE_AS_FILE_URL=file://{self.archive_as_file}',
            f'-DTEST_EXPLICIT_LINK_MODE={explicit_link_mode}',
        ]
        if with_filtered:
            patterns = filtered_patterns or ['pkg/include/**', 'pkg/lib/x86_64/libfoo*']
            cmd += [
                f'-DTEST_FILTERED_ARCHIVE_URL=file://{self.filtered_archive_file}',
                f'-DTEST_FILTERED_ARCHIVE_HASH=SHA256={self.filtered_archive_hash}',
                '-DTEST_FILTERED_ARCHIVE_STRIP_PREFIX=pkg',
                '-DTEST_FILTERED_ARCHIVE_PATTERNS=' + ';'.join(patterns),
            ]
        if keep_archive:
            cmd += ['-DTEST_ARCHIVE_KEEP_ARCHIVE=TRUE']
        if extra_args:
            cmd += extra_args
        env = None
        if cache_via_env:
            env = {**os.environ, 'ISAAC_RESOURCE_CACHE': str(self.resource_cache_dir)}
        return self._run(cmd, cwd=self.workspace, check=check, env=env)

    def _build(
        self, build_dir: Path, target: str = 'all_resources', check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return self._run(
            ['cmake', '--build', str(build_dir), '--target', target],
            cwd=self.workspace,
            check=check,
        )

    def _run(
        self,
        cmd: list[str],
        cwd: Path,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False, env=env)
        if check and result.returncode != 0:
            cmd_str = shlex.join(cmd)
            raise AssertionError(
                f'Command failed ({result.returncode}): {cmd_str}\n'
                f'--- STDOUT ---\n{result.stdout}\n'
                f'--- STDERR ---\n{result.stderr}\n'
            )
        return result

    @staticmethod
    def _communicate(proc: subprocess.Popen[str], timeout: float) -> tuple[str, str]:
        """Communicate with a child, killing and draining it before propagating a timeout."""

        try:
            return proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise

    def _rmtree(self, path: Path) -> None:
        """Remove path; if it's a symlink, removes both the symlink and its target."""
        if path.is_symlink():
            target = (path.parent / os.readlink(path)).resolve()
            path.unlink()
            if target.is_dir():
                shutil.rmtree(target)
        elif path.is_dir():
            shutil.rmtree(path)

    def _clear_all(self) -> None:
        """Delete all resource state: symlinks, cache content, and stamps."""
        shutil.rmtree(self.data_dir, ignore_errors=True)
        shutil.rmtree(self.resource_cache_dir)
        self.resource_cache_dir.mkdir()

    # -------------------------------------------------------------------------
    # Per-resource path helpers
    # -------------------------------------------------------------------------

    # res_file (plain file, accessed via LINK_DIR symlink)

    def _file_out(self) -> Path:
        """Content file accessible via the res_file symlink."""
        return self.data_dir / 'res_file' / self.source_file.name

    def _file_stamp(self, file_hash: str | None = None) -> Path:
        """Hash verification stamp for res_file in the cache."""
        h = file_hash or self.file_hash
        return (
            self.resource_cache_dir
            / 'res_file'
            / h[: self.SHA_PREFIX_LEN]
            / f'{self.source_file.name}.hash.verified'
        )

    # res_archive (archive, accessed via LINK_DIR symlink)

    def _archive_out(self) -> Path:
        """Extracted file accessible via the res_archive symlink."""
        return self.data_dir / 'res_archive' / 'inner.txt'

    def _recipes_stamp(self, archive_hash: str | None = None) -> Path:
        """Recipes stamp for res_archive (lists extracted recipe tokens)."""
        h = archive_hash or self.archive_hash
        return (
            self.resource_cache_dir / 'res_archive' / h[: self.SHA_PREFIX_LEN] / 'content.recipes'
        )

    # res_filtered_archive (filtered archive, accessed via LINK_DIR symlink)

    def _filtered_out(self) -> tuple[Path, Path, Path, Path]:
        """Return (header, x86_lib, arm_lib, docs) via the res_filtered_archive symlink."""
        base = self.data_dir / 'res_filtered_archive'
        return (
            base / 'include' / 'api' / 'header.h',
            base / 'lib' / 'x86_64' / 'libfoo.so',
            base / 'lib' / 'aarch64' / 'libfoo.so',
            base / 'docs' / 'big_file.html',
        )

    # -------------------------------------------------------------------------
    # S1-S4: Basic correctness
    # -------------------------------------------------------------------------

    def test_subproject_default_paths(self) -> None:
        cmake_module_path = self.REPO_ROOT / 'cmake'
        (self.workspace / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.24)\n'
            'project(parent NONE)\n'
            'add_subdirectory(fixture)\n',
            encoding='utf-8',
        )
        (self.project_dir / 'CMakeLists.txt').write_text(
            'project(resource_fixture NONE)\n'
            'include(isaac_add_resource)\n'
            f'isaac_add_resource(sample URL "{self.source_file.as_uri()}" '
            f'EXPECTED_HASH SHA256={self.file_hash})\n'
            'isaac_add_resource_bundle(bundle\n'
            f'  RESOURCE member URL "{self.source_file.as_uri()}" '
            f'EXPECTED_HASH SHA256={self.file_hash})\n',
            encoding='utf-8',
        )
        env = dict(os.environ)
        env.pop('ISAAC_RESOURCE_CACHE', None)
        self._run(
            [
                'cmake',
                '-S',
                str(self.workspace),
                '-B',
                str(self.build_dir),
                f'-DCMAKE_MODULE_PATH={cmake_module_path}',
            ],
            cwd=self.workspace,
            env=env,
        )
        self._build(self.build_dir, target='sample')
        self._build(self.build_dir, target='bundle')
        self.assertEqual(
            (self.data_dir / 'sample' / self.source_file.name).read_text(), 'payload-v1\n'
        )
        self.assertEqual(
            (self.data_dir / 'bundle' / self.source_file.name).read_text(), 'payload-v1\n'
        )
        self.assertTrue((self.project_dir / '.cache/resources').is_dir())
        self.assertFalse((self.workspace / 'data').exists())
        self.assertFalse((self.workspace / '.cache').exists())

    # S1: Clean checkout, first build.
    def test_s1_clean_checkout_first_build(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        self.assertTrue(self._file_out().exists())
        self.assertTrue(self._archive_out().exists())
        self.assertTrue(self._recipes_stamp().exists())

    # S2: Rebuild without changes - no expensive re-verification.
    def test_s2_rebuild_without_changes_build_only(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        self.assertTrue(self._file_out().exists())
        stamp = self._file_stamp()
        self.assertTrue(stamp.exists())

        mtime_before = stamp.stat().st_mtime_ns
        self._build(self.build_dir, target='res_file')
        self.assertEqual(
            mtime_before,
            stamp.stat().st_mtime_ns,
            'Rebuild without changes should not rerun verification work.',
        )

    # S3: Reconfigure without semantic changes - no expensive work on rebuild.
    def test_s3_reconfigure_unchanged_identity_then_build(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_file')
        stamp = self._file_stamp()
        self.assertTrue(stamp.exists())
        mtime_before = stamp.stat().st_mtime_ns

        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_file')
        self.assertEqual(
            mtime_before,
            stamp.stat().st_mtime_ns,
            'Reconfigure without semantic changes should avoid expensive verify work.',
        )

    # S4: Delete non-archive output only - repaired on next build.
    def test_s4_delete_non_archive_output_only_build_only(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_file')
        file_out = self._file_out()
        stamp = self._file_stamp()
        self.assertTrue(file_out.exists())

        file_out.unlink()
        self._build(self.build_dir, target='res_file')
        self.assertTrue(file_out.exists())
        self.assertTrue(stamp.exists())

    # -------------------------------------------------------------------------
    # S5-S6: Archive output deletion (stamp survives)
    # -------------------------------------------------------------------------

    # S5: Delete archive extracted output only - repaired on next build (build_only).
    def test_s5_delete_archive_output_only_keep_extraction_stamp_build_only(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        archive_out = self._archive_out()
        archive_stamp = self._recipes_stamp()
        self.assertTrue(archive_out.exists())
        self.assertTrue(archive_stamp.exists())

        self._rmtree(archive_out.parent)
        self._build(self.build_dir, target='res_archive')
        self.assertTrue(
            archive_out.exists(),
            'Archive output should be recreated when missing, even if stamp exists.',
        )

    # S6: Delete archive extracted output only - repaired after reconfigure+build.
    def test_s6_delete_archive_output_only_keep_extraction_stamp_reconfigure_then_build(
        self,
    ) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        archive_out = self._archive_out()
        self.assertTrue(archive_out.exists())

        self._rmtree(archive_out.parent)
        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_archive')
        self.assertTrue(
            archive_out.exists(),
            'Archive output should be recreated after reconfigure+build.',
        )

    # -------------------------------------------------------------------------
    # S7: Archive absent (default) - no-op across build dirs (cross-build reuse)
    # -------------------------------------------------------------------------

    # S7: Archive absent by default; second build dir reuses valid cache content.
    def test_s7_delete_archive_artifact_only_keep_valid_extracted_output(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        archive_out = self._archive_out()
        self.assertTrue(archive_out.exists())
        # Archive is deleted by default - this is the normal post-extraction state.
        sha_dir = self.resource_cache_dir / 'res_archive' / self.archive_hash[: self.SHA_PREFIX_LEN]
        self.assertFalse(
            (sha_dir / self.archive_file.name).exists(),
            'Archive should not exist after default extraction.',
        )

        self._configure(self.build2_dir)
        self._build(self.build2_dir, target='res_archive')
        self.assertTrue(archive_out.exists())

    # -------------------------------------------------------------------------
    # S8: Identity change forces refresh
    # -------------------------------------------------------------------------

    def test_s8_change_expected_identity_hash(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        file_out = self._file_out()
        self.assertEqual(file_out.read_text(encoding='utf-8'), 'payload-v1\n')

        self.source_file.write_text('payload-v2\n', encoding='utf-8')
        new_hash = self._sha256(self.source_file)
        self.file_hash = new_hash
        self._configure(self.build_dir, file_hash=new_hash)
        self._build(self.build_dir, target='res_file')
        self.assertEqual(file_out.read_text(encoding='utf-8'), 'payload-v2\n')

    # -------------------------------------------------------------------------
    # S9: Missing stamps with valid output - self-heal
    # -------------------------------------------------------------------------

    def test_s9_delete_stamps_only_outputs_intact_build_only(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        file_out = self._file_out()
        archive_out = self._archive_out()
        file_stamp = self._file_stamp()
        archive_stamp = self._recipes_stamp()
        self.assertTrue(file_out.exists())
        self.assertTrue(archive_out.exists())
        self.assertTrue(file_stamp.exists())
        self.assertTrue(archive_stamp.exists())

        file_stamp.unlink()
        archive_stamp.unlink()
        self._build(self.build_dir)
        self.assertTrue(file_out.exists())
        self.assertTrue(archive_out.exists())

    # -------------------------------------------------------------------------
    # S10-S11: Full cache deletion - complete recovery
    # -------------------------------------------------------------------------

    # S10: Delete all cache (outputs, content, stamps) - build-only recovery.
    def test_s10_delete_outputs_and_stamps_build_only(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        file_out = self._file_out()
        archive_out = self._archive_out()
        self.assertTrue(file_out.exists())
        self.assertTrue(archive_out.exists())

        self._clear_all()
        self._build(self.build_dir)
        self.assertTrue(file_out.exists())
        self.assertTrue(archive_out.exists())

    # S11: Delete all cache - reconfigure+build recovery.
    def test_s11_delete_outputs_and_stamps_reconfigure_then_build(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        file_out = self._file_out()
        archive_out = self._archive_out()
        self.assertTrue(file_out.exists())
        self.assertTrue(archive_out.exists())

        self._clear_all()
        self._configure(self.build_dir)
        self._build(self.build_dir)
        self.assertTrue(file_out.exists())
        self.assertTrue(archive_out.exists())
        self.assertTrue(self._file_stamp().exists())
        self.assertTrue(self._recipes_stamp().exists())

    # -------------------------------------------------------------------------
    # S12: Deletion between configure and build - build-only recovery
    # -------------------------------------------------------------------------

    def test_s12_delete_between_configure_and_build_build_only(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        archive_out = self._archive_out()
        archive_stamp = self._recipes_stamp()
        self.assertTrue(archive_stamp.exists())

        self._configure(self.build_dir)
        self._rmtree(archive_out.parent)
        archive_stamp.unlink()
        self._build(self.build_dir, target='res_archive')
        self.assertTrue(
            archive_out.exists(),
            'Build-only path must recover and restore missing required archive output.',
        )

    # -------------------------------------------------------------------------
    # S13-S15: Cross-build reuse
    # -------------------------------------------------------------------------

    # S13: Two build directories share one source/data tree - cache reused.
    def test_s13_two_build_dirs_shared_data_tree_reuse(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        archive_out = self._archive_out()
        self.assertTrue(archive_out.exists())

        self._configure(self.build2_dir)
        result = self._build(self.build2_dir, target='res_archive', check=False)
        self.assertEqual(
            result.returncode,
            0,
            (
                'Shared cache reuse should succeed.\n'
                f'STDOUT:\n{result.stdout}\n'
                f'STDERR:\n{result.stderr}'
            ),
        )

    # S14: Shared metadata, stale output - must not falsely succeed (build_only).
    def test_s14_shared_metadata_stale_output_build_only(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        archive_out = self._archive_out()
        self.assertTrue(archive_out.exists())

        self._configure(self.build2_dir)
        self._build(self.build2_dir, target='res_archive')
        self._rmtree(archive_out.parent)
        result = self._build(self.build2_dir, target='res_archive', check=False)
        self.assertTrue(
            result.returncode != 0 or archive_out.exists(),
            'Shared metadata must not cause false-valid success when output is missing.',
        )

    # S15: Shared metadata, stale output - must not falsely succeed (reconfigure_then_build).
    def test_s15_shared_metadata_stale_output_reconfigure_then_build(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir)
        archive_out = self._archive_out()
        self.assertTrue(archive_out.exists())

        self._configure(self.build2_dir)
        self._build(self.build2_dir, target='res_archive')
        self._rmtree(archive_out.parent)
        self._configure(self.build2_dir)
        self._build(self.build2_dir, target='res_archive')
        self.assertTrue(
            archive_out.exists(),
            'Shared metadata must not cause false-valid success when output is missing.',
        )

    # -------------------------------------------------------------------------
    # S16-S17: Efficiency / timestamp stability
    # -------------------------------------------------------------------------

    # S16: Timestamp-only metadata change - no expensive work triggered.
    def test_s16_metadata_timestamp_only_change_build_only(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_file')
        stamp = self._file_stamp()
        self.assertTrue(stamp.exists())

        self._run(['cmake', '-E', 'touch', str(stamp)], cwd=self.workspace)

        build_result = self._build(self.build_dir, target='res_file')
        self.assertNotIn('Verifying resource:', build_result.stdout)
        self.assertNotIn('Downloading resource:', build_result.stdout)

    # S17: Reconfigure unchanged archive identity - no expensive work on rebuild.
    def test_s17_reconfigure_unchanged_archive_identity_then_build(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_archive')
        self.assertTrue(self._archive_out().exists())

        self._configure(self.build_dir)
        build_result = self._build(self.build_dir, target='res_archive')
        self.assertNotIn('Downloading resource:', build_result.stdout)
        self.assertNotIn('Extracting resource:', build_result.stdout)

    # -------------------------------------------------------------------------
    # S18-S19: Missing state recovery with absent archive (default)
    # -------------------------------------------------------------------------

    # S18: Output missing + stale stamp, archive absent by default - re-downloads and recovers.
    def test_s18_archive_missing_output_missing_stamp_present_build_only(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_archive')
        archive_out = self._archive_out()
        archive_stamp = self._recipes_stamp()
        self.assertTrue(archive_out.exists())
        self.assertTrue(archive_stamp.exists())

        self._rmtree(archive_out.parent)
        self.assertFalse(archive_out.exists())
        self.assertTrue(archive_stamp.exists())

        self._build(self.build_dir, target='res_archive')
        self.assertTrue(
            archive_out.exists(),
            'Build-only path must rebuild prerequisites and restore archive output.',
        )

    # S19: Archive absent (default), content valid - rebuild skips all heavy work.
    def test_s19_archive_missing_extracted_valid_build_only_skip_intermediates(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_archive')
        archive_out = self._archive_out()
        self.assertTrue(archive_out.exists())

        build_result = self._build(self.build_dir, target='res_archive')
        self.assertTrue(archive_out.exists())
        self.assertNotIn('Downloading resource:', build_result.stdout)
        self.assertNotIn('Verifying resource:', build_result.stdout)
        self.assertNotIn('Extracting resource:', build_result.stdout)

    # -------------------------------------------------------------------------
    # S20-S22: Filtered extraction (ARCHIVE_PATTERNS)
    # -------------------------------------------------------------------------

    # S20: Filtered extraction extracts only matching patterns.
    def test_s20_filtered_extraction_only_extracts_matching_patterns(self) -> None:
        self._configure(self.build_dir, with_filtered=True)
        self._build(self.build_dir, target='res_filtered_archive')
        header, x86_lib, arm_lib, docs = self._filtered_out()
        self.assertTrue(header.exists(), 'Header should be extracted (matches include/**).')
        self.assertTrue(
            x86_lib.exists(), 'x86_64 lib should be extracted (matches lib/x86_64/libfoo*).'
        )
        self.assertFalse(arm_lib.exists(), 'aarch64 lib should NOT be extracted.')
        self.assertFalse(docs.exists(), 'docs should NOT be extracted.')

    # S21: Filtered extraction is idempotent on rebuild.
    def test_s21_filtered_extraction_idempotent_rebuild(self) -> None:
        self._configure(self.build_dir, with_filtered=True)
        self._build(self.build_dir, target='res_filtered_archive')
        header, x86_lib, _, _ = self._filtered_out()
        self.assertTrue(header.exists())
        self.assertTrue(x86_lib.exists())

        build_result = self._build(self.build_dir, target='res_filtered_archive')
        self.assertTrue(header.exists())
        self.assertTrue(x86_lib.exists())
        self.assertNotIn('Extracting', build_result.stdout)

    # S22: Filtered extraction recovers after output deletion.
    def test_s22_filtered_extraction_recovers_after_delete(self) -> None:
        self._configure(self.build_dir, with_filtered=True)
        self._build(self.build_dir, target='res_filtered_archive')
        header, x86_lib, arm_lib, docs = self._filtered_out()
        self.assertTrue(header.exists())

        self._rmtree(self.data_dir / 'res_filtered_archive')
        self.assertFalse(header.exists())

        self._build(self.build_dir, target='res_filtered_archive')
        self.assertTrue(header.exists(), 'Header should be restored.')
        self.assertTrue(x86_lib.exists(), 'x86_64 lib should be restored.')
        self.assertFalse(arm_lib.exists(), 'aarch64 lib should still NOT be extracted.')
        self.assertFalse(docs.exists(), 'docs should still NOT be extracted.')

    # -------------------------------------------------------------------------
    # S23-S24: Symlink (LINK_DIR) behavior
    # -------------------------------------------------------------------------

    # S23: Symlink recreated after deletion - content stays in cache, no re-extraction.
    def test_s23_symlink_recreated_after_deletion(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_archive')
        archive_out = self._archive_out()
        self.assertTrue(archive_out.exists())

        link = self.data_dir / 'res_archive'
        self.assertTrue(link.is_symlink())
        link.unlink()
        self.assertFalse(link.exists())

        self._build(self.build_dir, target='res_archive')
        self.assertTrue(link.is_symlink(), 'Symlink should be recreated.')
        self.assertTrue(archive_out.exists(), 'Content should still be accessible.')

    # S24: No LINK_DIR - content in cache only (sha8 subdir), no symlink, CONTENT_DIR correct.
    def test_s24_no_link_dir_content_in_cache_no_symlink(self) -> None:
        self._configure(self.build_dir)
        sha8 = self.file_hash[: self.SHA_PREFIX_LEN]
        expected_content_dir = self.resource_cache_dir / 'res_config_no_link' / sha8 / 'content'
        self.assertTrue(
            (expected_content_dir / self.source_file.name).exists(),
            'Content should be in cache (sha8 subdir) after configure.',
        )
        actual_content_dir = Path(
            (self.build_dir / 'res_config_no_link_content_dir.txt').read_text().strip()
        )
        self.assertEqual(
            actual_content_dir,
            expected_content_dir,
            'CONTENT_DIR must include sha8 subdir so callers can find resources.',
        )
        self.assertFalse(
            (self.data_dir / 'res_config_no_link').exists(),
            'No symlink should be created without LINK_DIR.',
        )

    # -------------------------------------------------------------------------
    # S25-S26: Archive lifecycle (KEEP_ARCHIVE)
    # -------------------------------------------------------------------------

    # S25: Archive deleted by default after extraction.
    def test_s25_archive_deleted_by_default_after_extraction(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_archive')
        sha8 = self.archive_hash[: self.SHA_PREFIX_LEN]
        archive_path = self.resource_cache_dir / 'res_archive' / sha8 / self.archive_file.name
        self.assertFalse(archive_path.exists(), 'Archive should be deleted by default.')

    # S26: KEEP_ARCHIVE preserves archive after extraction.
    def test_s26_keep_archive_preserves_archive_after_extraction(self) -> None:
        self._configure(self.build_dir, keep_archive=True)
        self._build(self.build_dir, target='res_archive')
        sha8 = self.archive_hash[: self.SHA_PREFIX_LEN]
        archive_path = self.resource_cache_dir / 'res_archive' / sha8 / self.archive_file.name
        self.assertTrue(archive_path.exists(), 'Archive should be kept when KEEP_ARCHIVE.')

    # -------------------------------------------------------------------------
    # S27: SHA-8 versioning
    # -------------------------------------------------------------------------

    # S27: Hash change creates new sha8 dir, old dir preserved, symlink updated.
    def test_s27_hash_change_creates_new_sha8_dir_old_preserved(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_archive')
        archive_out = self._archive_out()
        old_sha8 = self.archive_hash[: self.SHA_PREFIX_LEN]
        old_sha_dir = self.resource_cache_dir / 'res_archive' / old_sha8
        self.assertTrue(old_sha_dir.exists())
        self.assertTrue(archive_out.exists())

        archive_root = self.sources_dir / 'archive-tree-v2'
        (archive_root / 'payload-root').mkdir(parents=True, exist_ok=True)
        (archive_root / 'payload-root' / 'inner.txt').write_text('archive-v2\n', encoding='utf-8')
        with tarfile.open(self.archive_file, 'w:gz') as tar:
            tar.add(archive_root / 'payload-root', arcname='payload-root')
        new_hash = self._sha256(self.archive_file)
        new_sha8 = new_hash[: self.SHA_PREFIX_LEN]

        self._configure(self.build_dir, archive_hash=new_hash)
        self._build(self.build_dir, target='res_archive')

        new_sha_dir = self.resource_cache_dir / 'res_archive' / new_sha8
        self.assertTrue(new_sha_dir.exists(), 'New SHA dir should be created.')
        self.assertTrue(old_sha_dir.exists(), 'Old SHA dir should be preserved.')
        self.assertTrue((self.data_dir / 'res_archive').is_symlink())
        self.assertTrue(archive_out.exists(), 'Content accessible via updated symlink.')

    # -------------------------------------------------------------------------
    # S28-S29: Concurrency safety
    # -------------------------------------------------------------------------

    # S28: cmake blocks on lock acquisition when another process holds the lock.
    # Approach: hold the .lock file from Python, start cmake, verify it waits.
    # Before fix (no locking): cmake completes immediately, assertIsNone fails.
    # After fix: cmake blocks until Python releases the lock.
    def test_s28_concurrent_same_resource_both_succeed(self) -> None:
        self._configure(self.build_dir)

        sha8 = self.archive_hash[: self.SHA_PREFIX_LEN]
        sha8_dir = self.resource_cache_dir / 'res_archive' / sha8
        sha8_dir.mkdir(parents=True, exist_ok=True)
        lock_path = sha8_dir / '.lock'
        lock_path.touch()

        # Acquire exclusive lock from Python using fcntl.lockf (which maps to fcntl(F_SETLKW)),
        # matching CMake's file(LOCK) implementation. Note: fcntl.flock uses the flock(2)
        # syscall which is a different lock namespace on Linux and would NOT conflict.
        with lock_path.open('w') as lock_fd:
            fcntl.lockf(lock_fd.fileno(), fcntl.LOCK_EX)
            try:
                # A Popen context here would wait for the process before releasing the lock.
                proc = subprocess.Popen(  # pylint: disable=consider-using-with
                    ['cmake', '--build', str(self.build_dir), '--target', 'res_archive'],
                    cwd=self.workspace,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                time.sleep(0.5)
                blocked_on_lock = proc.poll() is None
            finally:
                fcntl.lockf(lock_fd.fileno(), fcntl.LOCK_UN)

        stdout, stderr = self._communicate(proc, timeout=30)
        self.assertTrue(
            blocked_on_lock,
            'cmake should be blocked waiting for the lock held by this process.',
        )
        self.assertEqual(
            proc.returncode,
            0,
            f'cmake should succeed after lock release.\n{stdout}\n{stderr}',
        )
        self.assertTrue(self._archive_out().exists())

    # S29: cmake building a different version is NOT blocked by a lock on another version's dir.
    # Approach: hold lock on v1's sha8 dir, run cmake for v2 (different sha8), verify it
    # does NOT wait (completes within a short window despite v1 lock being held).
    # Before fix (no locking): cmake always runs freely, test passes trivially.
    # After fix with correct per-sha8 granularity: cmake for v2 is not blocked by v1 lock.
    # Would FAIL if locking was implemented at CACHE_DIR level (too coarse).
    def test_s29_concurrent_different_versions_no_blocking(self) -> None:
        # Build a v2 archive with different content and hash.
        v2_root = self.sources_dir / 'archive-v2-tree'
        (v2_root / 'payload-root').mkdir(parents=True, exist_ok=True)
        (v2_root / 'payload-root' / 'inner.txt').write_text('archive-v2\n', encoding='utf-8')
        v2_archive = self.sources_dir / 'sample_archive_v2.tar.gz'
        with tarfile.open(v2_archive, 'w:gz') as tar:
            tar.add(v2_root / 'payload-root', arcname='payload-root')
        v2_hash = self._sha256(v2_archive)

        self._configure(self.build_dir, archive_hash=v2_hash, archive_url=v2_archive)

        # Hold the lock on v1's sha8 dir (not v2's).
        v1_sha8 = self.archive_hash[: self.SHA_PREFIX_LEN]
        v1_sha8_dir = self.resource_cache_dir / 'res_archive' / v1_sha8
        v1_sha8_dir.mkdir(parents=True, exist_ok=True)
        lock_path = v1_sha8_dir / '.lock'
        lock_path.touch()

        with lock_path.open('w') as lock_fd:
            fcntl.lockf(lock_fd.fileno(), fcntl.LOCK_EX)
            try:
                # A Popen context here would wait for the process before releasing the lock.
                proc = subprocess.Popen(  # pylint: disable=consider-using-with
                    ['cmake', '--build', str(self.build_dir), '--target', 'res_archive'],
                    cwd=self.workspace,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                time.sleep(0.5)
                completed_without_lock = proc.poll() is not None
            finally:
                fcntl.lockf(lock_fd.fileno(), fcntl.LOCK_UN)

        stdout, stderr = self._communicate(proc, timeout=30)
        self.assertTrue(
            completed_without_lock,
            "cmake building v2 should NOT be blocked by a lock on v1's sha8 dir.",
        )
        self.assertEqual(
            proc.returncode,
            0,
            f'cmake should succeed without waiting for the v1 lock.\n{stdout}\n{stderr}',
        )
        v2_sha8_dir = self.resource_cache_dir / 'res_archive' / v2_hash[: self.SHA_PREFIX_LEN]
        self.assertTrue(v2_sha8_dir.exists(), 'v2 sha8 dir should be created.')

    # -------------------------------------------------------------------------
    # S30: Environment variable resolution
    # -------------------------------------------------------------------------

    # S30: ISAAC_RESOURCE_CACHE resolved from ENV when -D is not set.
    def test_s30_cache_dir_resolved_from_env(self) -> None:
        self._configure(self.build_dir, cache_via_env=True)
        self._build(self.build_dir)
        sha8 = self.archive_hash[: self.SHA_PREFIX_LEN]
        content_dir = self.resource_cache_dir / 'res_archive' / sha8 / 'content'
        self.assertTrue(
            content_dir.exists(),
            'Content should land in cache dir specified via environment variable.',
        )
        self.assertTrue(self._archive_out().exists())

    # S31: ISAAC_RESOURCE_CACHE defaults to source-tree .cache/resources when neither -D nor ENV.
    def test_s31_cache_dir_defaults_when_unset(self) -> None:
        # Configure without -D and without ENV - should use project_dir/.cache/resources
        cmake_module_path = self.REPO_ROOT / 'cmake'
        cmd = [
            'cmake',
            '-S',
            str(self.project_dir),
            '-B',
            str(self.build_dir),
            f'-DCMAKE_MODULE_PATH={cmake_module_path}',
            f'-DTEST_FILE_URL=file://{self.source_file}',
            f'-DTEST_FILE_HASH=SHA256={self.file_hash}',
            f'-DTEST_ARCHIVE_URL=file://{self.archive_file}',
            f'-DTEST_ARCHIVE_HASH=SHA256={self.archive_hash}',
        ]
        # Ensure ISAAC_RESOURCE_CACHE is not in the environment.
        env = {k: v for k, v in os.environ.items() if k != 'ISAAC_RESOURCE_CACHE'}
        self._run(cmd, cwd=self.workspace, env=env)
        self._build(self.build_dir)

        sha8 = self.archive_hash[: self.SHA_PREFIX_LEN]
        default_cache = self.project_dir / '.cache' / 'resources'
        content_dir = default_cache / 'res_archive' / sha8 / 'content'
        self.assertTrue(
            content_dir.exists(),
            'Content should land in default cache dir (.cache/resources) when unset.',
        )

    # S32: -D takes priority over ENV for ISAAC_RESOURCE_CACHE.
    def test_s32_cache_dir_d_overrides_env(self) -> None:
        env_cache = self.workspace / 'env_cache'
        env_cache.mkdir()
        env = {**os.environ, 'ISAAC_RESOURCE_CACHE': str(env_cache)}

        cmake_module_path = self.REPO_ROOT / 'cmake'
        cmd = [
            'cmake',
            '-S',
            str(self.project_dir),
            '-B',
            str(self.build_dir),
            f'-DCMAKE_MODULE_PATH={cmake_module_path}',
            f'-DISAAC_RESOURCE_CACHE={self.resource_cache_dir}',
            f'-DTEST_FILE_URL=file://{self.source_file}',
            f'-DTEST_FILE_HASH=SHA256={self.file_hash}',
            f'-DTEST_ARCHIVE_URL=file://{self.archive_file}',
            f'-DTEST_ARCHIVE_HASH=SHA256={self.archive_hash}',
        ]
        self._run(cmd, cwd=self.workspace, env=env)
        self._build(self.build_dir)

        sha8 = self.archive_hash[: self.SHA_PREFIX_LEN]
        # Content in -D path, NOT in ENV path.
        d_content = self.resource_cache_dir / 'res_archive' / sha8 / 'content'
        env_content = env_cache / 'res_archive' / sha8 / 'content'
        self.assertTrue(d_content.exists(), '-D path should contain the content.')
        self.assertFalse(env_content.exists(), 'ENV path should NOT be used when -D is set.')

    # -------------------------------------------------------------------------
    # S33-S36: Extraction recipe isolation (STRIP_PREFIX view, PATTERNS keying)
    # -------------------------------------------------------------------------

    def _configure_variant(
        self,
        build_dir: Path,
        *,
        strip_prefix: str = '',
        patterns: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Configure res_variant pointing at filtered_archive_file with tunable recipe."""
        cmake_module_path = self.REPO_ROOT / 'cmake'
        cmd = [
            'cmake',
            '-S',
            str(self.project_dir),
            '-B',
            str(build_dir),
            f'-DCMAKE_MODULE_PATH={cmake_module_path}',
            f'-DISAAC_RESOURCE_CACHE={self.resource_cache_dir}',
            f'-DTEST_FILE_URL=file://{self.source_file}',
            f'-DTEST_FILE_HASH=SHA256={self.file_hash}',
            f'-DTEST_ARCHIVE_URL=file://{self.archive_file}',
            f'-DTEST_ARCHIVE_HASH=SHA256={self.archive_hash}',
            f'-DTEST_VARIANT_URL=file://{self.filtered_archive_file}',
            f'-DTEST_VARIANT_HASH=SHA256={self.filtered_archive_hash}',
            f'-DTEST_VARIANT_STRIP_PREFIX={strip_prefix}',
            '-DTEST_VARIANT_PATTERNS=' + ';'.join(patterns or []),
        ]
        return self._run(cmd, cwd=self.workspace)

    def _variant_content_dir(self, build_dir: Path) -> Path:
        return Path((build_dir / 'res_variant_content_dir.txt').read_text().strip())

    def _variant_sha_dir(self) -> Path:
        return (
            self.resource_cache_dir
            / 'res_variant'
            / self.filtered_archive_hash[: self.SHA_PREFIX_LEN]
        )

    def _variant_recipes(self) -> list[str]:
        """Read `<sha8>/content.recipes` and return stripped non-empty lines."""
        recipes_file = self._variant_sha_dir() / 'content.recipes'
        if not recipes_file.exists():
            return []
        return [r.strip() for r in recipes_file.read_text().splitlines() if r.strip()]

    # S33: Same hash, different STRIP_PREFIX - both CONTENT_DIRs resolve to distinct
    # sub-paths of the single canonical content/ tree. No duplicate on-disk
    # extraction: only one `<sha8>/content/` directory exists and recipes records
    # `*` once.
    def test_s33_strip_prefix_is_view_over_shared_extraction(self) -> None:
        self._configure_variant(self.build_dir, strip_prefix='pkg/include/api')
        self._configure_variant(self.build2_dir, strip_prefix='pkg/lib/x86_64')

        content_a = self._variant_content_dir(self.build_dir)
        content_b = self._variant_content_dir(self.build2_dir)

        self.assertTrue(str(content_a).endswith('pkg/include/api'))
        self.assertTrue(str(content_b).endswith('pkg/lib/x86_64'))
        self.assertTrue((content_a / 'header.h').exists())
        self.assertTrue((content_b / 'libfoo.so').exists())

        # Both views are sub-paths of the single canonical content/ tree.
        content_root = self._variant_sha_dir() / 'content'
        self.assertTrue(str(content_a).startswith(str(content_root) + '/'))
        self.assertTrue(str(content_b).startswith(str(content_root) + '/'))
        # Only `content/` exists under <sha8>; no per-recipe directory.
        sha_subdirs = sorted(p.name for p in self._variant_sha_dir().iterdir() if p.is_dir())
        self.assertEqual(sha_subdirs, ['content'])
        # Recipes has one `*` token (second caller was a cache hit via recipes).
        self.assertEqual(self._variant_recipes(), ['*'])

    # S34: Same hash, different ARCHIVE_PATTERNS coexist additively in one
    # content/ tree. Both callers' views land in the same content/ tree; the
    # recipes stamp records one token per extracted recipe.
    # Order: patterns caller first, full caller second - ensures the patterns
    # extraction happens to disk (rather than being skipped by a pre-existing `*`).
    def test_s34_patterns_coexist_in_shared_extraction(self) -> None:
        # B (patterns include only) extracts first.
        self._configure_variant(self.build2_dir, patterns=['pkg/include/**'])
        content_b = self._variant_content_dir(self.build2_dir)
        content_root = self._variant_sha_dir() / 'content'
        self.assertTrue((content_root / 'pkg' / 'include' / 'api' / 'header.h').exists())
        self.assertFalse(
            (content_root / 'pkg' / 'lib').exists(),
            'After B alone, only include/ should be in content/',
        )
        recipes_after_b = self._variant_recipes()
        self.assertEqual(
            len(recipes_after_b),
            1,
            f'Recipes should record the patterns hash only, got {recipes_after_b}',
        )
        self.assertNotIn('*', recipes_after_b)

        # A (no patterns - "*") extracts on top.
        self._configure_variant(self.build_dir)
        content_a = self._variant_content_dir(self.build_dir)
        # Full tree now present.
        self.assertTrue((content_root / 'pkg' / 'lib' / 'x86_64' / 'libfoo.so').exists())
        self.assertTrue((content_root / 'pkg' / 'docs' / 'big_file.html').exists())
        # Recipes has both the pattern hash and `*`.
        recipes_after_a = self._variant_recipes()
        self.assertIn('*', recipes_after_a)
        self.assertEqual(len(recipes_after_a), 2)

        # Both callers see the same canonical tree (no per-recipe subdir).
        self.assertEqual(content_a.resolve(), content_b.resolve())
        sha_subdirs = sorted(p.name for p in self._variant_sha_dir().iterdir() if p.is_dir())
        self.assertEqual(sha_subdirs, ['content'])

    # S35: Canonicalization - reordered pattern list is a cache hit; no re-extract,
    # recipes stamp mtime stable.
    def test_s35_patterns_canonicalization_is_cache_hit(self) -> None:
        self._configure_variant(self.build_dir, patterns=['pkg/include/**', 'pkg/lib/x86_64/**'])
        content_a = self._variant_content_dir(self.build_dir)
        recipes_file = self._variant_sha_dir() / 'content.recipes'
        self.assertTrue(recipes_file.exists())
        recipes_before = self._variant_recipes()
        mtime_before = recipes_file.stat().st_mtime_ns

        result = self._configure_variant(
            self.build2_dir, patterns=['pkg/lib/x86_64/**', 'pkg/include/**']
        )
        content_b = self._variant_content_dir(self.build2_dir)

        self.assertEqual(content_a.resolve(), content_b.resolve())
        self.assertEqual(
            self._variant_recipes(),
            recipes_before,
            'Canonical cache hit must leave recipes unchanged',
        )
        self.assertEqual(
            recipes_file.stat().st_mtime_ns,
            mtime_before,
            'Recipes file mtime must be stable on cache hit',
        )
        self.assertNotIn('Extracting', result.stdout)

    # S36: `*` recipe covers any subsequent patterns request. After a full
    # extraction, a patterns caller skips extraction and doesn't add a new token.
    def test_s36_star_recipe_covers_subsequent_patterns(self) -> None:
        self._configure_variant(self.build_dir)  # full first
        content_a = self._variant_content_dir(self.build_dir)
        self.assertEqual(self._variant_recipes(), ['*'])

        self._configure_variant(self.build2_dir, patterns=['pkg/include/**'])
        content_b = self._variant_content_dir(self.build2_dir)

        self.assertEqual(content_a.resolve(), content_b.resolve())
        self.assertEqual(
            self._variant_recipes(),
            ['*'],
            'Patterns caller must not append to recipes when `*` is present',
        )

    # S37: Generated siblings written through a file resource remain in the shared cache.
    def test_s37_file_resource_keeps_generated_siblings_in_cache(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_file')
        generated = self.data_dir / 'res_file' / 'generated.engine'
        generated.write_text('engine\n', encoding='utf-8')
        cached = (
            self.resource_cache_dir
            / 'res_file'
            / self.file_hash[: self.SHA_PREFIX_LEN]
            / 'content'
            / 'generated.engine'
        )

        self.assertTrue(generated.parent.is_symlink())
        self.assertEqual(generated.parent.resolve(), cached.parent)
        self.assertEqual(generated.resolve(), cached)

    # S38: LINK_DIR explicitly names the public resource directory.
    def test_s38_explicit_link_dir_points_to_content(self) -> None:
        self._configure(self.build_dir)
        self._build(self.build_dir, target='res_file_explicit')
        link = self.data_dir / 'explicit'
        cached = (
            self.resource_cache_dir
            / 'res_file_explicit'
            / self.file_hash[: self.SHA_PREFIX_LEN]
            / 'content'
        )

        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), cached)
        self.assertTrue((link / self.source_file.name).is_file())

    # S39: Non-empty LINK_DIR strings retain their literal CMake-false spelling.
    def test_s39_link_dir_cmake_false_constant_is_not_treated_as_unset(self) -> None:
        self._configure(self.build_dir, explicit_link_mode='FALSE_VALUE')
        self._build(self.build_dir, target='res_file_explicit')

        self.assertTrue((self.data_dir / 'explicit-NOTFOUND').is_symlink())

    # S40: Content IDs describe the resolved view, not declaration order or target name.
    def test_s40_resource_content_view_identity_is_canonical(self) -> None:
        self._configure(self.build_dir, with_filtered=True)
        identities = dict(
            line.split('=', 1)
            for line in (self.build_dir / 'resource_content_ids.txt').read_text().splitlines()
        )

        self.assertEqual(identities['patterns_a'], identities['patterns_b'])
        self.assertEqual(identities['patterns_a'], identities['keep_archive'])
        self.assertEqual(identities['file'], identities['bare_hash'])
        self.assertEqual(identities['file'], identities['uppercase_hash'])
        self.assertEqual(identities['file_content_dir'], identities['bare_hash_content_dir'])
        self.assertIn(self.file_hash[: self.SHA_PREFIX_LEN], identities['bare_hash_content_dir'])
        self.assertNotEqual(identities['patterns_a'], identities['patterns_c'])
        self.assertNotEqual(identities['patterns_a'], identities['patch'])
        self.assertNotEqual(identities['patterns_a'], identities['strip'])
        self.assertNotEqual(identities['file'], identities['renamed_file'])
        self.assertNotEqual(identities['patterns_a'], identities['archive_as_file'])

    # S41: Build-time resource targets publish their resolved content description.
    def test_s41_resource_targets_publish_content_view_properties(self) -> None:
        self._configure(self.build_dir)
        properties = dict(
            line.split('=', 1)
            for line in (self.build_dir / 'resource_target_properties.txt').read_text().splitlines()
        )

        self.assertEqual(properties['file_destination'], self.source_file.name)
        self.assertEqual(properties['file_is_archive'], 'FALSE')
        self.assertEqual(len(properties['file_content_id']), 64)
        self.assertEqual(
            Path(properties['file_content_dir']),
            self.resource_cache_dir
            / 'res_file'
            / self.file_hash[: self.SHA_PREFIX_LEN]
            / 'content',
        )
        self.assertEqual(properties['archive_destination'], 'res_archive')
        self.assertEqual(properties['archive_is_archive'], 'TRUE')
        self.assertEqual(
            Path(properties['archive_content_dir']),
            self.resource_cache_dir
            / 'res_archive'
            / self.archive_hash[: self.SHA_PREFIX_LEN]
            / 'content'
            / 'payload-root',
        )

    # S47: Verification stamps preserve the stripped spelling of EXPECTED_HASH.
    def test_s47_expected_hash_stamp_preserves_stripped_input(self) -> None:
        expected_hash = f'sha256={self.file_hash.upper()}'
        self._configure(
            self.build_dir,
            extra_args=[f'-DTEST_FILE_HASH=  {expected_hash}  '],
        )
        self._build(self.build_dir, target='res_file')

        self.assertEqual(self._file_stamp().read_text(encoding='utf-8'), f'{expected_hash}\n')

    # S42: Repeated RESOURCE clauses create unlinked member targets and bundle dependencies.
    def test_s42_bundle_parser_creates_member_targets(self) -> None:
        state_file = self.build_dir / 'bundle_parser_state.txt'
        self._configure_bundle_project(
            f'''
isaac_add_resource_bundle(test_bundle
  LINK_DIR "${{CMAKE_SOURCE_DIR}}/data/bundle"
  RESOURCE bundle_file_a
    URL "file://{self.source_file}"
    EXPECTED_HASH "SHA256={self.file_hash}"
  RESOURCE bundle_file_b
    URL "file://{self.renamed_source_file}"
    EXPECTED_HASH "SHA256={self.file_hash}"
)
isaac_add_resource_bundle(default_bundle
  RESOURCE default_bundle_file
    URL "file://{self.source_file}"
    EXPECTED_HASH "SHA256={self.file_hash}"
)
get_target_property(_deps test_bundle MANUALLY_ADDED_DEPENDENCIES)
get_target_property(_explicit_link test_bundle ISAAC_RESOURCE_BUNDLE_LINK_DIR)
get_target_property(_members test_bundle ISAAC_RESOURCE_BUNDLE_MEMBERS)
get_target_property(_default_link default_bundle ISAAC_RESOURCE_BUNDLE_LINK_DIR)
file(WRITE "{state_file}"
  "default_link=${{_default_link}}\n"
  "dependencies=${{_deps}}\n"
  "explicit_link=${{_explicit_link}}\n"
  "members=${{_members}}\n")
'''
        )
        self._build(self.build_dir, target='test_bundle')

        state = dict(
            line.split('=', 1) for line in state_file.read_text(encoding='utf-8').splitlines()
        )
        self.assertEqual(state['default_link'], str(self.data_dir / 'default_bundle'))
        self.assertEqual(state['dependencies'], 'bundle_file_a;bundle_file_b')
        self.assertEqual(state['explicit_link'], str(self.data_dir / 'bundle'))
        self.assertEqual(state['members'], 'bundle_file_a;bundle_file_b')
        self.assertFalse((self.data_dir / 'bundle_file_a').exists())
        self.assertFalse((self.data_dir / 'bundle_file_b').exists())

    # S43: Bundle declarations reject ambiguous or unsupported syntax at configure time.
    def test_s43_bundle_parser_rejects_invalid_clauses(self) -> None:
        file_args = f'URL "file://{self.source_file}" EXPECTED_HASH "SHA256={self.file_hash}"'
        cases = {
            'no_resource': ('LINK_DIR "${CMAKE_SOURCE_DIR}/data/bundle"', 'requires RESOURCE'),
            'bundle_config_time': ('CONFIG_TIME', 'cannot use CONFIG_TIME'),
            'reserved_link_dir': (
                f'LINK_DIR RESOURCE RESOURCE member {file_args}',
                'cannot be used as LINK_DIR',
            ),
            'missing_name': (f'RESOURCE URL {file_args}', 'requires a member name'),
            'reserved_name': (f'RESOURCE RESOURCE member {file_args}', 'reserved delimiter'),
            'duplicate_target': (
                f'RESOURCE member {file_args} RESOURCE member {file_args}',
                'duplicate member target',
            ),
            'bundle_named_member': (
                f'RESOURCE test_bundle {file_args}',
                'duplicate member target test_bundle',
            ),
            'member_link_dir': (
                f'RESOURCE member {file_args} LINK_DIR somewhere',
                'member member cannot set LINK_DIR',
            ),
            'member_config_time': (
                f'RESOURCE member {file_args} CONFIG_TIME',
                'member member cannot use CONFIG_TIME',
            ),
            'missing_hash': (
                f'RESOURCE member URL "file://{self.source_file}"',
                'member member requires EXPECTED_HASH',
            ),
            'unknown_bundle_argument': ('BOGUS value', 'unknown bundle argument BOGUS'),
            'duplicate_destination': (
                f'RESOURCE member_a {file_args} RESOURCE member_b {file_args}',
                'duplicate destination sample_data.txt',
            ),
            'resource_pattern': (
                'RESOURCE archive '
                f'URL "file://{self.filtered_archive_file}" '
                f'EXPECTED_HASH "SHA256={self.filtered_archive_hash}" '
                'ARCHIVE_PATTERNS RESOURCE',
                'RESOURCE is a reserved delimiter',
            ),
        }

        for name, (arguments, diagnostic) in cases.items():
            with self.subTest(name=name):
                shutil.rmtree(self.build_dir, ignore_errors=True)
                result = self._configure_bundle_project(
                    f'isaac_add_resource_bundle(test_bundle {arguments})', check=False
                )
                diagnostic_output = ' '.join(result.stderr.split())
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('isaac_add_resource_bundle(test_bundle)', diagnostic_output)
                self.assertIn(diagnostic, diagnostic_output)

    # S44: Bundle assembly exposes files at root, archives as mounts, and preserves artifacts.
    def test_s44_bundle_assembly_layout_and_generated_artifacts(self) -> None:
        state_file = self.build_dir / 'bundle_assembly_state.txt'
        self._configure_bundle_project(
            f'''
isaac_add_resource_bundle(layout_bundle
  LINK_DIR "${{CMAKE_SOURCE_DIR}}/data/layout"
  RESOURCE layout_file_a
    URL "file://{self.source_file}"
    EXPECTED_HASH "SHA256={self.file_hash}"
  RESOURCE layout_file_b
    URL "file://{self.renamed_source_file}"
    EXPECTED_HASH "SHA256={self.file_hash}"
  RESOURCE layout_archive
    URL "file://{self.archive_file}"
    EXPECTED_HASH "SHA256={self.archive_hash}"
    STRIP_PREFIX payload-root
)
get_target_property(_content layout_bundle ISAAC_RESOURCE_BUNDLE_CONTENT_DIR)
get_target_property(_id layout_bundle ISAAC_RESOURCE_BUNDLE_CONTENT_ID)
file(WRITE "{state_file}" "content=${{_content}}\nid=${{_id}}\n")
'''
        )
        self._build(self.build_dir, target='layout_bundle')

        state = dict(
            line.split('=', 1) for line in state_file.read_text(encoding='utf-8').splitlines()
        )
        bundle = self.data_dir / 'layout'
        content = Path(state['content'])
        cached_file_a = (
            self.resource_cache_dir
            / 'layout_file_a'
            / self.file_hash[: self.SHA_PREFIX_LEN]
            / 'content'
            / self.source_file.name
        )
        cached_file_b = (
            self.resource_cache_dir
            / 'layout_file_b'
            / self.file_hash[: self.SHA_PREFIX_LEN]
            / 'content'
            / self.renamed_source_file.name
        )
        cached_archive = (
            self.resource_cache_dir
            / 'layout_archive'
            / self.archive_hash[: self.SHA_PREFIX_LEN]
            / 'content'
            / 'payload-root'
        )

        self.assertEqual(len(state['id']), 64)
        self.assertTrue(bundle.is_symlink())
        self.assertEqual(bundle.resolve(), content)
        self.assertEqual((bundle / self.source_file.name).resolve(), cached_file_a)
        self.assertEqual((bundle / self.renamed_source_file.name).resolve(), cached_file_b)
        self.assertEqual((bundle / 'layout_archive').resolve(), cached_archive)

        (bundle / 'generated.engine').write_text('root-engine\n', encoding='utf-8')
        (bundle / 'layout_archive' / 'archive.engine').write_text(
            'archive-engine\n', encoding='utf-8'
        )
        member_link = content / self.source_file.name
        member_link.unlink()
        member_link.symlink_to(self.renamed_source_file)
        bundle.unlink()
        self._build(self.build_dir, target='layout_bundle')

        self.assertTrue(bundle.is_symlink())
        self.assertEqual(member_link.resolve(), cached_file_a)
        self.assertEqual((content / 'generated.engine').read_text(), 'root-engine\n')
        self.assertEqual((cached_archive / 'archive.engine').read_text(), 'archive-engine\n')

    # S45: Bundle identity ignores declaration order and changes with one member view.
    def test_s45_bundle_identity_and_independent_member_reuse(self) -> None:
        def body(state_file: Path, file_url: Path, file_hash: str, *, reverse: bool) -> str:
            members = [
                f'''RESOURCE order_file_a
  URL "file://{file_url}"
  EXPECTED_HASH "SHA256={file_hash}"''',
                f'''RESOURCE order_file_b
  URL "file://{self.renamed_source_file}"
  EXPECTED_HASH "SHA256={self.file_hash}"''',
            ]
            if reverse:
                members.reverse()
            return f'''
isaac_add_resource_bundle(order_bundle
  LINK_DIR "${{CMAKE_SOURCE_DIR}}/data/order"
  {' '.join(members)}
)
get_target_property(_content order_bundle ISAAC_RESOURCE_BUNDLE_CONTENT_DIR)
get_target_property(_id order_bundle ISAAC_RESOURCE_BUNDLE_CONTENT_ID)
file(WRITE "{state_file}" "content=${{_content}}\nid=${{_id}}\n")
'''

        state1_file = self.build_dir / 'bundle_identity.txt'
        self._configure_bundle_project(
            body(state1_file, self.source_file, self.file_hash, reverse=False)
        )
        self._build(self.build_dir, target='order_bundle')
        state1 = dict(line.split('=', 1) for line in state1_file.read_text().splitlines())
        (Path(state1['content']) / 'generated.engine').write_text('engine\n', encoding='utf-8')
        unchanged_stamp = (
            self.resource_cache_dir
            / 'order_file_b'
            / self.file_hash[: self.SHA_PREFIX_LEN]
            / f'{self.renamed_source_file.name}.hash.verified'
        )
        stamp_mtime = unchanged_stamp.stat().st_mtime_ns

        state2_file = self.build2_dir / 'bundle_identity.txt'
        self._configure_bundle_project(
            body(state2_file, self.source_file, self.file_hash, reverse=True),
            build_dir=self.build2_dir,
        )
        self._build(self.build2_dir, target='order_bundle')
        state2 = dict(line.split('=', 1) for line in state2_file.read_text().splitlines())
        self.assertEqual(state1, state2)
        self.assertTrue((Path(state2['content']) / 'generated.engine').is_file())
        self.assertEqual(unchanged_stamp.stat().st_mtime_ns, stamp_mtime)

        changed_dir = self.sources_dir / 'changed'
        changed_dir.mkdir()
        changed_file = changed_dir / self.source_file.name
        changed_file.write_text('payload-v2\n', encoding='utf-8')
        changed_hash = self._sha256(changed_file)
        build3_dir = self.workspace / 'build3'
        state3_file = build3_dir / 'bundle_identity.txt'
        self._configure_bundle_project(
            body(state3_file, changed_file, changed_hash, reverse=False), build_dir=build3_dir
        )
        self._build(build3_dir, target='order_bundle')
        state3 = dict(line.split('=', 1) for line in state3_file.read_text().splitlines())

        self.assertNotEqual(state1['id'], state3['id'])
        self.assertNotEqual(state1['content'], state3['content'])
        self.assertFalse((Path(state3['content']) / 'generated.engine').exists())
        self.assertEqual(unchanged_stamp.stat().st_mtime_ns, stamp_mtime)

    # S46: Assembly never replaces a real object at a declared member destination.
    def test_s46_bundle_assembly_rejects_real_destination_collision(self) -> None:
        state_file = self.build_dir / 'bundle_collision_state.txt'
        self._configure_bundle_project(
            f'''
isaac_add_resource_bundle(collision_bundle
  RESOURCE collision_file
    URL "file://{self.source_file}"
    EXPECTED_HASH "SHA256={self.file_hash}"
)
get_target_property(_content collision_bundle ISAAC_RESOURCE_BUNDLE_CONTENT_DIR)
file(WRITE "{state_file}" "${{_content}}")
'''
        )
        content = Path(state_file.read_text(encoding='utf-8'))
        content.mkdir(parents=True)
        (content / self.source_file.name).write_text('do-not-replace\n', encoding='utf-8')

        result = self._build(self.build_dir, target='collision_bundle', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('refusing to replace real object', ' '.join(result.stderr.split()))
        self.assertEqual((content / self.source_file.name).read_text(), 'do-not-replace\n')


if __name__ == '__main__':
    unittest.main(verbosity=2)
