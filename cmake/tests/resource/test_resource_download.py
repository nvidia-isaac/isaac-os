# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small local HTTP fixtures exercise resume and CMake integration."""

import hashlib
import http.server
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / 'cmake/isaac_resource_download.py'
spec = importlib.util.spec_from_file_location('download', HELPER)
assert spec is not None and spec.loader is not None
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
PAYLOAD = bytes(range(256)) * 8192
IDENTITY = 'SHA256=' + hashlib.sha256(PAYLOAD).hexdigest()


class Handler(http.server.BaseHTTPRequestHandler):
    """Serve deterministic range responses and injected failures."""

    def log_message(self, *args):
        """Suppress request logs so test output highlights transfer failures."""
        pass

    def do_GET(self):  # pylint: disable=invalid-name
        """Serve a byte range or inject the selected HTTP or connection failure."""
        server = self.server
        offset = int(self.headers.get('Range', 'bytes=0-')[6:-1])
        server.requests.append((offset, self.headers.get('If-Range')))
        if server.mode == 'offline':
            self.send_error(503)
            return
        if (server.mode == '416' and offset) or offset >= len(PAYLOAD):
            self.send_error(416)
            return
        if server.mode in ('ignore', 'changed'):
            offset = 0
        self.send_response(206 if offset else 200)
        self.send_header('ETag', '"new"' if server.mode == 'changed' else '"v1"')
        self.send_header('Content-Length', str(len(PAYLOAD) - offset))
        if offset:
            start = offset + 1 if server.mode == 'bad_range' else offset
            self.send_header('Content-Range', f'bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}')
        self.end_headers()
        data = PAYLOAD[offset:]
        if server.mode == 'drop' and len(server.requests) == 1:
            self.wfile.write(data[: 512 * 1024])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        try:
            for index in range(0, len(data), 65536):
                self.wfile.write(data[index : index + 65536])
                self.wfile.flush()
                if server.mode == 'gated' and index == 0:
                    if not server.progress_gate.wait(timeout=10):
                        return
                if server.mode == 'slow':
                    time.sleep(0.08)
        except (BrokenPipeError, ConnectionResetError):
            pass


class DownloadTests(unittest.TestCase):
    """Exercise recovery and resource publication through real processes."""

    def setUp(self):
        """Start an isolated loopback server and temporary resource cache for each test."""
        self.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = Path(self.temp.name)
        self.dest = self.root / 'model.bin'
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.mode = 'normal'
        self.server.requests = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/model.bin'

    def tearDown(self):
        """Stop the fixture server and remove its temporary cache."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def seed(self, identity=IDENTITY):
        """Create retained bytes and the current transport identity metadata."""
        Path(str(self.dest) + '.part').write_bytes(PAYLOAD[:524288])
        Path(str(self.dest) + '.part.json').write_text(
            json.dumps({'url': self.url, 'identity': identity}),
            encoding='utf-8',
        )

    def run_download(self):
        """Download without retry delays and assert exact content and partial cleanup."""
        transport.download(self.url, self.dest, IDENTITY, retry_delay=0)
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)
        self.assertFalse(Path(str(self.dest) + '.part').exists())

    def test_resume(self):
        """Request only the missing suffix when retained bytes match the resource identity."""
        self.seed()
        self.run_download()
        self.assertEqual(self.server.requests, [(524288, None)])

    def test_completed_partial_returns_candidate_after_416(self):
        """Return retained complete bytes as a candidate when the server rejects the EOF range."""
        self.seed()
        Path(str(self.dest) + '.part').write_bytes(PAYLOAD)
        self.run_download()
        self.assertEqual(self.server.requests, [(len(PAYLOAD), None)])

    def test_network_drop(self):
        """Resume from bytes retained after the server closes an incomplete response."""
        self.server.mode = 'drop'
        self.run_download()
        self.assertEqual(self.server.requests[1][0], 524288)

    def test_range_fallbacks(self):
        """Restart safely when a server ignores, changes, or rejects the requested range."""
        for mode in ('ignore', 'changed', 'bad_range'):
            with self.subTest(mode=mode):
                self.server.mode = mode
                self.server.requests.clear()
                self.seed()
                self.run_download()
                self.assertEqual(self.server.requests[0][0], 524288)

    def test_changed_identity_and_missing_metadata(self):
        """Discard partials with a different identity or missing resume metadata."""
        self.seed(identity='old')
        self.run_download()
        self.assertEqual(self.server.requests[0][0], 0)
        self.server.requests.clear()
        Path(str(self.dest) + '.part').write_bytes(b'old partial')
        self.run_download()
        self.assertEqual(self.server.requests[0][0], 0)

    def test_process_interruptions(self):
        """Resume retained bytes after interrupting or killing the download process group."""
        for sig in (signal.SIGINT, signal.SIGKILL):
            with self.subTest(signal=sig):
                self.dest.unlink(missing_ok=True)
                self.server.mode = 'slow'
                self.server.requests.clear()
                partial = Path(str(self.dest) + '.part')
                with subprocess.Popen(
                    [sys.executable, str(HELPER), self.url, str(self.dest), IDENTITY],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                ) as process:
                    deadline = time.monotonic() + 10
                    while not partial.exists() or partial.stat().st_size < 262144:
                        self.assertIsNone(process.poll())
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.02)
                    os.killpg(process.pid, sig)
                    process.communicate(timeout=5)
                retained = partial.stat().st_size
                self.server.mode = 'normal'
                self.run_download()
                self.assertEqual(self.server.requests[-1][0], retained)

    def test_exhausted_retries_keep_bytes(self):
        """Preserve partial bytes through an outage and resume once the server recovers."""
        self.seed()
        self.server.mode = 'offline'
        with self.assertRaises(RuntimeError):
            transport.download(self.url, self.dest, IDENTITY, retry_delay=0)
        self.assertEqual(Path(str(self.dest) + '.part').read_bytes(), PAYLOAD[:524288])
        self.assertFalse(self.dest.exists())
        self.server.mode = 'normal'
        self.run_download()
        self.assertEqual(self.server.requests[-1][0], 524288)

    def test_transport_writers_serialize(self):
        """Allow competing download processes to share one verified transfer."""
        self.server.mode = 'slow'
        command = [sys.executable, str(HELPER), self.url, str(self.dest), IDENTITY]
        with subprocess.Popen(command, stdout=subprocess.DEVNULL) as first:
            with subprocess.Popen(command, stdout=subprocess.DEVNULL) as second:
                self.assertEqual(first.wait(timeout=10), 0)
                self.assertEqual(second.wait(timeout=10), 0)
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)

    def test_parent_verifies_416_and_restarts_invalid_candidates(self):
        """Let parent CMake accept complete candidates and restart stale or rejected ranges."""
        fixture = self.root / 'ensure.cmake'
        fixture.write_text(
            f'include("{ROOT}/cmake/isaac_resource_helpers.cmake")\n'
            'isaac_resource_ensure_verified(RESOURCE_LABEL model URL "${URL}" '
            'DEST_FILE "${DEST_FILE}" EXPECTED_HASH "${EXPECTED_HASH}" '
            'VERIFIED_FILE "${DEST_FILE}.verified")\n',
            encoding='utf-8',
        )
        candidate = Path(str(self.dest) + '.tmp')
        for mode, data in [
            ('normal', b'x' * 524288),
            ('416', PAYLOAD[:524288]),
            ('normal', PAYLOAD),
        ]:
            with self.subTest(mode=mode, size=len(data)):
                self.dest.unlink(missing_ok=True)
                Path(str(self.dest) + '.verified').unlink(missing_ok=True)
                self.server.requests.clear()
                self.server.mode = mode
                Path(str(candidate) + '.part').write_bytes(data)
                Path(str(candidate) + '.part.json').write_text(
                    json.dumps({'url': self.url, 'identity': IDENTITY}),
                    encoding='utf-8',
                )
                result = subprocess.run(
                    [
                        'cmake',
                        f'-DURL={self.url}',
                        f'-DDEST_FILE={self.dest}',
                        f'-DEXPECTED_HASH={IDENTITY}',
                        '-P',
                        str(fixture),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                expected_offsets = [len(data)] if data == PAYLOAD else [len(data), 0]
                self.assertEqual([offset for offset, _ in self.server.requests], expected_offsets)
                self.assertEqual(self.dest.read_bytes(), PAYLOAD)
                self.assertFalse(candidate.exists())
                self.assertEqual(result.stdout.count('Verifying resource:'), len(expected_offsets))

    def test_orphaned_curl_keeps_transport_lock(self):
        """Keep competing writers blocked while curl outlives its wrapper process."""
        self.server.mode = 'slow'
        command = [sys.executable, str(HELPER), self.url, str(self.dest), IDENTITY]
        partial = Path(str(self.dest) + '.part')
        with subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ) as first:
            deadline = time.monotonic() + 10
            while not partial.exists() or partial.stat().st_size < 262144:
                self.assertIsNone(first.poll())
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.02)
            first.kill()
            first.wait(timeout=5)
            with subprocess.Popen(command, stdout=subprocess.DEVNULL) as second:
                self.assertEqual(second.wait(timeout=10), 0)
        self.assertEqual([offset for offset, _ in self.server.requests], [0, len(PAYLOAD)])
        self.assertEqual(self.dest.read_bytes(), PAYLOAD)

    def test_checksum_failure_does_not_publish(self):
        """Reject invalid content without publishing a final file, stamp, or resource link."""
        cache = self.root / 'cache'
        link = self.root / 'published'
        command = [
            'cmake',
            f'-DURL={self.url}',
            '-DRESOURCE_LABEL=model',
            f'-DCACHE_DIR={cache}',
            f'-DLINK_DIR={link}',
            '-DEXPECTED_HASH=SHA256=' + '0' * 64,
            '-P',
            str(ROOT / 'cmake/isaac_resource_ensure_script.cmake'),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Checksum mismatch', result.stderr)
        self.assertFalse(link.exists())
        self.assertEqual(list(cache.rglob('*.verified')), [])
        self.assertEqual(list(cache.rglob('model.bin')), [])

    def test_downloads_verify_once_and_reuse_stamp(self):
        """Hash fresh HTTP and local downloads once, then reuse the verification stamp."""
        source = self.root / 'source.bin'
        source.write_bytes(PAYLOAD)
        for scheme, url in [('http', self.url), ('file', source.as_uri())]:
            with self.subTest(scheme=scheme):
                cache = self.root / scheme
                link = self.root / f'{scheme}-published'
                command = [
                    'cmake',
                    f'-DURL={url}',
                    '-DRESOURCE_LABEL=model',
                    f'-DCACHE_DIR={cache}',
                    f'-DLINK_DIR={link}',
                    f'-DEXPECTED_HASH={IDENTITY}',
                    '-P',
                    str(ROOT / 'cmake/isaac_resource_ensure_script.cmake'),
                ]
                first = subprocess.run(command, capture_output=True, text=True, check=True)
                self.assertEqual(first.stdout.count('Verifying resource:'), 1)
                stamps = list(cache.rglob('*.verified'))
                self.assertEqual(len(stamps), 1)
                self.assertEqual(stamps[0].read_text().strip(), IDENTITY)
                self.assertEqual((link / Path(url).name).read_bytes(), PAYLOAD)
                self.assertEqual(sorted(path.name for path in link.iterdir()), [Path(url).name])
                second = subprocess.run(command, capture_output=True, text=True, check=True)
                self.assertNotIn('Verifying resource:', second.stdout)
        self.assertEqual(len(self.server.requests), 1)

    def test_shared_verifier_for_predownloaded_files(self):
        """Use the same checksum check for local files, with optional publication and stamp."""
        fixture = self.root / 'verify.cmake'
        fixture.write_text(
            f'include("{ROOT}/cmake/isaac_resource_helpers.cmake")\n'
            'isaac_resource_verify(DEST_FILE "${DEST_FILE}" EXPECTED_HASH "${EXPECTED_HASH}" '
            'PUBLISH_FILE "${PUBLISH_FILE}" VERIFIED_FILE "${VERIFIED_FILE}")\n',
            encoding='utf-8',
        )
        source = self.root / 'predownloaded.bin'
        stamp = self.root / 'model.verified'
        for identity in [IDENTITY, IDENTITY.partition('=')[2], 'SHA256=' + '0' * 64]:
            with self.subTest(identity=identity):
                source.write_bytes(PAYLOAD)
                self.dest.unlink(missing_ok=True)
                stamp.unlink(missing_ok=True)
                command = [
                    'cmake',
                    f'-DDEST_FILE={source}',
                    f'-DEXPECTED_HASH={identity}',
                    f'-DPUBLISH_FILE={self.dest}',
                    f'-DVERIFIED_FILE={stamp}',
                    '-P',
                    str(fixture),
                ]
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                self.assertEqual(result.stdout.count('Verifying resource:'), 1)
                if identity.endswith('0' * 64):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(self.dest.exists())
                    self.assertFalse(stamp.exists())
                    self.assertEqual(source.read_bytes(), PAYLOAD)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(self.dest.read_bytes(), PAYLOAD)
                    self.assertFalse(source.exists())
                    self.assertEqual(stamp.read_text().strip(), identity)

    def test_ninja_progress_and_explicit_phase(self):
        """Stream progress before Ninja exits and warm resources without compiling the app."""
        self.server.mode = 'gated'
        self.server.progress_gate = threading.Event()
        source = self.root / 'source'
        source.mkdir()
        (source / 'CMakeLists.txt').write_text(
            f'''cmake_minimum_required(VERSION 3.20)
project(download_fixture NONE)
list(APPEND CMAKE_MODULE_PATH "{ROOT}/cmake")
set(ISAAC_RESOURCE_CACHE "${{CMAKE_BINARY_DIR}}/resources")
include(isaac_add_resource)
isaac_add_resource_bundle(models RESOURCE model URL "{self.url}" EXPECTED_HASH "{IDENTITY}")
add_custom_target(application
  COMMAND "${{CMAKE_COMMAND}}" -E touch "${{CMAKE_BINARY_DIR}}/compiled")
add_dependencies(application models)
''',
            encoding='utf-8',
        )
        build = self.root / 'build'
        subprocess.run(['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja'], check=True)
        with subprocess.Popen(
            ['cmake', '--build', str(build), '--target', 'isaac_download_resources'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            live = False
            for line in process.stdout:
                if re.match(r'\s*[1-9]\d*\s+\d', line):
                    live = live or process.poll() is None
                    self.server.progress_gate.set()
            self.assertEqual(process.wait(timeout=10), 0)
        self.assertTrue(live, 'Progress must reach the pipe before Ninja finishes.')
        self.assertFalse((build / 'compiled').exists())
        self.assertEqual((source / 'data/models/model.bin').read_bytes(), PAYLOAD)
        count = len(self.server.requests)
        subprocess.run(['cmake', '--build', str(build), '--target', 'application'], check=True)
        self.assertTrue((build / 'compiled').exists())
        self.assertEqual(len(self.server.requests), count)


if __name__ == '__main__':
    unittest.main()
