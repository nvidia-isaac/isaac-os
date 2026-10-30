# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resumable curl transport; parent CMake verifies and publishes returned candidates."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def reset_partial(partial, metadata):
    """Invalidate metadata before removing bytes, including across crashes."""
    metadata.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)


def load_partial_state(url, identity, partial, metadata):
    """Load resume metadata only when it belongs to the requested resource."""
    try:
        state = json.loads(metadata.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    if state.get('url') != url or state.get('identity') != identity:
        reset_partial(partial, metadata)
        state = {}
    return state


def save_partial_state(metadata, state):
    """Persist resume metadata atomically before writing response bytes."""
    temporary_metadata = Path(str(metadata) + '.new')
    with temporary_metadata.open('w', encoding='utf-8') as stream:
        json.dump(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_metadata, metadata)


def download_attempt(url, destination, identity, lock_fds):
    """Download into a partial file and return a stable candidate for parent verification."""
    partial = Path(str(destination) + '.part')
    metadata = Path(str(partial) + '.json')
    load_partial_state(url, identity, partial, metadata)
    offset = partial.stat().st_size if partial.exists() else 0
    save_partial_state(metadata, {'url': url, 'identity': identity})
    print(f'Downloading {destination.name}: continuing from {offset:,} bytes.', flush=True)
    # Retry outside curl: its internal retries can discard bytes from a failed attempt.
    # Inherit the lock so an orphaned curl still excludes competing cache writers.
    result = subprocess.run(
        [
            'curl',
            '--disable',
            '--fail',
            '--location',
            '--globoff',
            '--continue-at',
            '-',
            '--output',
            str(partial),
            '--connect-timeout',
            '60',
            '--speed-limit',
            '1',
            '--speed-time',
            '60',
            '--proto',
            '=http,https',
            '--proto-redir',
            '=http,https',
            '--header',
            'Accept-Encoding: identity',
            '--write-out',
            '%{http_code}',
            '--url',
            url,
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
        pass_fds=lock_fds,
    )
    # HTTP 416 may mean the retained bytes are complete. Only parent CMake's
    # checksum check can decide; hand it a candidate even for rejected ranges.
    if result.returncode == 0 or result.stdout == '416':
        with partial.open('rb') as stream:
            os.fsync(stream.fileno())
        # Finish staging under the transport lock. Orphaned curl processes write
        # only .part files, never the candidate being verified by parent CMake.
        os.replace(partial, destination)
        metadata.unlink(missing_ok=True)
        return
    if result.returncode == 33:
        reset_partial(partial, metadata)
    raise OSError(f'curl exited with code {result.returncode} (HTTP {result.stdout}).')


def download(url, destination, identity, attempts=3, retry_delay=5, *, lock_fds=()):
    """Retry resumable HTTP downloads without discarding retained partial bytes."""
    partial = Path(str(destination) + '.part')
    for attempt in range(1, attempts + 1):
        try:
            download_attempt(url, destination, identity, lock_fds)
            return
        except (OSError, ValueError) as error:
            print(
                f'Download attempt {attempt} of {attempts} failed: {error}',
                file=sys.stderr,
                flush=True,
            )
            if attempt == attempts:
                raise RuntimeError(
                    f'Unable to download {destination.name}. '
                    f'Any resumable bytes are retained at {partial}; '
                    'rerun the same command to retry.'
                ) from error
            time.sleep(retry_delay)


def main():
    """Run a download under the transport lock and report actionable failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('destination', type=Path)
    parser.add_argument('identity')
    args = parser.parse_args()
    try:
        # Also protect transport writes if the CMake parent alone is killed.
        # The lock file must remain in place so all writers use the same inode.
        with Path(str(args.destination) + '.download.lock').open('a', encoding='utf-8') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not args.destination.exists():
                download(
                    args.url,
                    args.destination,
                    args.identity,
                    lock_fds=(lock.fileno(),),
                )
    except KeyboardInterrupt:
        print(
            'Download interrupted. Partial bytes are retained; rerun the same command to continue.',
            file=sys.stderr,
            flush=True,
        )
        return 130
    except RuntimeError as error:
        print(error, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
