#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge the per-app build/<app>/compile_commands.json files into a single repo-root
compile_commands.json for clangd, rewriting the in-container build path prefix to this checkout.

Builds run inside the dev container where the repo is mounted at /workspace/isaac_os, so every
generated compile DB embeds that prefix. The IDE/clangd sees the repo at its real checkout path,
so the paths must be rewritten. Re-run this after each build.

Usage: python3 tools/sync_compile_commands.py
"""

import json
from pathlib import Path

# Path the repo is mounted at *inside the build container*.
CONTAINER_PREFIX = '/workspace/isaac_os'


def main() -> None:
    """
    Merge the per-app compile_commands.json files into a single repo-root compile_commands.json.
    """
    repo_root = Path(__file__).resolve().parent.parent
    # Sort by mtime (oldest first) so the newest build is processed last and wins the dedupe.
    dbs = sorted(
        repo_root.glob('build/*/compile_commands.json'),
        key=lambda p: (p.stat().st_mtime, p),
    )
    if not dbs:
        raise SystemExit('No build/<app>/compile_commands.json found — build an app first.')

    merged: dict[str, dict] = {}
    for path in dbs:
        for e in json.loads(path.read_text()):
            for k in ('file', 'directory', 'command', 'output'):
                if isinstance(e.get(k), str):
                    e[k] = e[k].replace(CONTAINER_PREFIX, str(repo_root))
            if 'arguments' in e:
                e['arguments'] = [
                    a.replace(CONTAINER_PREFIX, str(repo_root)) for a in e['arguments']
                ]
            merged[e['file']] = e  # dedupe by file; latest build wins

    out_path = repo_root / 'compile_commands.json'
    out_path.write_text(json.dumps(list(merged.values()), indent=1))
    print(f'Wrote {len(merged)} entries to {out_path} (from {len(dbs)} per-app DBs).')


if __name__ == '__main__':
    main()
