# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REQUIRED_CORE_COUNT = 4
GR00T_FIFO_PRIORITY = 5
WBC_FIFO_PRIORITY = 10
EXPECTED_MESSAGE_COUNT = 125
AFFINITY_POLL_INTERVAL_SECONDS = 0.01
SCHED_RESET_ON_FORK = getattr(os, 'SCHED_RESET_ON_FORK', 0)


def fifo_available() -> bool:
    policy = os.sched_getscheduler(0)
    parameter = os.sched_getparam(0)
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(GR00T_FIFO_PRIORITY))
    except PermissionError:
        return False
    finally:
        if os.sched_getscheduler(0) == os.SCHED_FIFO:
            os.sched_setscheduler(0, policy, parameter)
    return True


def thread_placements(pid: int) -> list[tuple[frozenset[int], int, int]]:
    placements: list[tuple[frozenset[int], int, int]] = []
    try:
        tasks = list(Path(f'/proc/{pid}/task').iterdir())
    except FileNotFoundError:
        return placements
    for task in tasks:
        try:
            tid = int(task.name)
            placements.append(
                (
                    frozenset(os.sched_getaffinity(tid)),
                    os.sched_getscheduler(tid) & ~SCHED_RESET_ON_FORK,
                    os.sched_getparam(tid).sched_priority,
                )
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
    return placements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-config')
    parser.add_argument('--executable', required=True)
    args = parser.parse_args()
    if not args.model_config:
        parser.error('--model-config is required')
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < REQUIRED_CORE_COUNT:
        print(f'Benchmark affinity smoke test requires {REQUIRED_CORE_COUNT} available CPU cores.')
        return 77
    controller, background, gr00t, wbc = cores[:REQUIRED_CORE_COUNT]
    check_fifo = fifo_available()
    command = [
        args.executable,
        f'--message-count={EXPECTED_MESSAGE_COUNT}',
        f'--benchmark-controller-core={controller}',
        f'--benchmark-background-core={background}',
        f'--benchmark-gr00t-core={gr00t}',
        f'--benchmark-wbc-core={wbc}',
    ]
    command.extend(
        (
            f'--application-config-path={args.config}',
            f'--model-config-path={args.model_config}',
        )
    )
    if not check_fifo:
        command.append('--benchmark-affinity-only')
    expected_affinities = {
        frozenset({controller}): 1,
        frozenset({background}): 1,
        frozenset({gr00t}): 1,
        frozenset({wbc}): 1,
    }
    observed = False
    last_placements: list[tuple[frozenset[int], int, int]] = []
    with (
        tempfile.TemporaryFile(mode='w+', encoding='utf-8') as output,
        subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, text=True) as process,
    ):
        while process.poll() is None:
            placements = thread_placements(process.pid)
            last_placements = placements
            affinities = [placement[0] for placement in placements]
            if not observed and all(
                affinities.count(mask) >= count for mask, count in expected_affinities.items()
            ):
                observed = not check_fifo or (
                    (frozenset({gr00t}), os.SCHED_FIFO, GR00T_FIFO_PRIORITY) in placements
                    and (frozenset({wbc}), os.SCHED_FIFO, WBC_FIFO_PRIORITY) in placements
                )
            time.sleep(AFFINITY_POLL_INTERVAL_SECONDS)
        output.seek(0)
        application_output = output.read()

    if process.returncode != 0:
        sys.stderr.write(application_output)
        raise RuntimeError(f'benchmark application exited with {process.returncode}')
    if not observed:
        sys.stderr.write(application_output)
        raise RuntimeError(
            f'did not observe the requested runtime thread placements; last observed: '
            f'{last_placements}'
        )
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f'Benchmark affinity smoke test failed: {error}', file=sys.stderr)
        sys.exit(1)
