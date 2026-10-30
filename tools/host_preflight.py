#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect the Isaac OS Thor reference host without changing system settings."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from collections.abc import Mapping
from datetime import datetime
from datetime import timezone
from functools import partial
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import textwrap
from typing import Any

GUIDE = 'docs/how-to/optimize_your_host.mdx'
REFERENCE_REVISION = 'thor-120w-rt/v2'
WARNING = 'warning'
ERROR = 'error'
CANONICAL_ISOLATED_CPUS = set(range(8, 14))
CANONICAL_HOUSEKEEPING_CPUS = set(range(8))
CANONICAL_ONLINE_CPUS = set(range(14))
ISOLATED_CPU_PATH = Path('/sys/devices/system/cpu/isolated')
NOHZ_FULL_PATH = Path('/sys/devices/system/cpu/nohz_full')
ONLINE_CPU_PATH = Path('/sys/devices/system/cpu/online')
CMDLINE_PATH = Path('/proc/cmdline')
DEFAULT_IRQ_AFFINITY_PATH = Path('/proc/irq/default_smp_affinity')
WORKQUEUE_CPU_MASK_PATH = Path('/sys/devices/virtual/workqueue/cpumask')
RT_RUNTIME_PATH = Path('/proc/sys/kernel/sched_rt_runtime_us')
NVPMODEL_CONFIG_PATH = Path('/etc/nvpmodel.conf')


def parse_cpu_list(value: str) -> set[int]:
    """Parse the kernel's comma-separated CPU-list format."""
    cpus: set[int] = set()
    for part in value.strip().split(','):
        if not part:
            continue
        bounds = part.split('-', maxsplit=1)
        start = int(bounds[0])
        end = int(bounds[-1])
        if start < 0 or end < start:
            raise ValueError(f'invalid CPU range: {part}')
        cpus.update(range(start, end + 1))
    return cpus


def format_cpu_list(cpus: set[int]) -> str:
    """Return a stable explicit CPU list for evidence output."""
    return ','.join(str(cpu) for cpu in sorted(cpus))


def parse_cpu_mask(value: str) -> set[int]:
    """Parse a Linux hexadecimal CPU mask."""
    mask = int(value.strip().replace(',', ''), 16)
    return {cpu for cpu in range(mask.bit_length()) if mask & (1 << cpu)}


def parse_cmdline_cpu_option(cmdline: str, name: str) -> set[int]:
    """Read a CPU list from a kernel command-line option."""
    prefix = f'{name}='
    value = next((item[len(prefix) :] for item in cmdline.split() if item.startswith(prefix)), None)
    if value is None:
        raise RuntimeError(f'The kernel command line does not define {name}.')
    cpu_parts = [part for part in value.split(',') if part not in {'domain', 'managed_irq', 'nohz'}]
    return parse_cpu_list(','.join(cpu_parts))


def validate_jetson_clocks(output: str, *, cpus: set[int]) -> dict[str, str]:
    """Require fixed CPU, GPU, and EMC frequencies from jetson_clocks --show."""
    differences = []
    for cpu in sorted(cpus):
        match = re.search(
            rf'^cpu{cpu}:\s+Governor=(\S+) MinFreq=(\d+) MaxFreq=(\d+) CurrentFreq=(\d+)',
            output,
            re.MULTILINE,
        )
        if match is None:
            raise OSError(f'jetson_clocks did not report CPU {cpu}.')
        governor, minimum, maximum, current = match.groups()
        if governor != 'performance' or len({minimum, maximum, current}) != 1:
            differences.append(f'CPU {cpu} is not fixed at one performance frequency.')

    accelerator_rows = re.findall(
        r'^(gpu-\S+|EMC) MinFreq=(\d+) MaxFreq=(\d+) CurrentFreq=(\d+)',
        output,
        re.MULTILINE,
    )
    labels = {row[0] for row in accelerator_rows}
    if 'EMC' not in labels or not any(label.startswith('gpu-') for label in labels):
        raise OSError('jetson_clocks did not report both GPU and EMC frequencies.')
    for label, minimum, maximum, current in accelerator_rows:
        if len({minimum, maximum, current}) != 1:
            differences.append(f'{label} is not fixed at one frequency.')
    if differences:
        raise RuntimeError(' '.join(differences))
    return {'fixed_clocks': 'pass', 'clock_source': 'jetson_clocks --show'}


def parse_power_mode(output: str) -> tuple[str, str]:
    """Read the selected mode independently of the reference power budget."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith('NV Power Mode:') or not lines[-1].isdigit():
        raise OSError('nvpmodel did not report a power mode and ID.')
    return lines[-1], lines[0].partition(':')[2].strip()


def validate_power_mode(output: str) -> dict[str, str]:
    """Require Thor's 120 W nvpmodel mode 1."""
    mode_id, name = parse_power_mode(output)
    if name != '120W' or mode_id != '1':
        raise RuntimeError('nvpmodel must report 120W power mode ID 1.')
    return {'nvpmodel_mode': '1', 'power_mode': '120W'}


def parse_mode_limits(config: str, mode_id: str, name: str) -> dict[str, int]:
    """Read the selected power-model section, excluding comments and other modes."""
    config = re.sub(r'#[^\n]*', '', config)
    sections = re.findall(r'<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=([^>]+)>([^<]*)', config)
    selected = [(label.strip(), body) for key, label, body in sections if key == mode_id]
    if len(selected) != 1 or selected[0][0] != name:
        raise ValueError(
            f'{NVPMODEL_CONFIG_PATH} must define selected mode {mode_id} ({name}) once.'
        )
    limits = {}
    for line in selected[0][1].splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[1] != 'MAX_FREQ':
            continue
        if len(fields) != 3 or fields[0] in limits:
            raise ValueError(f'Invalid or duplicate maximum frequency: {line.strip()}')
        limits[fields[0]] = int(fields[2])
    return limits


def clock_targets(cpus: set[int]) -> dict[str, tuple[str, int, Path]]:
    """Map reported clocks to power-model names, units, and supported frequency tables."""
    targets = {
        f'cpu{cpu}': (
            f'CPU_AE_{cpu}',
            1000,
            Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_available_frequencies'),
        )
        for cpu in sorted(cpus)
    }
    for label, parameter, device in (
        ('gpu-gpc-0', 'GPU', 'gpu-gpc-0'),
        ('gpu-nvd-0', 'VIDEO', 'gpu-nvd-0'),
        ('EMC', 'EMC', 'bwmgr'),
    ):
        targets[label] = (parameter, 1, Path(f'/sys/class/devfreq/{device}/available_frequencies'))
    return targets


def maximum_frequency(limit: int, unit: int, table: Path) -> int:
    """Resolve nvpmodel's unlimited (-1) setting using the supported frequency table."""
    if limit == -1:
        frequencies = [int(value) for value in table.read_text(encoding='utf-8').split()]
        if not frequencies or min(frequencies) <= 0:
            raise ValueError(f'{table} does not contain positive supported frequencies.')
        return max(frequencies) * unit
    if limit <= 0:
        raise ValueError(f'Invalid maximum frequency: {limit}.')
    return limit * unit


def observed_clock_frequencies(output: str) -> dict[str, list[int]]:
    """Normalize minimum, maximum, and current CPU and accelerator readings to hertz."""
    readings = {}
    for label, minimum, maximum, current in re.findall(
        r'^(cpu\d+):\s+Governor=\S+ MinFreq=(\d+) MaxFreq=(\d+) CurrentFreq=(\d+)',
        output,
        re.MULTILINE,
    ):
        readings[label] = [int(value) * 1000 for value in (minimum, maximum, current)]
    for label, minimum, maximum, current in re.findall(
        r'^(gpu-\S+|EMC) MinFreq=(\d+) MaxFreq=(\d+) CurrentFreq=(\d+)', output, re.MULTILINE
    ):
        readings[label] = [int(value) for value in (minimum, maximum, current)]
    return readings


def idle_evidence() -> dict[str, object]:
    """Record exposed idle-state controls; PM QoS is not a substitute for disabling idle."""
    states = {}
    online = parse_cpu_list(ONLINE_CPU_PATH.read_text(encoding='utf-8'))
    if not online:
        raise ValueError('The online CPU list is empty.')
    for cpu in sorted(online):
        paths = sorted(Path(f'/sys/devices/system/cpu/cpu{cpu}/cpuidle').glob('state*/disable'))
        if not paths:
            raise OSError(f'CPU {cpu} does not expose idle-state controls.')
        states.update({str(path): path.read_text(encoding='utf-8').strip() for path in paths})
    return {'idle_states': states, 'deep_idle_disabled': all(v == '1' for v in states.values())}


def read_nohz_full_cpus() -> set[int] | None:
    """Read optional tickless CPUs; absent files and null masks have no CPU evidence."""
    try:
        value = NOHZ_FULL_PATH.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        return None
    # Kernel cpumask formatting can emit this when the full-tickless mask is unset.
    if value == '(null)':
        return None
    return parse_cpu_list(value)


def read_kernel_config() -> tuple[str | None, str]:
    """Read available running-kernel configuration without requiring config export support."""
    for path in (Path('/proc/config.gz'), Path('/boot') / f'config-{platform.release()}'):
        try:
            if path.suffix == '.gz':
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    return str(path), stream.read()
            return str(path), path.read_text(encoding='utf-8')
        except OSError:
            continue
    return None, ''


def kernel_config_flags(config: str) -> dict[str, str | None]:
    """Distinguish enabled, disabled and unavailable options without inferring runtime state."""
    lines = set(config.splitlines())
    flags = {}
    for name in (
        'CONFIG_PREEMPT_RT',
        'CONFIG_EFI_DISABLE_RUNTIME',
        'CONFIG_NO_HZ_FULL',
        'CONFIG_RCU_NOCB_CPU',
    ):
        value = None
        if f'{name}=y' in lines:
            value = 'y'
        elif f'# {name} is not set' in lines:
            value = 'n'
        flags[name] = value
    return flags


def kernel_evidence(*, require_rt: bool = True) -> dict[str, object]:
    """Require running PREEMPT_RT and separate boot requests from kernel configuration."""
    realtime = Path('/sys/kernel/realtime')
    try:
        observed = realtime.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        observed = 'unavailable'
    if require_rt and observed != '1':
        raise RuntimeError('The running kernel is not PREEMPT_RT (/sys/kernel/realtime != 1).')
    source, config = read_kernel_config()
    flags = kernel_config_flags(config)
    cmdline = CMDLINE_PATH.read_text(encoding='utf-8').strip()
    return {
        'preempt_rt': 'pass' if observed == '1' else 'unverified',
        'kernel_release': platform.release(),
        'realtime_sysfs': observed,
        'kernel_config_source': source,
        'kernel_config': flags,
        'kernel_cmdline': cmdline,
        'efi_boot_arguments': [arg for arg in cmdline.split() if arg.startswith('efi=')],
        'efi_runtime_observed': 'unknown',
    }


class Report:
    """Collect independent observations so one mismatch does not hide the remaining checks."""

    def __init__(self) -> None:
        self.evidence: dict[str, object] = {}
        self.checks: list[dict[str, str]] = []

    def add(self, name: str, status: str, message: str) -> None:
        """Append an actionable finding without modifying the host."""
        self.checks.append({'name': name, 'status': status, 'message': message})

    def check(
        self,
        name: str,
        reader: Callable[[], Mapping[str, object]],
        advice: str,
        severity: str = WARNING,
    ) -> None:
        """Retain successful readings, distinguishing tuning differences from inspection errors."""
        try:
            self.evidence.update(reader())
        except RuntimeError as error:
            self.add(name, severity, f'{error} {advice}')
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            label = CHECK_LABELS.get(name, name)
            self.add(name, ERROR, f'Unable to inspect {label}: {error} {advice}')
        else:
            self.add(name, 'pass', 'Matches the reference setting.')

    def compare(
        self, key: str, reader: Callable[[], str], expected: str, status: str, advice: str
    ) -> None:
        """Save an observed value and describe how it differs from the reference."""
        try:
            observed = reader()
            self.evidence[key] = observed
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.add(key, ERROR, f'Unable to inspect {key}: {error} {advice}')
            return
        if observed == expected:
            self.add(key, 'pass', f'Observed {observed}.')
        else:
            self.add(
                key,
                status,
                f'Observed {observed!r}; reference: {expected!r}. {advice}',
            )

    def document(self) -> dict[str, Any]:
        """Distinguish completed inspection from matching the reference profile."""
        return {
            'schema': 'host-preflight/v1',
            'profile': 'thor',
            'reference_revision': REFERENCE_REVISION,
            'captured_at_utc': datetime.now(timezone.utc).isoformat(),
            'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reference_scope': (
                'Checked settings and selected-mode clock limits; not sustained performance.'
            ),
            'reference_frequencies_verified': self.evidence.get('mode_maximum_clocks') == 'pass'
            and self.evidence.get('fixed_clocks') == 'pass',
            'guide': GUIDE,
            **self.evidence,
            'checks': self.checks,
            'inspection_complete': not any(c['status'] == ERROR for c in self.checks),
            'reference_match': all(c['status'] == 'pass' for c in self.checks),
        }


def query(argv: list[str]) -> str:
    """Bound read-only external queries and preserve their diagnostics."""
    result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=10)
    if result.returncode:
        raise OSError(
            (result.stderr or result.stdout).strip() or f'{argv[0]} exited {result.returncode}.'
        )
    return result.stdout


def desktop_state() -> str:
    """Inspect the display manager, without claiming to detect manually launched compositors."""
    output = query(
        ['systemctl', 'show', 'display-manager.service', '-p', 'LoadState', '-p', 'ActiveState']
    )
    properties = {
        key: value
        for line in output.splitlines()
        if '=' in line
        for key, value in [line.split('=', 1)]
    }
    if properties.get('LoadState') == 'not-found':
        return 'inactive'
    state = properties.get('ActiveState')
    if not state:
        raise ValueError('systemctl did not report display-manager.service activity.')
    return state


def read_cpu_set(path: Path, parser: Callable[[str], set[int]]) -> str:
    """Read a sysfs CPU list or mask and normalize it for comparison."""
    return format_cpu_list(parser(path.read_text(encoding='utf-8')))


def inspect_cpu_layout(report: Report) -> None:
    """Compare the Thor layout and housekeeping observations independently."""
    for key, path, parser, expected, severity in (
        ('online_cpus', ONLINE_CPU_PATH, parse_cpu_list, CANONICAL_ONLINE_CPUS, 'info'),
        ('isolated_cpus', ISOLATED_CPU_PATH, parse_cpu_list, CANONICAL_ISOLATED_CPUS, WARNING),
        (
            'default_irq_cpus',
            DEFAULT_IRQ_AFFINITY_PATH,
            parse_cpu_mask,
            CANONICAL_HOUSEKEEPING_CPUS,
            WARNING,
        ),
        (
            'unbound_workqueue_cpus',
            WORKQUEUE_CPU_MASK_PATH,
            parse_cpu_mask,
            CANONICAL_HOUSEKEEPING_CPUS,
            WARNING,
        ),
    ):
        report.compare(
            key,
            partial(read_cpu_set, path, parser),
            format_cpu_list(expected),
            severity,
            'See Reduce scheduling interference in the host guide.',
        )


def inspect_boot_settings(report: Report) -> None:
    """Record boot requests separately from observed kernel state."""
    cmdline = CMDLINE_PATH.read_text(encoding='utf-8').strip()
    report.evidence['kernel_cmdline'] = cmdline
    for name, expected in (
        ('isolcpus', CANONICAL_ISOLATED_CPUS),
        ('irqaffinity', CANONICAL_HOUSEKEEPING_CPUS),
        ('kthread_cpus', CANONICAL_HOUSEKEEPING_CPUS),
    ):
        value = next((a.partition('=')[2] for a in cmdline.split() if a.startswith(f'{name}=')), '')
        cpus = parse_cmdline_cpu_option(cmdline, name) if value else set()
        report.compare(
            f'requested_{name}',
            partial(format_cpu_list, cpus),
            format_cpu_list(expected),
            'info',
            'The reference boot options are in the host guide; reboot after editing.',
        )
        if name == 'isolcpus' and not {'domain', 'managed_irq'} <= set(value.split(',')):
            report.add(
                'isolation_flags',
                WARNING,
                'The reference uses isolcpus=domain,managed_irq to reduce scheduling '
                'and interrupt interference.',
            )
    nohz = read_nohz_full_cpus()
    rcu = parse_cmdline_cpu_option(cmdline, 'rcu_nocbs') if 'rcu_nocbs=' in cmdline else None
    report.evidence.update(
        nohz_full_cpus=None if nohz is None else format_cpu_list(nohz),
        rcu_nocbs_cpus=None if rcu is None else format_cpu_list(rcu),
        tickless_rcu_requirements='not-required',
        efi_boot_arguments=[arg for arg in cmdline.split() if arg.startswith('efi=')],
        efi_runtime_observed='unknown',
    )
    inspect_tickless_support(report, cmdline, nohz)


def inspect_tickless_support(report: Report, cmdline: str, nohz: set[int] | None) -> None:
    """Report kernel build support for the requested optional tickless and RCU features."""
    parameters = {key: value for arg in cmdline.split() for key, _, value in [arg.partition('=')]}
    required = []
    if 'nohz_full' in parameters or nohz or 'nohz' in parameters.get('isolcpus', '').split(','):
        required.append('CONFIG_NO_HZ_FULL')
    if {'rcu_nocbs', 'rcu_nocb_poll'} & parameters.keys():
        required.append('CONFIG_RCU_NOCB_CPU')
    if not required:
        return
    flags = report.evidence.get('kernel_config')
    source = report.evidence.get('kernel_config_source')
    if not isinstance(flags, dict):
        source, config = read_kernel_config()
        flags = kernel_config_flags(config)
        report.evidence.update(kernel_config_source=source, kernel_config=flags)
    missing = ', '.join(option for option in required if flags.get(option) != 'y')
    if not source:
        message = (
            'Tickless/RCU boot options are set; the running kernel configuration is unavailable.'
        )
    elif missing:
        message = (
            f'Tickless/RCU boot options are set, but this kernel lacks {missing}. '
            'Remove the unsupported boot options or use a kernel with that support enabled.'
        )
    else:
        message = (
            'Tickless/RCU tuning is configured; this differs from the reference configuration.'
        )
    report.add('tickless_rcu', 'info', message)


def inspect_power_and_clocks(report: Report) -> None:
    """Preserve raw readings even when they differ from the reference."""
    cpus = parse_cpu_list(str(report.evidence.get('online_cpus', '')))
    for key, argv, validator, advice in (
        (
            'nvpmodel_output',
            ['nvpmodel', '-q'],
            validate_power_mode,
            'Use 120W mode 1 for reference comparisons; choose another mode '
            'for different power budgets.',
        ),
        (
            'jetson_clocks_output',
            ['sudo', '-n', 'jetson_clocks', '--show'],
            partial(validate_jetson_clocks, cpus=cpus),
            'Run sudo -v, then sudo -n jetson_clocks --show. '
            'Apply jetson_clocks after selecting the power mode.',
        ),
    ):
        try:
            output = query(argv)
        except (OSError, subprocess.SubprocessError) as error:
            report.add(key, ERROR, f'Unable to read {argv[-1]} from {argv[0]}: {error} {advice}')
            continue
        report.evidence[key] = output
        report.check(
            key,
            partial(validator, output),
            advice,
            'info' if key == 'nvpmodel_output' else WARNING,
        )
    if not any(
        check['status'] == ERROR
        and check['name'] in {'online_cpus', 'nvpmodel_output', 'jetson_clocks_output'}
        for check in report.checks
    ):
        report.check(
            'mode_maximum_clocks',
            partial(check_mode_maximum_clocks, report),
            'Check /etc/nvpmodel.conf and apply jetson_clocks after selecting the power mode.',
        )


def check_mode_maximum_clocks(report: Report) -> dict[str, str]:
    """Compare all required clock readings with the selected mode's configured maxima."""
    cpus = parse_cpu_list(str(report.evidence.get('online_cpus', '')))
    if not cpus:
        raise ValueError('The online CPU list is empty.')
    mode_id, name = parse_power_mode(str(report.evidence['nvpmodel_output']))
    config = NVPMODEL_CONFIG_PATH.read_text(encoding='utf-8')
    limits = parse_mode_limits(config, mode_id, name)
    readings = observed_clock_frequencies(str(report.evidence['jetson_clocks_output']))
    report.evidence.update(
        clock_limit_mode_id=mode_id,
        clock_limit_mode_name=name,
        clock_limit_config=str(NVPMODEL_CONFIG_PATH),
        clock_limit_config_sha256=hashlib.sha256(config.encode('utf-8')).hexdigest(),
        observed_clock_frequencies_hz=readings,
    )
    expected = {}
    for label, (parameter, unit, table) in clock_targets(cpus).items():
        if parameter not in limits:
            raise ValueError(f'Mode {mode_id} ({name}) has no maximum frequency for {label}.')
        if label not in readings:
            raise OSError(f'jetson_clocks did not report {label}.')
        expected[label] = maximum_frequency(limits[parameter], unit, table)
    report.evidence['mode_clock_limits_hz'] = expected
    differences = []
    for label, maximum in expected.items():
        if any(value != maximum for value in readings[label]):
            minimum, limit, current = (value / 1_000_000 for value in readings[label])
            differences.append(
                f'{label}: observed min/max/current {minimum:g}/{limit:g}/{current:g} MHz; '
                f'expected {maximum / 1_000_000:g} MHz for mode {mode_id} ({name}).'
            )
    if differences:
        raise RuntimeError(' '.join(differences))
    return {'mode_maximum_clocks': 'pass'}


def collect_report() -> dict[str, Any]:
    """Inspect the reusable Thor profile, with advisory mismatches and explicit read failures."""
    report = Report()
    try:
        model = Path('/proc/device-tree/model').read_text(encoding='utf-8').strip('\0\n')
        report.evidence['device_model'] = model
        if 'thor' not in model.lower():
            raise ValueError(f'The Thor reference profile does not cover {model}.')
    except (OSError, ValueError) as error:
        report.add('platform', ERROR, f'{error} This checker currently supports AGX Thor only.')
        return report.document()
    report.check(
        'kernel',
        lambda: check_kernel(report),
        'Select the PREEMPT_RT boot entry for reference comparisons.',
    )
    inspect_cpu_layout(report)
    try:
        inspect_boot_settings(report)
    except (OSError, ValueError) as error:
        report.add('boot_settings', ERROR, f'Unable to read boot settings: {error}')
    report.compare(
        'sched_rt_runtime_us',
        lambda: str(int(RT_RUNTIME_PATH.read_text(encoding='utf-8').strip())),
        '-1',
        WARNING,
        'A finite RT budget can periodically throttle real-time workers. '
        'See Real-time CPU budget in the host guide to set kernel.sched_rt_runtime_us=-1.',
    )
    report.compare(
        'display_manager',
        desktop_state,
        'inactive',
        WARNING,
        'A desktop can add CPU/GPU work. Follow Run without a desktop '
        'if your application does not need a display.',
    )
    report.check(
        'idle', lambda: check_idle(report), 'Apply jetson_clocks; see Power mode and clocks.'
    )
    inspect_power_and_clocks(report)
    return report.document()


def check_kernel(report: Report) -> dict[str, object]:
    """Keep kernel evidence even when PREEMPT_RT cannot be confirmed."""
    evidence = kernel_evidence(require_rt=False)
    report.evidence.update(evidence)
    if evidence['preempt_rt'] != 'pass':
        raise RuntimeError('PREEMPT_RT is not confirmed; scheduling latency may differ.')
    return evidence


def check_idle(report: Report) -> dict[str, object]:
    """Save per-state evidence before reporting enabled idle states."""
    evidence = idle_evidence()
    report.evidence.update(evidence)
    if not evidence['deep_idle_disabled']:
        raise RuntimeError('Enabled CPU idle states can add wakeup latency.')
    return evidence


CHECK_LABELS = {
    'platform': 'Platform',
    'kernel': 'Kernel',
    'online_cpus': 'Online CPUs',
    'isolated_cpus': 'Isolated CPUs',
    'default_irq_cpus': 'Default IRQ CPUs',
    'unbound_workqueue_cpus': 'Unbound workqueue CPUs',
    'requested_isolcpus': 'Boot CPU isolation',
    'requested_irqaffinity': 'Boot IRQ affinity',
    'requested_kthread_cpus': 'Boot kernel thread CPUs',
    'isolation_flags': 'CPU isolation flags',
    'tickless_rcu': 'Additional tuning',
    'boot_settings': 'Boot settings',
    'sched_rt_runtime_us': 'Real-time CPU budget',
    'display_manager': 'Display manager',
    'idle': 'CPU idle states',
    'nvpmodel_output': 'Power mode',
    'jetson_clocks_output': 'Fixed clocks',
    'mode_maximum_clocks': 'Peak clock rates',
}
BOOT_CHECKS = {'requested_isolcpus', 'requested_irqaffinity', 'requested_kthread_cpus'}


def compact_cpu_list(value: str) -> str:
    """Display contiguous CPUs as ranges while preserving gaps."""
    ranges: list[list[int]] = []
    cpus = sorted(parse_cpu_list(value))
    for cpu in cpus:
        if ranges and cpu == ranges[-1][-1] + 1:
            ranges[-1][-1] = cpu
        else:
            ranges.append([cpu, cpu])
    return (
        ','.join(str(start) if start == end else f'{start}–{end}' for start, end in ranges)
        or 'None'
    )


def clock_rates_summary(report: dict[str, Any]) -> str:
    """Show verified clock rates, combining CPUs that share a frequency."""
    limits = report['mode_clock_limits_hz']
    cpu_rates = sorted({value for name, value in limits.items() if name.startswith('cpu')})
    cpu_text = '/'.join(f'{value / 1_000_000:g}' for value in cpu_rates)
    parts = [f'CPU {cpu_text} MHz']
    for name, label in (('gpu-gpc-0', 'GPU'), ('gpu-nvd-0', 'Video'), ('EMC', 'Memory')):
        parts.append(f'{label} {limits[name] / 1_000_000:g} MHz')
    return ' · '.join(parts)


def check_summary(check: dict[str, str], report: dict[str, Any]) -> str:
    """Show concise observations without claiming success for unavailable readings."""
    name, status = check['name'], check['status']
    if status == ERROR:
        return 'Unable to inspect'
    if name.endswith('_cpus') and name in report:
        return compact_cpu_list(str(report[name]))
    if name == 'display_manager' and name in report:
        return str(report[name]).capitalize()
    if status != 'pass':
        return 'See findings'
    if name == 'mode_maximum_clocks':
        return clock_rates_summary(report)
    return {
        'kernel': 'PREEMPT_RT',
        'idle': 'Disabled',
        'sched_rt_runtime_us': 'Unlimited (-1)',
        'nvpmodel_output': '120 W · mode 1',
        'jetson_clocks_output': 'Verified',
    }.get(name, 'Matches reference')


def print_check_table(
    report: dict[str, Any], marker: Callable[[str], str], *, verbose: bool
) -> None:
    """Align observations and group successful boot checks in the compact view."""
    checks = report['checks']
    grouped_boot = (
        not verbose
        and BOOT_CHECKS <= {c['name'] for c in checks if c['status'] == 'pass'}
        and not any(c['name'] == 'isolation_flags' for c in checks)
    )
    rows = []
    for check in checks:
        name = check['name']
        if name == 'tickless_rcu':
            continue
        if grouped_boot and name in BOOT_CHECKS:
            if name == 'requested_isolcpus':
                rows.append(('Boot CPU settings', 'pass', 'Match reference'))
            continue
        rows.append((CHECK_LABELS[name], check['status'], check_summary(check, report)))
    width = max((len(label) for label, _, _ in rows), default=0)
    for label, row_status, summary in rows:
        print(f'  {label:<{width}}  {marker(row_status)} {summary}')


def print_readings(report: dict[str, Any]) -> None:
    """Display available raw observations for the verbose report."""
    for key, label in (
        ('kernel_release', 'Running kernel'),
        ('kernel_cmdline', 'Kernel boot arguments'),
        ('nvpmodel_output', 'Power readings (nvpmodel -q)'),
        ('jetson_clocks_output', 'Clock readings (jetson_clocks --show)'),
        ('idle_states', 'CPU idle disable controls'),
        ('mode_clock_limits_hz', 'Selected-mode clock limits (Hz)'),
    ):
        if key not in report:
            continue
        print(f'\n{label}:')
        value = report[key]
        if isinstance(value, dict):
            value = '\n'.join(f'{path}: {observed}' for path, observed in value.items())
        print(textwrap.indent(str(value).rstrip(), '  '))


def print_report(report: dict[str, Any], *, verbose: bool = False) -> None:
    """Lead with inspection results, then show actionable findings and optional evidence."""
    color = (
        sys.stdout.isatty() and os.environ.get('TERM') != 'dumb' and 'NO_COLOR' not in os.environ
    )

    def marker(status: str) -> str:
        symbol, code = {
            'pass': ('✓', '32'),
            'info': ('ℹ', '36'),
            WARNING: ('!', '33'),
            ERROR: ('✗', '31'),
        }[status]
        return f'\033[{code}m{symbol}\033[0m' if color else symbol

    checks = report['checks']
    counts = {
        status: sum(c['status'] == status for c in checks)
        for status in ('pass', 'info', WARNING, ERROR)
    }
    state = 'complete' if report['inspection_complete'] else 'incomplete'
    if counts[ERROR]:
        status = ERROR
    elif counts[WARNING]:
        status = WARNING
    else:
        status = 'pass'
    passed, informational = counts['pass'], counts['info']
    print('Host preflight · AGX Thor')
    print()
    print(
        f'{marker(status)} Inspection {state}   {passed} passed · '
        f'{informational} informational · {counts[WARNING]} warnings · {counts[ERROR]} failed'
    )
    print()
    print_check_table(report, marker, verbose=verbose)

    for check in checks:
        if check['status'] == 'pass' and not verbose:
            continue
        status, label = check['status'], CHECK_LABELS[check['name']]
        print()
        print(f'{marker(status)} {label}')
        print(
            textwrap.fill(check['message'], width=88, initial_indent='  ', subsequent_indent='  ')
        )

    if verbose:
        print_readings(report)
    print()
    if not report.get('reference_frequencies_verified') and report.get('fixed_clocks') == 'pass':
        print('Clocks are fixed; selected-mode maximum frequencies were not verified.')
    elif report.get('fixed_clocks') != 'pass':
        print('Fixed frequencies were not verified; see the clock inspection findings.')
    print(f'Setup guide: {GUIDE}')


def main(argv: list[str] | None = None) -> int:
    """Return 1 for incomplete inspection, 2 for warnings, and 0 otherwise."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='Exit codes: 0 = clean or informational findings; 1 = incomplete inspection; '
        '2 = completed with warnings.',
    )
    parser.add_argument('--format', choices=('text', 'json'), default='text')
    parser.add_argument(
        '--verbose', action='store_true', help='Show individual checks and readings.'
    )
    args = parser.parse_args(argv)
    report = collect_report()
    if args.format == 'json':
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report, verbose=args.verbose)
    if not report['inspection_complete']:
        return 1
    if any(check['status'] == WARNING for check in report['checks']):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
