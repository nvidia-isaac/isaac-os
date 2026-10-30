#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared host checks independently of the machine running the tests."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# pylint: disable=wrong-import-position
from tools import host_preflight as host


class HostPreflightTest(unittest.TestCase):
    """Ensure the documented reference passes and deviations are classified honestly."""

    def setUp(self):
        """Provide a Thor configured exactly as the guide's reference example."""
        self.values = {
            Path('/proc/device-tree/model'): 'NVIDIA Jetson AGX Thor Developer Kit',
            host.ONLINE_CPU_PATH: '0-13',
            host.RT_RUNTIME_PATH: '-1\n',
            host.ISOLATED_CPU_PATH: '8-13',
            host.DEFAULT_IRQ_AFFINITY_PATH: 'ff',
            host.WORKQUEUE_CPU_MASK_PATH: 'ff',
            host.CMDLINE_PATH: 'isolcpus=domain,managed_irq,8-13 irqaffinity=0-7 kthread_cpus=0-7',
            Path('/sys/kernel/realtime'): '1',
        }
        self.idle = '1'
        limits = ''.join(f'CPU_AE_{cpu} MAX_FREQ 10\n' for cpu in range(14))
        limits += 'GPU MAX_FREQ 20\nVIDEO MAX_FREQ 25\nEMC MAX_FREQ 30\n'
        self.values[host.NVPMODEL_CONFIG_PATH] = (
            f'< POWER_MODEL ID=0 NAME=MAXN >\n{limits}'
            f'< POWER_MODEL ID=1 NAME=120W >\n{limits}'
            '< PM_CONFIG DEFAULT=1 >\n'
        )
        self.power = 'NV Power Mode: 120W\n1\n'
        self.desktop = 'LoadState=loaded\nActiveState=inactive\n'
        self.clocks = ''.join(
            f'cpu{cpu}: Governor=performance MinFreq=10 MaxFreq=10 CurrentFreq=10\n'
            for cpu in range(14)
        )
        self.clocks += (
            'gpu-gpc-0 MinFreq=20 MaxFreq=20 CurrentFreq=20\n'
            'gpu-nvd-0 MinFreq=25 MaxFreq=25 CurrentFreq=25\n'
            'EMC MinFreq=30 MaxFreq=30 CurrentFreq=30\n'
        )
        for patch in (
            mock.patch.object(host.gzip, 'open', side_effect=FileNotFoundError),
            mock.patch.object(Path, 'read_text', autospec=True, side_effect=self.read_text),
            mock.patch.object(
                Path,
                'glob',
                autospec=True,
                side_effect=lambda p, _pattern: [p / 'state1' / 'disable'],
            ),
            mock.patch.object(host.subprocess, 'run', side_effect=self.run_query),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def read_text(self, path, **_kwargs):
        """Return observations or raise the requested read error."""
        if path.name == 'disable':
            return self.idle
        value = self.values.get(path, FileNotFoundError(str(path)))
        if isinstance(value, Exception):
            raise value
        return value

    def run_query(self, argv, **_kwargs):
        """Return realistic command status and stdout without executing host tools."""
        values = {'systemctl': self.desktop, 'nvpmodel': self.power, 'sudo': self.clocks}
        value = values[argv[0]]
        if isinstance(value, Exception):
            raise value
        return subprocess.CompletedProcess(argv, 0, value, '')

    def test_reference_passes_without_tickless_or_pm_qos(self):
        """Following the guide is sufficient without undocumented tuning."""
        report = host.collect_report()
        self.assertTrue(report['inspection_complete'])
        self.assertTrue(report['reference_match'], report['checks'])
        self.assertIsNone(report['nohz_full_cpus'])
        self.assertEqual(report['jetson_clocks_output'], self.clocks)
        self.assertEqual(report['nvpmodel_output'], self.power)
        self.assertEqual(report['efi_runtime_observed'], 'unknown')

    def test_finite_rt_budget_warns_without_failing_inspection(self):
        """Retain the effective budget and explain periodic RT throttling."""
        self.values[host.RT_RUNTIME_PATH] = '950000\n'
        report = host.collect_report()
        self.assertTrue(report['inspection_complete'])
        self.assertFalse(report['reference_match'])
        self.assertEqual(report['sched_rt_runtime_us'], '950000')
        check = next(c for c in report['checks'] if c['name'] == 'sched_rt_runtime_us')
        self.assertEqual(check['status'], host.WARNING)
        self.assertIn('kernel.sched_rt_runtime_us=-1', check['message'])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(host.main([]), 2)

    def test_unavailable_rt_budget_is_an_inspection_error(self):
        """Missing or malformed readings cannot pass reference inspection."""
        for value in (
            FileNotFoundError('missing sysctl'),
            PermissionError('denied'),
            '',
            'invalid',
        ):
            with self.subTest(value=value):
                self.values[host.RT_RUNTIME_PATH] = value
                report = host.collect_report()
                self.assertFalse(report['inspection_complete'])
                check = next(c for c in report['checks'] if c['name'] == 'sched_rt_runtime_us')
                self.assertEqual(check['status'], host.ERROR)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(host.main([]), 1)

    def test_report_identifies_scope_and_revision(self):
        """Clock verification records the selected configuration and normalized limits."""
        report = host.collect_report()
        self.assertTrue(report['reference_match'])
        self.assertTrue(report['reference_frequencies_verified'])
        self.assertEqual(report['mode_clock_limits_hz']['cpu0'], 10000)
        self.assertEqual(report['mode_clock_limits_hz']['gpu-gpc-0'], 20)
        self.assertEqual(report['mode_clock_limits_hz']['gpu-nvd-0'], 25)
        self.assertEqual(report['clock_limit_mode_id'], '1')
        self.assertEqual(len(report['clock_limit_config_sha256']), 64)
        self.assertEqual(report['reference_revision'], host.REFERENCE_REVISION)
        self.assertEqual(len(report['checker_sha256']), 64)
        self.assertTrue(report['captured_at_utc'].endswith('+00:00'))
        self.assertIn('requested_kthread_cpus', report)
        with mock.patch('builtins.print') as output:
            host.print_report(report)
        self.assertIn('Peak clock rates', str(output.call_args_list))
        self.assertIn(
            'CPU 0.01 MHz · GPU 2e-05 MHz · Video 2.5e-05 MHz · Memory 3e-05 MHz',
            str(output.call_args_list),
        )
        self.assertNotIn('Clocks fixed at the selected power mode', str(output.call_args_list))

    def test_desktop_and_idle_warn_without_aborting(self):
        """An active desktop and idle states remain actionable performance warnings."""
        self.desktop = 'LoadState=loaded\nActiveState=active\n'
        self.idle = '0'
        report = host.collect_report()
        self.assertTrue(report['inspection_complete'])
        self.assertFalse(report['reference_match'])
        self.assertEqual(
            {c['name'] for c in report['checks'] if c['status'] == 'warning'},
            {'idle', 'display_manager'},
        )
        self.assertIn('nvpmodel_output', report)

    def test_different_power_is_reference_information(self):
        """Another valid power budget is recorded, not silently normalized to 120W."""
        self.power = 'NV Power Mode: MAXN\n0\n'
        report = host.collect_report()
        self.assertTrue(report['inspection_complete'])
        self.assertEqual(report['nvpmodel_output'], self.power)
        self.assertTrue(
            any(c['status'] == 'info' and c['name'] == 'nvpmodel_output' for c in report['checks'])
        )

    def test_clock_mismatch_preserves_actual_frequencies(self):
        """A changing clock is a warning with raw evidence available for comparison."""
        self.clocks = self.clocks.replace('CurrentFreq=20', 'CurrentFreq=19')
        report = host.collect_report()
        self.assertTrue(report['inspection_complete'])
        self.assertFalse(report['reference_match'])
        self.assertIn('CurrentFreq=19', report['jetson_clocks_output'])

    def test_fixed_below_mode_maximum_warns_for_each_clock_domain(self):
        """Equal min/max/current readings cannot hide a clock locked below its mode limit."""
        original = self.clocks
        for label, old, lower in (
            ('cpu0:', 10, 9),
            ('gpu-gpc-0', 20, 19),
            ('gpu-nvd-0', 25, 24),
            ('EMC', 30, 29),
        ):
            with self.subTest(clock=label):
                self.clocks = '\n'.join(
                    line.replace(f'Freq={old}', f'Freq={lower}') if line.startswith(label) else line
                    for line in original.splitlines()
                )
                report = host.collect_report()
                self.assertTrue(report['inspection_complete'])
                self.assertEqual(report['fixed_clocks'], 'pass')
                self.assertFalse(report['reference_frequencies_verified'])
                finding = next(c for c in report['checks'] if c['name'] == 'mode_maximum_clocks')
                self.assertEqual(finding['status'], 'warning')
                self.assertIn('observed min/max/current', finding['message'])
                self.assertIn('expected', finding['message'])
                self.assertIn('mode 1 (120W)', finding['message'])
                with mock.patch('builtins.print'):
                    self.assertEqual(host.main([]), 2)

    def test_only_selected_mode_limits_are_used(self):
        """A different mode's limits and commented-out entries do not affect verification."""
        config = self.values[host.NVPMODEL_CONFIG_PATH]
        self.values[host.NVPMODEL_CONFIG_PATH] = config.replace(
            'GPU MAX_FREQ 20', 'GPU MAX_FREQ 99', 1
        ).replace('GPU MAX_FREQ 20', '# GPU MAX_FREQ 99\nGPU MAX_FREQ 20 # selected limit')
        self.assertTrue(host.collect_report()['reference_frequencies_verified'])
        self.power = 'NV Power Mode: MAXN\n0\n'
        report = host.collect_report()
        self.assertEqual(report['mode_clock_limits_hz']['gpu-gpc-0'], 99)
        self.assertFalse(report['reference_frequencies_verified'])
        self.assertEqual(report['clock_limit_mode_id'], '0')

    def test_maxn_unlimited_limits_use_supported_frequencies(self):
        """MAXN's -1 sentinel resolves to available rates rather than a mutable runtime cap."""
        self.power = 'NV Power Mode: MAXN\n0\n'
        for limit in (10, 20, 25, 30):
            self.values[host.NVPMODEL_CONFIG_PATH] = self.values[host.NVPMODEL_CONFIG_PATH].replace(
                f'MAX_FREQ {limit}', 'MAX_FREQ -1'
            )
        for cpu in range(14):
            self.values[
                Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_available_frequencies')
            ] = '5 10'
        for device, rates in (('gpu-gpc-0', '10 20'), ('gpu-nvd-0', '15 25'), ('bwmgr', '20 30')):
            self.values[Path(f'/sys/class/devfreq/{device}/available_frequencies')] = rates
        report = host.collect_report()
        self.assertTrue(report['reference_frequencies_verified'])
        self.assertEqual(report['mode_clock_limits_hz']['cpu0'], 10000)
        self.assertEqual(report['mode_clock_limits_hz']['gpu-nvd-0'], 25)
        with mock.patch('builtins.print'):
            self.assertEqual(host.main([]), 0)

    def test_unavailable_or_invalid_limits_fail_inspection(self):
        """Missing, ambiguous, and malformed limits cannot produce a positive verdict."""
        original = self.values[host.NVPMODEL_CONFIG_PATH]
        for config in (
            FileNotFoundError('missing power configuration'),
            original.replace('ID=1 NAME=120W', 'ID=2 NAME=120W'),
            original.replace('NAME=120W', 'NAME=custom'),
            original + original,
            original.replace('GPU MAX_FREQ 20', ''),
            original.replace('GPU MAX_FREQ 20', 'GPU MAX_FREQ invalid'),
            original.replace('GPU MAX_FREQ 20', 'GPU MAX_FREQ 0'),
            original.replace('GPU MAX_FREQ 20', 'GPU MAX_FREQ -2'),
            original.replace('GPU MAX_FREQ 20', 'GPU MAX_FREQ 20\nGPU MAX_FREQ 21'),
            original.replace('GPU MAX_FREQ 20', 'GPU MAX_FREQ'),
        ):
            with self.subTest(config=str(config)[:80]):
                self.values[host.NVPMODEL_CONFIG_PATH] = config
                report = host.collect_report()
                self.assertFalse(report['inspection_complete'])
                self.assertFalse(report['reference_frequencies_verified'])
                with mock.patch('builtins.print'):
                    self.assertEqual(host.main([]), 1)

    def test_unlimited_limit_requires_valid_supported_frequency_table(self):
        """An unreadable or empty frequency table must not be replaced with the runtime cap."""
        self.values[host.NVPMODEL_CONFIG_PATH] = self.values[host.NVPMODEL_CONFIG_PATH].replace(
            'GPU MAX_FREQ 20', 'GPU MAX_FREQ -1'
        )
        table = Path('/sys/class/devfreq/gpu-gpc-0/available_frequencies')
        for value in (FileNotFoundError('missing table'), '', 'invalid', '0 20'):
            with self.subTest(table=value):
                self.values[table] = value
                report = host.collect_report()
                self.assertFalse(report['inspection_complete'])
                self.assertFalse(report['reference_frequencies_verified'])

    def test_missing_video_reading_takes_precedence_over_low_cpu_clock(self):
        """All required domains must be readable before a mismatch is only a warning."""
        self.clocks = self.clocks.replace('gpu-nvd-0 MinFreq=25 MaxFreq=25 CurrentFreq=25\n', '')
        self.clocks = self.clocks.replace('CurrentFreq=10', 'CurrentFreq=9', 1)
        report = host.collect_report()
        self.assertFalse(report['inspection_complete'])
        self.assertFalse(report['reference_frequencies_verified'])
        finding = next(c for c in report['checks'] if c['name'] == 'mode_maximum_clocks')
        self.assertEqual(finding['status'], 'error')
        self.assertIn('gpu-nvd-0', finding['message'])
        with mock.patch('builtins.print'):
            self.assertEqual(host.main([]), 1)

    def test_missing_clock_rows_are_unknown(self):
        """Successful execution with incomplete clock output is not a performance warning."""
        self.clocks = 'No clocks reported'
        self.assertFalse(host.collect_report()['inspection_complete'])

    def test_early_clock_difference_cannot_hide_missing_readings(self):
        """Missing later readings must fail inspection even if CPU 0 is unlocked."""
        original = self.clocks.replace(
            'cpu0: Governor=performance MinFreq=10 MaxFreq=10 CurrentFreq=10',
            'cpu0: Governor=performance MinFreq=10 MaxFreq=10 CurrentFreq=9',
        )
        for missing in ('cpu13:', 'gpu-', 'EMC'):
            with self.subTest(missing=missing):
                self.clocks = '\n'.join(
                    line for line in original.splitlines() if not line.startswith(missing)
                )
                report = host.collect_report()
                self.assertFalse(report['inspection_complete'])
                finding = next(c for c in report['checks'] if c['name'] == 'jetson_clocks_output')
                self.assertEqual(finding['status'], 'error')
                self.assertEqual(report['jetson_clocks_output'], self.clocks)
                with mock.patch('builtins.print'):
                    self.assertEqual(host.main([]), 1)

    def test_empty_power_output_is_unknown(self):
        """A malformed power query cannot be interpreted as another valid power budget."""
        self.power = ''
        self.assertFalse(host.collect_report()['inspection_complete'])

    def test_offline_cpu_is_only_a_reference_difference(self):
        """Assess clocks and idle on the online CPUs instead of demanding missing controls."""
        self.values[host.ONLINE_CPU_PATH] = '0-12'
        self.clocks = self.clocks.replace(
            'cpu13: Governor=performance MinFreq=10 MaxFreq=10 CurrentFreq=10\n', ''
        )
        report = host.collect_report()
        self.assertTrue(report['inspection_complete'], report['checks'])
        self.assertFalse(report['reference_match'])

    def test_sudo_failure_is_incomplete_not_a_tuning_warning(self):
        """Unavailable observations must not claim a successful inspection."""
        self.clocks = OSError('sudo: a password is required')
        report = host.collect_report()
        self.assertFalse(report['inspection_complete'])
        error = next(c for c in report['checks'] if c['name'] == 'jetson_clocks_output')
        self.assertEqual(error['status'], 'error')
        self.assertIn('sudo -v', error['message'])
        self.assertIn('password is required', error['message'])

    def test_failed_desktop_query_is_unknown(self):
        """A broken systemd query is not evidence of an inactive desktop."""
        self.desktop = OSError('Failed to connect to bus')
        self.assertFalse(host.collect_report()['inspection_complete'])

    def test_missing_display_manager_is_inactive(self):
        """A host without a display-manager unit needs no service shutdown."""
        self.desktop = 'LoadState=not-found\nActiveState=inactive\n'
        self.assertTrue(host.collect_report()['reference_match'])

    def test_optional_tickless_is_information(self):
        """Optional tuning is retained without becoming a requirement."""
        self.values[host.CMDLINE_PATH] += ' nohz_full=12-13 rcu_nocbs=8-13'
        self.values[host.NOHZ_FULL_PATH] = '12-13'
        report = host.collect_report()
        self.assertTrue(report['inspection_complete'])
        self.assertEqual(report['nohz_full_cpus'], '12,13')
        self.assertEqual(report['rcu_nocbs_cpus'], '8,9,10,11,12,13')
        self.assertTrue(
            any(c['name'] == 'tickless_rcu' and c['status'] == 'info' for c in report['checks'])
        )

    def test_null_tickless_is_not_a_failure(self):
        """The kernel's null mask is acceptable for the stock RT profile."""
        self.values[host.NOHZ_FULL_PATH] = '(null)'
        self.assertTrue(host.collect_report()['reference_match'])

    def test_tickless_message_uses_requested_kernel_build_options(self):
        """Optional tuning stays informational and reports only the missing required options."""
        original = self.values[host.CMDLINE_PATH]
        for arguments, config, missing in (
            ('nohz_full=8-13 rcu_nocbs=8-13', 'CONFIG_NO_HZ_FULL=y\nCONFIG_RCU_NOCB_CPU=y', ''),
            (
                'nohz_full=8-13 rcu_nocbs=8-13',
                '# CONFIG_NO_HZ_FULL is not set',
                'CONFIG_NO_HZ_FULL, CONFIG_RCU_NOCB_CPU',
            ),
            ('nohz_full=8-13 rcu_nocbs=8-13', 'CONFIG_NO_HZ_FULL=y', 'CONFIG_RCU_NOCB_CPU'),
            ('nohz_full=8-13', 'CONFIG_NO_HZ_FULL=y', ''),
            ('rcu_nocbs=8-13', 'CONFIG_RCU_NOCB_CPU=y', ''),
            ('rcu_nocb_poll', 'CONFIG_PREEMPT_RT=y', 'CONFIG_RCU_NOCB_CPU'),
        ):
            with self.subTest(arguments=arguments, config=config):
                self.values[host.CMDLINE_PATH] = f'{original} {arguments}'
                self.values[Path('/boot') / f'config-{host.platform.release()}'] = config
                report = host.collect_report()
                finding = next(c for c in report['checks'] if c['name'] == 'tickless_rcu')
                self.assertEqual(finding['status'], 'info')
                if missing:
                    self.assertIn(f'this kernel lacks {missing}.', finding['message'])
                    self.assertIn('Remove the unsupported boot options', finding['message'])
                else:
                    self.assertEqual(
                        finding['message'],
                        'Tickless/RCU tuning is configured; '
                        'this differs from the reference configuration.',
                    )
                with mock.patch('builtins.print'):
                    self.assertEqual(host.main([]), 0)

    def test_tickless_support_prefers_running_kernel_config(self):
        """The embedded running-kernel configuration takes precedence over the boot file."""
        self.values[host.CMDLINE_PATH] += ' nohz_full=8-13 rcu_nocb_poll'
        self.values[Path('/boot') / f'config-{host.platform.release()}'] = (
            '# CONFIG_NO_HZ_FULL is not set\n# CONFIG_RCU_NOCB_CPU is not set'
        )
        with mock.patch.object(
            host.gzip,
            'open',
            return_value=io.StringIO('CONFIG_NO_HZ_FULL=y\nCONFIG_RCU_NOCB_CPU=y'),
        ):
            report = host.collect_report()
        self.assertEqual(report['kernel_config_source'], '/proc/config.gz')
        finding = next(c for c in report['checks'] if c['name'] == 'tickless_rcu')
        self.assertIn('differs from the reference configuration', finding['message'])

    def test_unused_tickless_support_does_not_add_a_finding(self):
        """Kernel support alone is not a requested tuning difference."""
        self.values[Path('/boot') / f'config-{host.platform.release()}'] = (
            'CONFIG_NO_HZ_FULL=y\nCONFIG_RCU_NOCB_CPU=y'
        )
        report = host.collect_report()
        self.assertFalse(any(c['name'] == 'tickless_rcu' for c in report['checks']))

    def test_unreadable_optional_evidence_is_not_absent(self):
        """An access error differs from an unsupported kernel feature."""
        self.values[host.NOHZ_FULL_PATH] = PermissionError('denied')
        self.assertFalse(host.collect_report()['inspection_complete'])

    def test_missing_rt_warns_but_benchmark_can_require_it(self):
        """The generic tool advises; applications can require observed RT evidence."""
        self.values[Path('/sys/kernel/realtime')] = '0'
        report = host.collect_report()
        self.assertTrue(report['inspection_complete'])
        self.assertEqual(report['realtime_sysfs'], '0')
        with self.assertRaisesRegex(RuntimeError, 'not PREEMPT_RT'):
            host.kernel_evidence()

    def test_isolation_mismatch_does_not_hide_other_findings(self):
        """One tuning mismatch must not prevent the remaining settings being inspected."""
        self.values[host.ISOLATED_CPU_PATH] = ''
        self.idle = '0'
        report = host.collect_report()
        self.assertTrue(report['inspection_complete'])
        warnings = {c['name'] for c in report['checks'] if c['status'] == 'warning'}
        self.assertLessEqual({'isolated_cpus', 'idle'}, warnings)

    def test_unsupported_machine_does_not_pass(self):
        """The Thor reference must not be silently applied to a different board."""
        self.values[Path('/proc/device-tree/model')] = 'Another board'
        self.assertFalse(host.collect_report()['inspection_complete'])

    def test_failed_clock_query_does_not_claim_fixed_frequencies(self):
        """Missing observations must not produce a positive clock summary."""
        self.clocks = OSError('sudo: a password is required')
        with mock.patch('builtins.print') as output:
            host.print_report(host.collect_report())
        output.assert_any_call(
            'Fixed frequencies were not verified; see the clock inspection findings.'
        )

    def test_cli_status_and_json(self):
        """Warnings exit 2, unavailable evidence exits 1, and JSON remains parsable."""
        self.idle = '0'
        with mock.patch('builtins.print') as output:
            self.assertEqual(host.main(['--format', 'json']), 2)
            self.assertIn('"inspection_complete": true', output.call_args.args[0])
        self.clocks = OSError('missing')
        with mock.patch('builtins.print'):
            self.assertEqual(host.main([]), 1)

    def test_reference_and_informational_findings_exit_zero(self):
        """Optional reference differences do not cause a nonzero exit status."""
        for extra_tuning in ('', ' nohz_full=8-13'):
            self.values[host.CMDLINE_PATH] += extra_tuning
            for output_format in ('text', 'json'):
                with self.subTest(tuning=extra_tuning, output_format=output_format):
                    with mock.patch('builtins.print'):
                        self.assertEqual(host.main(['--format', output_format]), 0)

    def test_warnings_exit_two_and_preserve_diagnostics(self):
        """Warnings cause a nonzero exit status while retaining diagnostic output."""
        self.idle = '0'
        for output_format in ('text', 'json'):
            with self.subTest(output_format=output_format):
                with mock.patch('builtins.print') as output:
                    self.assertEqual(host.main(['--format', output_format]), 2)
                    self.assertTrue(output.called)
                    self.assertIn('Enabled CPU idle states', str(output.call_args_list))

    def test_inspection_error_takes_precedence_over_warnings(self):
        """Incomplete inspection exits 1 even when it also finds warnings."""
        self.clocks = OSError('sudo: a password is required')
        for idle in ('1', '0'):
            self.idle = idle
            for output_format in ('text', 'json'):
                with self.subTest(idle=idle, output_format=output_format):
                    with mock.patch('builtins.print'):
                        self.assertEqual(host.main(['--format', output_format]), 1)

    def render(self, *, verbose=False):
        """Capture the text interface without terminal escape sequences."""
        output = io.StringIO()
        with redirect_stdout(output):
            host.print_report(host.collect_report(), verbose=verbose)
        return output.getvalue()

    def test_compact_report_keeps_extra_tuning_informational(self):
        """Extra tuning does not turn a completed inspection into a failed headline."""
        self.values[host.CMDLINE_PATH] += ' nohz_full=8-13'
        output = self.render()
        self.assertIn(
            '✓ Inspection complete   14 passed · 1 informational · 0 warnings · 0 failed', output
        )
        self.assertIn('✓ 0–13', output)
        self.assertIn('✓ 8–13', output)
        self.assertIn('Boot CPU settings', output)
        self.assertIn('ℹ Additional tuning', output)
        self.assertIn('running kernel configuration is unavailable', ' '.join(output.split()))
        self.assertNotIn('Boot arguments alone', output)
        self.assertNotIn('settings match: no', output)
        self.assertNotIn('\033[', output)

    def test_warning_report_preserves_observations_and_advice(self):
        """Warnings remain visible without losing the other successful checks."""
        self.desktop = 'LoadState=loaded\nActiveState=active\n'
        self.values[host.CMDLINE_PATH] = 'isolcpus=8-13 irqaffinity=0-7 kthread_cpus=0-6'
        output = self.render()
        self.assertIn('! Inspection complete', output)
        self.assertIn('! Active', output)
        self.assertIn('A desktop can add CPU/GPU work.', output)
        self.assertNotIn('Boot CPU settings', output)
        self.assertIn('Boot kernel thread CPUs', output)
        self.assertIn('0–6', output)
        self.assertIn('CPU isolation flags', output)

    def test_incomplete_report_preserves_error_and_recovery(self):
        """A failed read cannot appear as verified clocks or a successful inspection."""
        self.clocks = OSError('sudo: a password is required')
        output = self.render()
        self.assertIn('✗ Inspection incomplete', output)
        self.assertIn('1 failed', output)
        self.assertIn('✗ Unable to inspect', output)
        self.assertIn('sudo: a password is required', output)
        self.assertIn('sudo -v', output)
        self.assertNotIn('✓ Verified', output)

    def test_verbose_cli_includes_individual_checks_and_readings(self):
        """Detailed output exposes boot checks and original diagnostic evidence."""
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(host.main(['--verbose']), 0)
        self.assertIn('Boot CPU isolation', output.getvalue())
        self.assertNotIn('Boot CPU settings', output.getvalue())
        self.assertIn(self.values[host.CMDLINE_PATH], output.getvalue())
        self.assertIn('gpu-gpc-0 MinFreq=20 MaxFreq=20 CurrentFreq=20', output.getvalue())
        self.assertIn('CPU idle disable controls:', output.getvalue())

    def test_color_respects_terminal_and_opt_out(self):
        """Interactive terminals get color unless NO_COLOR or TERM=dumb disables it."""
        for environment, expected in (
            ({'TERM': 'xterm'}, True),
            ({'NO_COLOR': ''}, False),
            ({'TERM': 'dumb'}, False),
        ):
            with self.subTest(environment=environment):
                output = io.StringIO()
                with (
                    mock.patch.dict(host.os.environ, environment, clear=True),
                    redirect_stdout(output),
                    mock.patch.object(output, 'isatty', return_value=True),
                ):
                    host.print_report(host.collect_report())
                self.assertEqual('\033[' in output.getvalue(), expected)

    def test_unsupported_platform_renders_only_available_results(self):
        """An early platform error does not fabricate successful Thor observations."""
        self.values[Path('/proc/device-tree/model')] = 'Another board'
        output = self.render()
        self.assertIn('0 passed · 0 informational · 0 warnings · 1 failed', output)
        self.assertIn('Another board', output)
        self.assertNotIn('Boot CPU settings', output)

    def test_cpu_ranges_preserve_gaps(self):
        """Sparse CPU masks must not look like contiguous ranges."""
        self.assertEqual(host.compact_cpu_list('0,1,2,5,8,9'), '0–2,5,8–9')
        self.assertEqual(host.compact_cpu_list(''), 'None')


if __name__ == '__main__':
    unittest.main()
