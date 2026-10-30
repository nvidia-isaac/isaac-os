# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the data_flow_tracking Nsight Systems report helpers."""

# pylint: disable=protected-access

import sqlite3
import struct
import sys
import types

import pytest


class _StatsReport:
    """Minimal stand-in for Nsight Systems' nsysstats.StatsReport."""

    def start_query(self):
        return None


nsysstats_module = types.ModuleType('nsysstats')
nsysstats_module.StatsReport = _StatsReport  # type: ignore[attr-defined]
sys.modules.setdefault('nsysstats', nsysstats_module)

# pylint: disable=wrong-import-position
from tools.profiling import data_flow_tracking as dft  # noqa: E402

# pylint: enable=wrong-import-position


def _compute(op, start, end):
    return (f'{op}{dft.COMPUTE_SUFFIX}', start, end, None)


def _emit(port_label, start, sequence):
    return (f'{port_label}{dft.EMIT_SUFFIX}', start, None, (port_label, sequence))


def _receive(receive_port_label, start, source_port_label, sequence):
    return (
        f'{receive_port_label}{dft.RECEIVE_SUFFIX}',
        start,
        None,
        (source_port_label, sequence),
    )


def _series_for(events):
    adj, sources, sinks, all_ops = dft.build_topology(events)
    compute_idx = dft.build_range_index(events, dft.COMPUTE_SUFFIX)
    records, receive_lookup, emit_port_lookup = dft.build_compute_records(
        events, compute_idx, all_ops
    )
    _, _, emits_outside, receives_outside, _ = dft._trace_health(events, compute_idx)
    paths = dft.enumerate_paths(adj, sources, sinks)
    series = dft.path_series(
        paths,
        records,
        receive_lookup,
        emit_port_lookup,
        compute_idx,
    )
    return adj, sources, sinks, paths, series, emits_outside + receives_outside


def _latencies(entry):
    return [sample.latency_ns for sample in entry['samples']]


def _source_keys(entry):
    return [sample.source_key for sample in entry['samples']]


@pytest.mark.parametrize(
    ('domain_ids', 'expected_error'),
    [
        ([1], 'The trace does not contain the expected'),
        ([1, 2], "2 'Holoscan OCL Message Payload' schema registrations were found"),
    ],
)
def test_load_events_rejects_invalid_ocl_schema_registration(domain_ids, expected_error):
    database = sqlite3.connect(':memory:')
    database.execute('CREATE TABLE NVTX_EVENTS (binaryData BLOB)')
    database.execute(
        'CREATE TABLE NVTX_PAYLOAD_SCHEMAS '
        '(domainId INTEGER, schemaId INTEGER, payloadSize INTEGER, name TEXT)'
    )
    database.execute(
        'CREATE TABLE NVTX_PAYLOAD_SCHEMA_ENTRIES '
        '(domainId INTEGER, schemaId INTEGER, name TEXT, type INTEGER, offset INTEGER)'
    )
    database.executemany(
        'INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (?, 1, 16, ?)',
        [(domain_id, dft.OCL_MESSAGE_PAYLOAD_SCHEMA) for domain_id in domain_ids],
    )
    report = dft.Ocl.__new__(dft.Ocl)
    report._dbcon = database

    events, error = report._load_events()

    assert events is None
    assert error.startswith(expected_error)


def test_load_events_reads_only_the_ocl_schema_domain():
    database = sqlite3.connect(':memory:')
    database.execute('CREATE TABLE StringIds (id INTEGER, value TEXT)')
    database.execute("INSERT INTO StringIds VALUES (7, 'A.out')")
    database.execute(
        'CREATE TABLE NVTX_EVENTS '
        '(text TEXT, textId INTEGER, start INTEGER, end INTEGER, binaryData BLOB, '
        'globalTid INTEGER, domainId INTEGER)'
    )
    database.execute(
        'CREATE TABLE NVTX_PAYLOAD_SCHEMAS '
        '(domainId INTEGER, schemaId INTEGER, payloadSize INTEGER, name TEXT)'
    )
    database.execute(
        'CREATE TABLE NVTX_PAYLOAD_SCHEMA_ENTRIES '
        '(domainId INTEGER, schemaId INTEGER, name TEXT, type INTEGER, offset INTEGER)'
    )
    database.execute(
        'INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (1, 2, 16, ?)',
        (dft.OCL_MESSAGE_PAYLOAD_SCHEMA,),
    )
    database.executemany(
        'INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES (1, 2, ?, ?, ?)',
        [
            ('source_identity', dft.REGISTERED_STRING_HANDLE_TYPE, 0),
            ('sequence', dft.UINT64_TYPE, 8),
        ],
    )
    payload = struct.pack('<QQQQQQ', 1, 2, 16, 32, 7, 1)
    database.executemany(
        'INSERT INTO NVTX_EVENTS VALUES (?, NULL, ?, ?, ?, 1, ?)',
        [
            ('A:compute', 0, 100, None, 1),
            ('A.out:emit', 10, 10, payload, 1),
            ('foreign:compute', 0, 100, None, 2),
        ],
    )
    report = dft.Ocl.__new__(dft.Ocl)
    report._dbcon = database

    events, error = report._load_events()

    assert error is None
    assert [event[0] for event in events] == ['A:compute', 'A.out:emit']


def test_linear_path_builds_one_completed_sample():
    events = [
        _compute('A', 0, 100),
        _emit('A.out', 20, 1),
        _compute('B', 110, 210),
        _receive('B.in', 120, 'A.out', 1),
        _emit('B.out', 180, 1),
        _compute('C', 220, 330),
        _receive('C.in', 230, 'B.out', 1),
    ]

    adj, sources, sinks, paths, series, dropped = _series_for(events)

    assert dropped == 0
    assert adj == {'A': {'B'}, 'B': {'C'}}
    assert sources == ['A']
    assert sinks == ['C']
    assert paths == [['A', 'B', 'C']]
    assert len(series) == 1
    assert series[0]['path_str'] == 'A -> B -> C'
    assert series[0]['port_path'] == 'A.out -> B.in -> B.out -> C.in'
    assert _latencies(series[0]) == [330]
    assert _source_keys(series[0]) == [('A.out', 1)]
    compute_idx = dft.build_range_index(events, dft.COMPUTE_SUFFIX)
    assert dft._trace_health(events, compute_idx) == ([], [], 0, 0, 0)


def test_topology_does_not_require_emit_marker_before_receive_marker():
    events = [
        _compute('A', 0, 100),
        _compute('B', 50, 150),
        _receive('B.in', 80, 'A.out', 1),
        _emit('A.out', 90, 1),
    ]

    adj, sources, sinks, paths, series, dropped = _series_for(events)

    assert dropped == 0
    assert adj == {'A': {'B'}}
    assert sources == ['A']
    assert sinks == ['B']
    assert paths == [['A', 'B']]
    assert len(series) == 1
    assert _latencies(series[0]) == [150]


def test_fanout_enumerates_one_path_per_downstream_branch():
    events = [
        _compute('A', 0, 100),
        _emit('A.to_b', 20, 1),
        _emit('A.to_c', 30, 1),
        _compute('B', 110, 210),
        _receive('B.in', 120, 'A.to_b', 1),
        _compute('C', 115, 215),
        _receive('C.in', 125, 'A.to_c', 1),
    ]

    paths, series, dropped = _series_for(events)[3:6]

    assert dropped == 0
    assert sorted(paths) == [['A', 'B'], ['A', 'C']]
    assert {(entry['path_str'], entry['port_path']) for entry in series} == {
        ('A -> B', 'A.to_b -> B.in'),
        ('A -> C', 'A.to_c -> C.in'),
    }
    assert {(entry['path_str'], entry['port_path']): _latencies(entry) for entry in series} == {
        ('A -> B', 'A.to_b -> B.in'): [210],
        ('A -> C', 'A.to_c -> C.in'): [215],
    }


def test_repeated_messages_on_same_edge_aggregate_into_one_operator_path():
    events = [
        _compute('A', 0, 100),
        _emit('A.out', 10, 1),
        _compute('B', 110, 200),
        _receive('B.in', 120, 'A.out', 1),
        _compute('A', 300, 400),
        _emit('A.out', 320, 2),
        _compute('B', 410, 530),
        _receive('B.in', 420, 'A.out', 2),
        _compute('A', 600, 700),
        _emit('A.out', 620, 3),
    ]

    series, dropped = _series_for(events)[4:6]

    assert dropped == 0
    assert len(series) == 1
    assert series[0]['port_path'] == 'A.out -> B.in'
    assert _latencies(series[0]) == [200, 230]
    assert _source_keys(series[0]) == [('A.out', 1), ('A.out', 2)]
    assert series[0]['source_emits'] == [
        (0, 10, ('A.out', 1)),
        (300, 320, ('A.out', 2)),
        (600, 620, ('A.out', 3)),
    ]

    selected = dft.select_series(series, warmup=1)[0]
    assert selected['eligible_source_count'] == 2
    assert selected['source_identity_count'] == 1
    assert selected['unmatched_source_count'] == 1
    assert selected['excluded_source_count'] == 1
    assert selected['emit_hz'] == 3_333_333.33


def test_never_received_sibling_source_port_has_unmatched_row():
    events = [
        _compute('A', 0, 100),
        _emit('A.connected', 10, 1),
        _emit('A.unreceived', 20, 1),
        _compute('B', 110, 210),
        _receive('B.in', 120, 'A.connected', 1),
    ]

    series = _series_for(events)[4]

    assert {(entry['path_str'], entry['port_path']): _latencies(entry) for entry in series} == {
        ('A -> B', 'A.connected -> B.in'): [210],
        ('A -> ?', 'A.unreceived -> ?'): [],
    }
    orphan = next(entry for entry in series if entry['port_path'] == 'A.unreceived -> ?')
    selected = dft.select_series([orphan])[0]
    assert selected['eligible_source_count'] == 1
    assert selected['unmatched_source_count'] == 1
    assert selected['source_callback_count'] == 1


def test_different_sink_ports_in_same_callback_have_separate_port_paths():
    events = [
        _compute('A', 0, 100),
        _emit('A.out', 10, 1),
        _compute('B', 110, 200),
        _receive('B.command', 120, 'A.out', 1),
        _receive('B.validation', 130, 'A.out', 1),
    ]

    series, dropped = _series_for(events)[4:6]

    assert dropped == 0
    assert {entry['port_path']: _latencies(entry) for entry in series} == {
        'A.out -> B.command': [200],
        'A.out -> B.validation': [200],
    }


def test_different_sink_ports_in_separate_callbacks_keep_their_latencies():
    events = [
        _compute('A', 0, 100),
        _emit('A.out', 10, 1),
        _compute('B', 110, 200),
        _receive('B.command', 120, 'A.out', 1),
        _compute('B', 300, 400),
        _receive('B.validation', 310, 'A.out', 1),
    ]

    series, dropped = _series_for(events)[4:6]

    assert dropped == 0
    assert {entry['port_path']: _latencies(entry) for entry in series} == {
        'A.out -> B.command': [200],
        'A.out -> B.validation': [400],
    }
    assert {entry['port_path']: entry['samples'][0].sink_start_ns for entry in series} == {
        'A.out -> B.command': 110,
        'A.out -> B.validation': 300,
    }


def test_two_messages_from_one_source_callback_remain_two_samples():
    events = [
        _compute('A', 0, 100),
        _emit('A.out', 10, 1),
        _emit('A.out', 20, 2),
        _compute('B', 110, 200),
        _receive('B.in', 120, 'A.out', 1),
        _receive('B.in', 130, 'A.out', 2),
    ]

    series, dropped = _series_for(events)[4:6]

    assert dropped == 0
    assert len(series) == 1
    assert _latencies(series[0]) == [200, 200]
    assert _source_keys(series[0]) == [('A.out', 1), ('A.out', 2)]


def test_reused_message_counts_each_distinct_compute_chain():
    events = [
        _compute('A', 0, 100),
        _emit('A.out', 10, 1),
        _compute('B', 110, 200),
        _receive('B.in', 120, 'A.out', 1),
        _compute('B', 410, 500),
        _receive('B.in', 420, 'A.out', 1),
    ]

    series, dropped = _series_for(events)[4:6]

    assert dropped == 0
    assert len(series) == 1
    assert series[0]['port_path'] == 'A.out -> B.in'
    assert _latencies(series[0]) == [200, 500]
    assert _source_keys(series[0]) == [('A.out', 1), ('A.out', 1)]


def test_first_response_keeps_counts_from_all_responses():
    events = [
        _compute('A', 0, 100),
        _emit('A.out', 10, 1),
        _compute('B', 110, 200),
        _receive('B.in', 120, 'A.out', 1),
        _compute('B', 400, 500),
        _receive('B.in', 410, 'A.out', 1),
    ]
    series = _series_for(events)[4]

    all_responses = dft.select_series(series, responses='all')[0]
    first_response = dft.select_series(series, responses='first')[0]

    assert _latencies(all_responses) == [200, 500]
    assert _latencies(first_response) == [200]
    assert first_response['response_count'] == 2
    assert first_response['source_identity_count'] == 1


def test_rate_uses_intervals_and_reports_unavailable_values():
    assert dft._rate_hz([]) is None
    assert dft._rate_hz([0]) is None
    assert dft._rate_hz([10, 10]) is None
    assert dft._rate_hz([index * 20_000_000 for index in range(1000)]) == 50.0


def test_first_response_keeps_sink_completion_cadence():
    sample_a = dft.PathSample(('A.out', 1), 0, 100, 200)
    repeated_a = dft.PathSample(('A.out', 1), 0, 300, 400)
    sample_b = dft.PathSample(('A.out', 2), 1000, 1100, 1200)
    series = [
        {
            'path_str': 'A -> B',
            'port_path': 'A.out -> B.in',
            'source_emits': [
                (0, 10, sample_a.source_key),
                (1000, 1010, sample_b.source_key),
            ],
            'source_callbacks': [0, 1000],
            'samples': [sample_a, repeated_a, sample_b],
        }
    ]

    selected = dft.select_series(series, responses='first')[0]

    assert selected['sink_completion_count'] == 3
    assert selected['sink_completion_hz'] == 2_000_000.0


def test_cadence_counts_source_callback_without_emit():
    events = [
        _compute('A', 0, 50),
        _emit('A.out', 10, 1),
        _compute('B', 60, 70),
        _receive('B.in', 65, 'A.out', 1),
        _compute('A', 100, 150),
        _compute('A', 200, 250),
        _emit('A.out', 220, 2),
        _compute('B', 260, 270),
        _receive('B.in', 265, 'A.out', 2),
        _compute('A', 300, 350),
    ]

    series = _series_for(events)[4]
    selected = dft.select_series(series)[0]

    assert selected['source_callback_count'] == 3
    assert selected['source_callback_hz'] == 10_000_000.0
    assert selected['eligible_source_count'] == 2
    assert selected['emit_hz'] == 4_761_904.76  # One interval from emit 10 to 220 ns.
    assert selected['sink_completion_count'] == 2
    assert selected['sink_completion_hz'] == 5_000_000.0

    time_selected = dft.select_series(series, window_unit='ns')[0]
    assert time_selected['source_callback_count'] == 4
    assert time_selected['source_callback_hz'] == 10_000_000.0


def test_message_window_uses_all_source_emits():
    keys = [('A.out', sequence) for sequence in range(1, 4)]
    series = [
        {
            'path_str': 'A -> B',
            'port_path': 'A.out -> B.in',
            'source_emits': list(zip((0, 100, 200), (10, 110, 210), keys, strict=True)),
            'source_callbacks': [0, 100, 200],
            'samples': [
                dft.PathSample(key, start, start + 10, start + 20)
                for start, key in zip((0, 100, 200), keys, strict=True)
            ],
        }
    ]

    selected = dft.select_series(series, warmup=1, window=1)[0]

    assert _source_keys(selected) == [keys[1]]
    assert selected['eligible_source_count'] == 1
    assert selected['source_identity_count'] == 1
    assert selected['unmatched_source_count'] == 0
    assert selected['excluded_source_count'] == 2


def test_window_keeps_late_response_and_counts_unmatched_source():
    responded = ('A.out', 1)
    unmatched = ('A.out', 2)
    excluded = ('A.out', 3)
    series = [
        {
            'path_str': 'A -> B',
            'port_path': 'A.out -> B.in',
            'source_emits': [
                (0, 10, responded),
                (100, 110, unmatched),
                (200, 210, excluded),
            ],
            'source_callbacks': [0, 100, 200],
            'samples': [dft.PathSample(responded, 0, 300, 400)],
        }
    ]

    selected = dft.select_series(series, window=2)[0]

    assert _source_keys(selected) == [responded]
    assert selected['eligible_source_count'] == 2
    assert selected['source_identity_count'] == 1
    assert selected['unmatched_source_count'] == 1
    assert selected['excluded_source_count'] == 1


def test_time_window_is_half_open_and_precedes_first_response_selection():
    before = ('A.out', 1)
    eligible = ('A.out', 2)
    at_end = ('A.out', 3)
    series = [
        {
            'path_str': 'A -> B',
            'port_path': 'A.out -> B.in',
            'source_emits': [
                (0, 10, before),
                (100, 110, eligible),
                (200, 210, at_end),
            ],
            'source_callbacks': [0, 100, 150, 200],
            'samples': [
                dft.PathSample(eligible, 100, 150, 160),
                dft.PathSample(eligible, 100, 300, 320),
            ],
        }
    ]

    selected = dft.select_series(
        series, responses='first', warmup=100, window=100, window_unit='ns'
    )[0]

    assert _latencies(selected) == [60]
    assert selected['response_count'] == 2
    assert selected['eligible_source_count'] == 1
    assert selected['excluded_source_count'] == 2
    assert selected['source_callback_count'] == 2
    assert selected['source_callback_hz'] == 20_000_000.0


def test_time_window_has_one_origin_across_paths(capsys):
    series = [
        {
            'path_str': op,
            'port_path': f'{op}.out -> sink.in',
            'source_emits': [(start, start, (f'{op}.out', 1))],
            'source_callbacks': [start],
            'samples': [],
        }
        for op, start in [('early', 0), ('late', 200)]
    ]

    selected = dft.select_series(series, window=100, window_unit='ns')

    assert [entry['eligible_source_count'] for entry in selected] == [1, 0]
    assert (
        'common time window ends before the first source message for: late.out -> sink.in'
        in capsys.readouterr().err
    )


def test_optional_period_error_and_latency_budget():
    keys = [('A.out', sequence) for sequence in range(4)]
    starts = [0, 1_000_000, 4_000_000, 5_000_000]
    samples = [
        dft.PathSample(key, start, start + 50, start + 100)
        for key, start in zip(keys, starts, strict=True)
    ]
    series = [
        {
            'path_str': 'A -> B',
            'port_path': 'A.out -> B.in',
            'source_emits': list(zip(starts, starts, keys, strict=True)),
            'source_callbacks': starts,
            'samples': samples,
        }
    ]

    without_limits = dft.select_series(series)[0]
    assert 'sink_period_error_p99_9_ns' not in without_limits
    assert 'latency_budget_exceedance_count' not in without_limits

    selected = dft.select_series(
        series,
        expected_period_ns=1_000_000,
        latency_budget_ns=100,
    )[0]
    assert selected['sink_period_error_p99_9_ns'] == pytest.approx(1_996_000)
    assert selected['latency_budget_exceedance_count'] == 0

    over_budget = dft.select_series(series, latency_budget_ns=99)[0]
    assert over_budget['latency_budget_exceedance_count'] == 4


@pytest.mark.parametrize(
    ('warmup', 'window', 'window_unit'),
    [(-1, None, 'messages'), (0, 0, 'messages'), (0, None, 'seconds')],
)
def test_rejects_invalid_source_windows(warmup, window, window_unit):
    with pytest.raises(ValueError):
        dft.select_series([], warmup=warmup, window=window, window_unit=window_unit)


@pytest.mark.parametrize('option', [{'expected_period_ns': 0}, {'latency_budget_ns': -1}])
def test_rejects_nonpositive_timing_limits(option):
    with pytest.raises(ValueError):
        dft.select_series([], **option)


@pytest.mark.parametrize('separator', ['->', ' -> '])
def test_port_path_selects_one_exact_lineage(separator):
    events = [
        _compute('A', 0, 100),
        _emit('A.out_a', 10, 1),
        _emit('A.out_b', 20, 2),
        _compute('B', 110, 200),
        _receive('B.in_a', 120, 'A.out_a', 1),
        _compute('B', 210, 300),
        _receive('B.in_b', 220, 'A.out_b', 2),
    ]
    series = _series_for(events)[4]

    selected = dft.select_series(series, port_path=f'A.out_b{separator}B.in_b')

    assert len(selected) == 1
    assert selected[0]['port_path'] == 'A.out_b -> B.in_b'
    assert _latencies(selected[0]) == [300]


@pytest.mark.parametrize(
    ('source', 'output', 'sink', 'input_port'),
    [
        ('left/stage/scale', 'out', 'right/stage/scale', 'in'),
        (r'left\/stage/scale', 'out', 'left/stage/scale', 'in'),
        (
            r'graph\.part\:0\\left\/right/camera\.front\:0\\left\/right',
            r'rgb\.port\:0\\left\/right',
            r'sink\.back\:1\\up\/down',
            r'in\.port\:1\\up\/down',
        ),
        (r'graph/source\\', r'out\\\.part', r'graph/sink\\', r'in\\\.part'),
        ('outer.inner.source', 'out', 'outer.inner.sink', 'in'),
    ],
)
def test_nested_and_escaped_labels_reconstruct_distinct_operator_paths(
    source, output, sink, input_port
):
    source_port = f'{source}.{output}'
    sink_port = f'{sink}.{input_port}'
    # Overlap the callbacks and receive before emit, as HSDK permits.
    events = [
        _compute(source, 0, 100),
        _compute(sink, 50, 150),
        _receive(sink_port, 80, source_port, 1),
        _emit(source_port, 90, 1),
    ]

    adj, sources, sinks, paths, series, dropped = _series_for(events)

    assert dropped == 0
    assert adj == {source: {sink}}
    assert sources == [source]
    assert sinks == [sink]
    assert paths == [[source, sink]]
    assert len(series) == 1
    assert series[0]['path_str'] == f'{source} -> {sink}'
    assert series[0]['port_path'] == f'{source_port} -> {sink_port}'
    assert _latencies(series[0]) == [150]
    assert _source_keys(series[0]) == [(source_port, 1)]
    selected = dft.select_series(series, port_path=f'{source_port}->{sink_port}')
    assert len(selected) == 1
    assert _latencies(selected[0]) == [150]


def test_port_path_keeps_graph_slashes_distinct_from_port_boundaries():
    events = [
        _compute('left/stage/source', 0, 100),
        _emit('left/stage/source.out', 20, 1),
        _compute('right/stage/sink', 110, 210),
        _receive('right/stage/sink.in', 120, 'left/stage/source.out', 1),
    ]
    series = _series_for(events)[4]

    assert (
        len(dft.select_series(series, port_path='left/stage/source.out -> right/stage/sink.in'))
        == 1
    )
    # Both spellings used to normalize to the correct path by replacing arrows
    # with slashes, silently conflating graph segments and transport edges.
    assert dft.select_series(series, port_path='left/stage/source.out/right/stage/sink.in') == []
    assert dft.select_series(series, port_path='left->stage/source.out->right/stage/sink.in') == []


def test_first_response_is_selected_independently_for_each_path():
    events = [
        _compute('A', 0, 50),
        _emit('A.out', 10, 1),
        _compute('B', 60, 100),
        _receive('B.in', 70, 'A.out', 1),
        _emit('B.to_c', 80, 2),
        _emit('B.to_d', 90, 3),
        _compute('C', 110, 200),
        _receive('C.in', 120, 'B.to_c', 2),
        _compute('D', 210, 500),
        _receive('D.in', 220, 'B.to_d', 3),
    ]
    selected = dft.select_series(_series_for(events)[4], responses='first')

    assert {entry['port_path']: _latencies(entry) for entry in selected} == {
        'A.out -> B.in -> B.to_c -> C.in': [200],
        'A.out -> B.in -> B.to_d -> D.in': [500],
    }


def test_same_compute_chain_from_multiple_source_ports_has_one_sample_per_port():
    events = [
        _compute('A', 0, 100),
        _emit('A.out_a', 10, 1),
        _emit('A.out_b', 20, 1),
        _compute('B', 110, 200),
        _receive('B.in_a', 120, 'A.out_a', 1),
        _receive('B.in_b', 130, 'A.out_b', 1),
        _emit('B.out', 150, 1),
        _compute('C', 210, 300),
        _receive('C.in', 220, 'B.out', 1),
    ]

    series, dropped = _series_for(events)[4:6]

    assert dropped == 0
    assert {entry['port_path']: _latencies(entry) for entry in series} == {
        'A.out_a -> B.in_a -> B.out -> C.in': [300],
        'A.out_b -> B.in_b -> B.out -> C.in': [300],
    }


def test_incomplete_port_path_keeps_zero_count_row():
    events = [
        _compute('A', 0, 100),
        _emit('A.complete', 10, 1),
        _emit('A.incomplete', 20, 1),
        _compute('B', 110, 200),
        _receive('B.in_complete', 120, 'A.complete', 1),
        _emit('B.out', 150, 1),
        _compute('B', 210, 280),
        _receive('B.in_incomplete', 220, 'A.incomplete', 1),
        _compute('C', 290, 400),
        _receive('C.in', 300, 'B.out', 1),
    ]

    series, dropped = _series_for(events)[4:6]

    assert dropped == 0
    assert {entry['port_path']: _latencies(entry) for entry in series} == {
        'A.complete -> B.in_complete -> B.out -> C.in': [400],
        'A.incomplete -> ?': [],
    }
    incomplete = next(entry for entry in series if entry['port_path'] == 'A.incomplete -> ?')
    selected = dft.select_series([incomplete])[0]
    assert selected['eligible_source_count'] == 1
    assert selected['source_identity_count'] == 0
    assert selected['unmatched_source_count'] == 1
    assert selected['excluded_source_count'] == 0


def test_completed_branch_keeps_incomplete_sibling_path():
    events = [
        _compute('A', 0, 100),
        _emit('A.out', 10, 1),
        _compute('X', 0, 100),
        _emit('X.out', 20, 2),
        _compute('B', 110, 200),
        _receive('B.from_a', 120, 'A.out', 1),
        _emit('B.to_d', 130, 3),
        _compute('B', 210, 300),
        _receive('B.from_x', 220, 'X.out', 2),
        _emit('B.to_c', 230, 4),
        _compute('C', 310, 400),
        _receive('C.in', 320, 'B.to_c', 4),
        _compute('D', 310, 400),
        _receive('D.in', 330, 'B.to_d', 3),
    ]

    paths, series, dropped = _series_for(events)[3:6]

    assert dropped == 0
    assert sorted(paths) == [
        ['A', 'B', 'C'],
        ['A', 'B', 'D'],
        ['X', 'B', 'C'],
        ['X', 'B', 'D'],
    ]
    # A.out did reach D. This partial row means only that no A-sourced message
    # completed the separately enumerated A -> B -> C operator path.
    assert {(entry['path_str'], entry['port_path']): _latencies(entry) for entry in series} == {
        ('A -> B -> C', 'A.out -> ?'): [],
        ('A -> B -> D', 'A.out -> B.from_a -> B.to_d -> D.in'): [400],
        ('X -> B -> C', 'X.out -> B.from_x -> B.to_c -> C.in'): [400],
        ('X -> B -> D', 'X.out -> ?'): [],
    }


def test_fanin_and_fanout_same_compute_creates_all_receive_emit_pairings():
    events = [
        _compute('A', 0, 100),
        _emit('A.out', 10, 1),
        _compute('B', 0, 100),
        _emit('B.out', 20, 1),
        _compute('C', 110, 210),
        _receive('C.from_a', 120, 'A.out', 1),
        _receive('C.from_b', 130, 'B.out', 1),
        _emit('C.to_d', 150, 1),
        _emit('C.to_e', 160, 1),
        _compute('D', 220, 320),
        _receive('D.in', 230, 'C.to_d', 1),
        _compute('E', 230, 340),
        _receive('E.in', 240, 'C.to_e', 1),
    ]

    paths, series, dropped = _series_for(events)[3:6]

    assert dropped == 0
    assert sorted(paths) == [['A', 'C', 'D'], ['A', 'C', 'E'], ['B', 'C', 'D'], ['B', 'C', 'E']]
    assert {(entry['path_str'], entry['port_path']): _latencies(entry) for entry in series} == {
        ('A -> C -> D', 'A.out -> C.from_a -> C.to_d -> D.in'): [320],
        ('A -> C -> E', 'A.out -> C.from_a -> C.to_e -> E.in'): [340],
        ('B -> C -> D', 'B.out -> C.from_b -> C.to_d -> D.in'): [320],
        ('B -> C -> E', 'B.out -> C.from_b -> C.to_e -> E.in'): [340],
    }


def test_path_series_keeps_reconstructed_path_with_no_completed_sample():
    series = dft.path_series(
        [['A', 'B', 'C']],
        {'A': {}, 'B': {}, 'C': {}},
        {'A': {}, 'B': {}, 'C': {}},
        {'A': {}, 'B': {}, 'C': {}},
        {'A': [], 'B': [], 'C': []},
    )

    assert len(series) == 1
    assert series[0]['path_str'] == 'A -> B -> C'
    assert series[0]['port_path'] is None
    assert _latencies(series[0]) == []
    assert _source_keys(series[0]) == []


def test_row_separates_responses_sources_and_selected_samples():
    row = dft._row(
        0,
        'A -> B',
        'A.out -> B.in',
        response_count=5,
        eligible_source_count=4,
        source_identity_count=3,
        unmatched_source_count=1,
        excluded_source_count=2,
        source_callback_count=6,
        source_callback_hz=50.0,
        emit_hz=25.0,
        sink_completion_count=7,
        sink_completion_hz=20.0,
        stats=dft._stat_cols([200]),
    )

    assert row[3:14] == (5, 4, 3, 1, 2, 6, 50.0, 25.0, 7, 20.0, 1)


def test_sample_rows_expose_selected_identity_and_timestamps():
    samples = [
        dft.PathSample(('A.out', 7), 100, 200, 250),
        dft.PathSample(('A.out', 7), 100, 300, 360),
    ]
    series = [
        {
            'path_str': 'A -> B',
            'port_path': 'A.out -> B.in',
            'samples': samples,
        }
    ]

    assert dft.sample_rows(series) == [
        (0, 'A -> B', 'A.out -> B.in', 'A.out', 7, 100, 200, 250, 150),
        (1, 'A -> B', 'A.out -> B.in', 'A.out', 7, 100, 300, 360, 260),
    ]


def test_samples_view_types_trace_positions_as_timestamps():
    class _Connection:
        def execute(self, _query):
            return None

        def executemany(self, _query, _rows):
            return None

    report = dft.Ocl()
    report._dbcon = _Connection()
    report.parsed_args = types.SimpleNamespace(view='samples')
    report._build_rows = lambda: ([], None)

    assert report.start_query() is None
    assert 'Source Start:ts_ns' in report.query
    assert 'Sink Start:ts_ns' in report.query
    assert 'Sink End:ts_ns' in report.query
    assert 'Latency:dur_ns' in report.query


def test_emit_and_receive_marks_outside_same_operator_compute_are_ignored():
    events = [
        _compute('A', 0, 100),
        _emit('A.out', 10, 1),
        _receive('B.in', 120, 'A.out', 1),
    ]

    series, dropped = _series_for(events)[4:6]

    assert dropped == 1
    assert len(series) == 1
    assert series[0]['path_str'] == 'A -> ?'
    assert series[0]['port_path'] == 'A.out -> ?'
    assert _latencies(series[0]) == []

    compute_idx = dft.build_range_index(events, dft.COMPUTE_SUFFIX)
    assert dft._trace_health(events, compute_idx)[2:4] == (0, 1)


def test_trace_health_reports_missing_conflicting_and_leading_markers():
    missing_receive = (f'B.missing{dft.RECEIVE_SUFFIX}', 30, None, None)
    conflicting_emit = (
        f'A.other{dft.EMIT_SUFFIX}',
        20,
        None,
        ('A.out', 2),
    )
    events = [
        _compute('B', 0, 100),
        (dft.EMIT_SUFFIX, 5, None, None),
        _emit('A.out', 10, 1),
        _emit('A.out', 15, 1),
        _emit('A.out', 18, 2),
        conflicting_emit,
        _receive('B.leading', 25, 'A.out', 3),
        _receive('C.outside', 25, 'A.out', 3),
        missing_receive,
    ]

    compute_idx = dft.build_range_index(events, dft.COMPUTE_SUFFIX)
    missing, conflicts, emits_outside, receives_outside, receive_without_emit = dft._trace_health(
        events, compute_idx
    )

    assert missing == [dft.EMIT_SUFFIX, f'B.missing{dft.RECEIVE_SUFFIX}']
    assert conflicts == [('A.out', 1), ('A.out', 2)]
    assert emits_outside == 5
    assert receives_outside == 1
    # A missing identity makes the leading-boundary count untrustworthy.
    assert receive_without_emit == 0


def test_overlapping_compute_ranges_include_touching_ranges():
    compute_idx = {
        'overlap': [(0, 100), (99, 200)],
        'touching': [(0, 100), (100, 200)],
    }

    assert dft._overlapping_compute_operators(compute_idx) == ['overlap', 'touching']


@pytest.mark.parametrize(
    ('events', 'expected_error'),
    [
        ([_emit('A.out', 10, 1)], "No ':compute' ranges were found"),
        (
            [_compute('A', 0, 100), _compute('A', 100, 200)],
            'Overlapping or touching OCL compute ranges',
        ),
        (
            [_compute('A', 0, 100), ('A.out:emit', 10, None, None)],
            '1 OCL emit/receive marker(s) have no decodable message identity: A.out:emit',
        ),
        (
            [_compute('A', 0, 100), _emit('A.out', 10, 1), _emit('A.out', 20, 1)],
            "1 OCL message identity value(s) were emitted more than once: ('A.out', 1)",
        ),
        (
            [_compute('A', 0, 100)]
            + [_emit('A.out', time, sequence) for sequence in range(4) for time in (10, 20)],
            '4 OCL message identity value(s) were emitted more than once: first three: '
            "('A.out', 0), ('A.out', 1), ('A.out', 2)",
        ),
    ],
)
def test_report_rejects_invalid_trace_before_reconstructing_paths(events, expected_error):
    report = dft.Ocl.__new__(dft.Ocl)
    report.parsed_args = types.SimpleNamespace(port_path=None, port_path_file=None)
    report._load_events = lambda: (events, None)

    rows, error = report._build_rows()

    assert rows is None
    assert error.startswith(expected_error)


def test_trace_validation_warns_without_rejecting_capture_boundaries(capsys):
    events = [
        _compute('A', 0, 100),
        _emit('A.out', 10, 1),
        _emit('A.out', 110, 2),
        _receive('B.in', 120, 'A.out', 1),
        _compute('B', 125, 200),
        _receive('B.in', 130, 'A.out', 3),
    ]
    compute_idx = dft.build_range_index(events, dft.COMPUTE_SUFFIX)

    assert dft._validate_trace(events, compute_idx) is None
    assert capsys.readouterr().err == (
        'WARNING: OCL trace health: 1 emit markers occurred outside operator callbacks, '
        'receives outside operator callbacks=1, receive identities with no captured emit=1 '
        '(possible leading capture boundary, not proof of a drop).\n'
    )


def test_zero_duration_path_is_kept_and_negative_path_fails():
    zero_events = [
        _compute('A', 0, 100),
        _emit('A.out', 10, 1),
        _compute('B', 0, 0),
        _receive('B.in', 0, 'A.out', 1),
    ]
    assert _latencies(_series_for(zero_events)[4][0]) == [0]

    negative_events = [
        _compute('A', 100, 200),
        _emit('A.out', 110, 1),
        _compute('B', 0, 50),
        _receive('B.in', 10, 'A.out', 1),
    ]
    with pytest.raises(ValueError, match='negative path latency'):
        _series_for(negative_events)


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('operator.port:emit', 'operator.port'),
        ('operator.port:receive', None),
        ('operator:compute', None),
    ],
)
def test_port_label_matches_only_requested_suffix(name, expected):
    assert dft._port_label(name, dft.EMIT_SUFFIX) == expected


@pytest.mark.parametrize('ending', ['', '\n', '\r\n'])
def test_port_path_file_preserves_escaped_names_and_spaces(tmp_path, ending):
    path = r' source.out -> graph\/stage/camera\.front.in\:0\\ '
    selector = tmp_path / 'path.txt'
    selector.write_bytes((path + ending).encode('utf-8'))

    assert dft._resolve_port_path(None, str(selector)) == path


@pytest.mark.parametrize('contents', ['', '\n', ' \t\n', 'A.out\nB.in', 'A.out -> B.in\n\n'])
def test_port_path_file_rejects_empty_or_multiline_contents(tmp_path, contents):
    selector = tmp_path / 'path.txt'
    selector.write_text(contents, encoding='utf-8')

    with pytest.raises(ValueError, match='exactly one non-empty line'):
        dft._resolve_port_path(None, str(selector))


@pytest.mark.parametrize('kind', ['missing', 'directory', 'invalid-utf8'])
def test_port_path_file_reports_read_errors(tmp_path, kind):
    selector = tmp_path / 'path.txt'
    if kind == 'directory':
        selector.mkdir()
    elif kind == 'invalid-utf8':
        selector.write_bytes(b'\xff')

    with pytest.raises(ValueError, match='Unable to read port path file') as error:
        dft._resolve_port_path(None, str(selector))
    assert str(selector) in str(error.value)


def test_port_path_options_are_mutually_exclusive_before_loading_trace():
    report = dft.Ocl.__new__(dft.Ocl)
    report.parsed_args = types.SimpleNamespace(port_path='A.out->B.in', port_path_file='path.txt')

    rows, error = report._build_rows()

    assert rows is None
    assert error == 'Specify either port-path or port-path-file, not both.'


@pytest.mark.parametrize('selection', ['all', 'inline', 'file', 'unmatched-file'])
def test_report_selects_escaped_long_paths_from_file(tmp_path, selection):
    source = r'outer\/stage/camera\.front\:0\\'
    source_port = source + '.' + 'long_' * 70 + r'out\:0'
    path = f'{source_port} -> sink.in'
    events = [
        _compute(source, 0, 100),
        _emit(source_port, 20, 1),
        _compute('sink', 110, 210),
        _receive('sink.in', 120, source_port, 1),
        _compute('other', 115, 215),
        _receive('other.in', 125, source_port, 1),
    ]
    selector = tmp_path / 'path.txt'
    selector.write_text('unknown.out -> sink.in' if selection == 'unmatched-file' else path + '\n')
    report = dft.Ocl.__new__(dft.Ocl)
    report.parsed_args = types.SimpleNamespace(
        port_path=path if selection == 'inline' else None,
        port_path_file=str(selector) if selection in ('file', 'unmatched-file') else None,
        responses='all',
        warmup=0,
        window=None,
        window_unit='messages',
        expected_period_ns=None,
        latency_budget_ns=None,
        view='samples',
    )
    report._load_events = lambda: (events, None)

    rows, error = report._build_rows()

    if selection == 'unmatched-file':
        assert rows is None
        assert "No Port Path exactly matched 'unknown.out -> sink.in'" in error
        assert path in error
    else:
        assert error is None
        assert len(rows) == (2 if selection == 'all' else 1)
        expected = {path, f'{source_port} -> other.in'} if selection == 'all' else {path}
        assert {row[2] for row in rows} == expected
        assert {row[-1] for row in rows} == ({210, 215} if selection == 'all' else {210})
