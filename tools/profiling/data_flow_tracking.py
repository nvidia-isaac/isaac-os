#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ocl -- Holoscan Operator-Centric Latency (nsys stats report)

A self-contained `nsys stats` report that turns a Holoscan (HSDK 5) NVTX
operator-profiling capture into per-path end-to-end latency statistics:

    nsys export --type sqlite --include-blobs=true --force-overwrite=true \
        -o <report>.sqlite <report>.nsys-rep
    nsys stats --report-dir <this_dir> -r data_flow_tracking <report>.nsys-rep

THE EXPORT STEP IS REQUIRED. This report reads its correlation key from an NVTX
extended (binary) payload, which is only present when the SQLite export is made
with --include-blobs. The SQLite that `nsys stats` generates on its own uses the
default --include-blobs=false and can never work; the report detects that case
and stops with the exact command to run.

It emits one flat table with one row per exact source-to-sink port path via the
standard report query, so nsys' own formatters and output routing apply directly:

    -f column|table|csv|tsv|json|hdoc|htable   (default: console table)
    -o <path>                                  write to a file (nsys-named)
    --timeunit nsec|usec|msec|sec              rescales timestamp and duration columns

Markers consumed (HSDK 5):

    <op>:compute            range          one operator compute cycle
    <op>.<port>:emit        instantaneous  output port emitted a message
    <op>.<port>:receive     instantaneous  input port took in a message
Nested operators use the full graph path, e.g. 'left/stage/scale.out:emit'.
The root graph name is omitted. Literal backslashes, dots, colons, and slashes
inside name components are backslash-escaped; keep those escapes in labels.
The emit/receive marks carry the structured payload (source_identity,
sequence). The message key is (source_identity, sequence). Matching
an emit key to a receive key reconstructs transport edges. Within an operator,
receives and emits are correlated when they occur in the same ':compute' range.
DFS over the resulting graph enumerates every source-to-sink path. Exact port
lineage separates rows when an operator path uses different ports.

End-to-end is measured start_to_end: the source operator's ':compute' start to
the sink operator's ':compute' end, for the ticks chained by one message. Emit
and receive marks are attributed to a compute range belonging to the SAME
operator name, so parallel branches cannot cross-assign.

Absolute trace positions are typed ':ts_ns' and elapsed times are typed
':dur_ns'. Both hold raw nanoseconds and nsys rescales them per --timeunit,
which DEFAULTS TO NANOSECONDS (pass usec|msec|sec to change it). Counts and Hz
are untyped.
"""

from collections import defaultdict
from functools import cache
import os
import struct
import sys
from typing import NamedTuple

import nsysstats

# ── OCL NVTX marker + payload decoding ───────────────────────────────────────

OCL_MESSAGE_PAYLOAD_SCHEMA = 'Holoscan OCL Message Payload'
PAYLOAD_HEADER_SIZE = 32
REGISTERED_STRING_HANDLE_TYPE = 80
UINT64_TYPE = 18

COMPUTE_SUFFIX = ':compute'
EMIT_SUFFIX = ':emit'
RECEIVE_SUFFIX = ':receive'


class PathSample(NamedTuple):
    """One source message's trace-visible completion on an exact port path."""

    source_key: tuple[str, int]
    source_start_ns: int
    sink_start_ns: int
    sink_end_ns: int

    @property
    def latency_ns(self):
        return self.sink_end_ns - self.source_start_ns


def _port_label(name, suffix):
    """Return the port label before an OCL annotation suffix, or None."""
    return name[: -len(suffix)] if name.endswith(suffix) else None


def _op_from_port_label(port_label):
    """Remove the port at the last unescaped dot, preserving the operator label.

    HSDK graph paths use '/', and literal punctuation in name components is
    backslash-escaped. Keeping escapes lets this label match its compute range
    without aliasing literal slashes with graph boundaries. Taking the last
    unescaped dot also supports older captures with dotted graph prefixes.
    """
    separator = None
    escaped = False
    for index, char in enumerate(port_label):
        if escaped:
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '.':
            separator = index
    return port_label if separator is None else port_label[:separator]


def _bytes_from_binary_data(binary_data):
    """Normalize SQLite BLOB/TEXT binaryData exports into bytes."""
    if binary_data is None:
        return None
    if isinstance(binary_data, bytes):
        return binary_data
    if isinstance(binary_data, memoryview):
        return binary_data.tobytes()
    if isinstance(binary_data, str):
        text = binary_data.strip()
        if not text:
            return None
        if text.startswith("x'") and text.endswith("'"):
            text = text[2:-1]
        elif text.startswith("X'") and text.endswith("'"):
            text = text[2:-1]
        elif text.startswith('0x') or text.startswith('0X'):
            text = text[2:]
        compact = ''.join(text.split())
        try:
            return bytes.fromhex(compact)
        except ValueError:
            return None
    return None


def _payload_layout(database, schema_name, expected_types):
    """Return (domain_id, schema_id, payload_size, offsets) for an NVTX payload schema."""
    schema = database.execute(
        'SELECT domainId, schemaId, payloadSize FROM NVTX_PAYLOAD_SCHEMAS WHERE name = ?',
        (schema_name,),
    ).fetchone()
    if schema is None:
        return None
    domain_id, schema_id, payload_size = schema
    entries = {
        name: (field_type, 0 if offset is None else offset)
        for name, field_type, offset in database.execute(
            'SELECT name, type, offset FROM NVTX_PAYLOAD_SCHEMA_ENTRIES '
            'WHERE domainId = ? AND schemaId = ?',
            (domain_id, schema_id),
        )
    }
    if not expected_types.keys() <= entries.keys():
        return None
    if any(entries[name][0] != field_type for name, field_type in expected_types.items()):
        return None
    return domain_id, schema_id, payload_size, {name: entries[name][1] for name in expected_types}


def _decode_payload(binary_data, layout):
    """Decode one static structured NVTX payload envelope into integer fields."""
    data = _bytes_from_binary_data(binary_data)
    if data is None or len(data) < PAYLOAD_HEADER_SIZE or layout is None:
        return None
    _, schema_id, layout_payload_size, offsets = layout
    try:
        payload_count, event_schema_id, payload_size, payload_offset = struct.unpack_from(
            '<QQQQ', data
        )
    except struct.error:
        return None
    if (
        payload_count != 1
        or event_schema_id != schema_id
        or payload_size < layout_payload_size
        or payload_offset + payload_size > len(data)
        or any(offset + 8 > payload_size for offset in offsets.values())
    ):
        return None
    try:
        return {
            name: struct.unpack_from('<Q', data, payload_offset + offset)[0]
            for name, offset in offsets.items()
        }
    except struct.error:
        return None


def _message_payload(binary_data, string_ids, layout):
    """Return the message key from a BLOB extended payload."""
    payload = _decode_payload(binary_data, layout)
    if payload is None:
        return None
    source_identity = string_ids.get(payload['source_identity'])
    if source_identity is None:
        return None
    return source_identity, int(payload['sequence'])


# ── Topology: emit(key) -> receive(key) ──────────────────────────────────────


def build_topology(events):
    """Match emit(key)->receive(key) marks -> (adj, sources, sinks, all_ops).

    Port labels embed the operator: 'tx.out' -> op='tx', 'inc.in' -> op='inc'.
    The key is (source_identity, sequence) from the structured payload. An operator
    that fans out simply gains more than one adjacency, and enumerate_paths
    turns each into its own path.
    """
    key_to_src = {}
    for name, _, _, corr in events:
        emit_port = _port_label(name, EMIT_SUFFIX)
        if corr is not None and emit_port is not None:
            # The runtime identity contains the output port and its monotonic sequence.
            key_to_src[corr] = _op_from_port_label(emit_port)
    edges = set()
    for name, _, _, corr in events:
        if corr is None:
            continue
        receive_port = _port_label(name, RECEIVE_SUFFIX)
        if receive_port is not None:
            src_op = key_to_src.get(corr)
            if src_op is None:
                continue
            dst_op = _op_from_port_label(receive_port)
            if src_op != dst_op:
                edges.add((src_op, dst_op))

    all_ops = set()
    adj = defaultdict(set)
    in_degree = defaultdict(int)
    for src, dst in edges:
        adj[src].add(dst)
        in_degree[dst] += 1
        all_ops.update([src, dst])

    sources = sorted(op for op in all_ops if in_degree[op] == 0)
    sinks = sorted(op for op in all_ops if not adj.get(op))
    return dict(adj), sources, sinks, all_ops


def enumerate_paths(adj, sources, sinks):
    """DFS from each source to enumerate all source-to-sink paths.

    A fan-out node contributes one path per outgoing branch; `visited` keeps a
    cycle from looping forever.
    """
    sink_set = set(sinks)
    paths = []

    def dfs(node, path, visited):
        if node in sink_set:
            paths.append(list(path))
            return
        for nxt in sorted(adj.get(node, [])):
            if nxt not in visited:
                visited.add(nxt)
                dfs(nxt, path + [nxt], visited)
                visited.discard(nxt)

    for src in sorted(sources):
        dfs(src, [src], {src})
    return paths


# ── Compute-scoped message records ───────────────────────────────────────────


def build_range_index(events, suffix):
    """{op: sorted[(start_ns, end_ns)]} from all '{op}{suffix}' Start/End ranges."""
    idx = defaultdict(list)
    for name, start_ns, end_ns, _ in events:
        if name.endswith(suffix) and end_ns is not None:
            idx[name[: -len(suffix)]].append((start_ns, end_ns))
    for op in idx:
        idx[op].sort()
    return dict(idx)


def _overlapping_compute_operators(compute_idx):
    """Return operators whose closed compute ranges overlap or touch."""
    return sorted(
        op
        for op, ranges in compute_idx.items()
        if any(current[0] <= previous[1] for previous, current in zip(ranges, ranges[1:]))
    )


def find_enclosing_compute(compute_index, op, query_ns):
    """Binary-search op's compute window containing query_ns -> (start_ns, end_ns) or (None, None).

    `compute_index` is keyed by operator name, so a mark can only ever match a
    compute range belonging to THAT operator -- parallel branches running
    concurrently cannot steal each other's marks. Assumes an operator's own
    compute ranges do not overlap, which holds because one operator's compute
    cycles are serialized.
    """
    windows = compute_index.get(op, [])
    lo, hi = 0, len(windows) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start_ns, end_ns = windows[mid]
        if start_ns <= query_ns <= end_ns:
            return start_ns, end_ns
        if query_ns < start_ns:
            hi = mid - 1
        else:
            lo = mid + 1
    return None, None


def _trace_health(events, compute_idx):
    """Return invalid marker details and non-overlapping boundary counts.

    The leading-boundary count is zero when any identity is missing because it
    cannot be trusted on a partially decodable trace.
    """
    missing = []
    emit_keys = set()
    conflicts = set()
    receive_keys = set()
    emit_count = 0
    receive_count = 0
    for name, start_ns, _, corr in events:
        emit_port = _port_label(name, EMIT_SUFFIX)
        receive_port = _port_label(name, RECEIVE_SUFFIX)
        port = emit_port if emit_port is not None else receive_port
        if port is None:
            continue
        compute_start_ns, _ = find_enclosing_compute(
            compute_idx, _op_from_port_label(port), start_ns
        )
        if compute_start_ns is None:
            if emit_port is not None:
                emit_count += 1
            else:
                receive_count += 1
        if corr is None:
            missing.append(name)
        elif emit_port is not None:
            if corr in emit_keys:
                conflicts.add(corr)
            else:
                emit_keys.add(corr)
        elif compute_start_ns is not None:
            receive_keys.add(corr)
    return (
        sorted(missing),
        sorted(conflicts),
        emit_count,
        receive_count,
        0 if missing else len(receive_keys - emit_keys),
    )


def build_compute_records(events, compute_idx, ops):
    """Bucket emit/receive message keys into the operator compute cycle they occurred in.

    Every emit/receive mark is assigned to the ':compute' window of its OWN
    operator that encloses it. Receives and emits in the same operator compute
    range are treated as causally connected.

    Returns:
        records        : {op: {compute_start: {"compute_start","compute_end",
                                               "published":{key: emit_ns}}}}
        receive_lookup      : {op: {key: [(compute_start, compute_end, port_label)]}}
                              inverse receive index
        emit_port_lookup    : {op: {key: '<op>.<port>'}} emit port labels
    """
    records = {op: {} for op in ops}
    receive_lookup = {op: {} for op in ops}
    emit_port_lookup = {op: {} for op in ops}

    def rec_for(op, query_ns):
        compute_start_ns, compute_end_ns = find_enclosing_compute(compute_idx, op, query_ns)
        if compute_start_ns is None:
            return None  # mark outside any compute -> skip
        record = records[op].get(compute_start_ns)
        if record is None:
            record = {
                'compute_start': compute_start_ns,
                'compute_end': compute_end_ns,
                'published': {},
            }
            records[op][compute_start_ns] = record
        return record

    for name, start_ns, _, corr in events:
        if corr is None:
            continue

        emit_port = _port_label(name, EMIT_SUFFIX)
        if emit_port is not None:
            op = _op_from_port_label(emit_port)
            if op in records:
                record = rec_for(op, start_ns)
                if record is not None:
                    record['published'].setdefault(corr, start_ns)
                    emit_port_lookup[op][corr] = emit_port
            continue

        receive_port = _port_label(name, RECEIVE_SUFFIX)
        if receive_port is not None:
            op = _op_from_port_label(receive_port)
            if op in records:
                record = rec_for(op, start_ns)
                if record is not None:
                    receive_lookup[op].setdefault(corr, []).append(
                        (record['compute_start'], record['compute_end'], receive_port)
                    )

    return records, receive_lookup, emit_port_lookup


def _rate_hz(timestamps_ns):
    if len(timestamps_ns) < 2:
        return None
    ordered = sorted(timestamps_ns)
    span_ns = ordered[-1] - ordered[0]
    if span_ns <= 0:
        return None
    return round((len(ordered) - 1) / (span_ns / 1e9), 2)


def chain_path(  # pylint: disable=cell-var-from-loop
    path, records, receive_lookup, emit_port_lookup
):
    """Per-path samples grouped by exact emit/receive port lineage.

    Transport is still matched by identical message keys:
        emit(message_id) -> receive(message_id)

    The internal operator hop is inferred inside one compute cycle:
        received message_id in compute range -> emitted message_id in same range

    Paths are grouped by the exact emit/receive port lineage. The readable
    operator-only path is retained alongside it.

    Reused/latest-selected messages can be received by the same operator in
    multiple compute cycles. Each receive occurrence is followed independently.
    Exact duplicates of port lineage, source message identity, and operator
    compute chain count once.
    """
    src = path[0]
    grouped = defaultdict(
        lambda: {
            'path_str': None,
            'port_path': None,
            'source_port_label': None,
            'samples': [],
        }
    )

    for compute_start_ns in sorted(records.get(src, {})):
        source_record = records[src][compute_start_ns]
        if not source_record['published']:
            continue
        source_start_ns = source_record['compute_start']
        source_compute = (src, source_record['compute_start'], source_record['compute_end'])
        matched = set()

        def follow(
            hop_index,
            message_id,
            display_nodes,
            lineage_nodes,
            compute_chain,
            *,
            source_key,
        ):
            nxt = path[hop_index]
            receive_occurrences = receive_lookup.get(nxt, {}).get(message_id, [])
            if not receive_occurrences:
                return

            for (
                receive_compute_start_ns,
                receive_compute_end_ns,
                receive_port,
            ) in receive_occurrences:
                receive_record = records[nxt][receive_compute_start_ns]
                next_compute_chain = compute_chain + (
                    (nxt, receive_compute_start_ns, receive_compute_end_ns),
                )

                if hop_index == len(path) - 1:
                    path_str = ' -> '.join(display_nodes + [nxt])
                    path_key = ' -> '.join(lineage_nodes + [receive_port])
                    matched.add(
                        (
                            path_key,
                            path_str,
                            source_key,
                            receive_record['compute_start'],
                            receive_record['compute_end'],
                            next_compute_chain,
                        )
                    )
                    continue

                for child_id in sorted(receive_record['published']):
                    emit_port = emit_port_lookup.get(nxt, {}).get(child_id)
                    if emit_port is None:
                        continue
                    follow(
                        hop_index + 1,
                        child_id,
                        display_nodes + [nxt],
                        lineage_nodes + [receive_port, emit_port],
                        next_compute_chain,
                        source_key=source_key,
                    )

        for key in sorted(source_record['published']):
            emit_port = emit_port_lookup.get(src, {}).get(key)
            if emit_port is None:
                continue
            follow(
                1,
                key,
                [src],
                [emit_port],
                (source_compute,),
                source_key=key,
            )

        for (
            path_key,
            path_str,
            source_key,
            sink_start_ns,
            sink_end_ns,
            _,
        ) in sorted(matched):
            sample = PathSample(source_key, source_start_ns, sink_start_ns, sink_end_ns)
            if sample.latency_ns < 0:
                raise ValueError(
                    f'Encountered negative path latency on {path_str}: source started at '
                    f'{source_start_ns} ns and sink ended at {sink_end_ns} ns for '
                    f'{source_key!r}.'
                )
            grouped[path_key]['path_str'] = path_str
            grouped[path_key]['port_path'] = path_key
            grouped[path_key]['source_port_label'] = emit_port_lookup[src][source_key]
            grouped[path_key]['samples'].append(sample)

    return grouped


def path_series(paths, records, receive_lookup, emit_port_lookup, compute_idx):
    """Chain every path and aggregate by exact port lineage."""
    grouped = defaultdict(
        lambda: {
            'path_str': None,
            'port_path': None,
            'samples': [],
        }
    )

    @cache
    def source_emits(source_op, source_port_label):
        return sorted(
            (record['compute_start'], emit_ns, key)
            for record in records.get(source_op, {}).values()
            for key, emit_ns in record['published'].items()
            if emit_port_lookup.get(source_op, {}).get(key) == source_port_label
        )

    def set_source_events(group, source_op, source_port_label):
        group['source_emits'] = source_emits(source_op, source_port_label)
        group['source_callbacks'] = [start_ns for start_ns, _ in compute_idx.get(source_op, [])]

    for path in paths:
        seeded_path_str = ' -> '.join(path)
        source_port_labels = {
            emit_port
            for key, emit_port in emit_port_lookup.get(path[0], {}).items()
            if receive_lookup.get(path[1], {}).get(key)
        }
        for source_port_label in source_port_labels:
            partial_port_path = f'{source_port_label} -> ?'
            partial_key = ('partial', seeded_path_str, partial_port_path)
            grouped[partial_key]['path_str'] = seeded_path_str
            grouped[partial_key]['port_path'] = partial_port_path
            set_source_events(grouped[partial_key], path[0], source_port_label)

        path_groups = chain_path(path, records, receive_lookup, emit_port_lookup)
        if not source_port_labels and not emit_port_lookup.get(path[0]):
            grouped[('empty', seeded_path_str, None)]['path_str'] = seeded_path_str
        for port_path, values in path_groups.items():
            path_str = values['path_str']
            source_port_label = values['source_port_label']
            partial_port_path = f'{source_port_label} -> ?'
            grouped.pop(('partial', seeded_path_str, partial_port_path), None)
            complete_key = ('complete', port_path, None)
            grouped[complete_key]['path_str'] = path_str
            grouped[complete_key]['port_path'] = port_path
            set_source_events(grouped[complete_key], path[0], source_port_label)
            grouped[complete_key]['samples'].extend(values['samples'])

    received_keys = {
        key for operator_receives in receive_lookup.values() for key in operator_receives
    }
    for source_op in {path[0] for path in paths}:
        for source_port_label in set(emit_port_lookup.get(source_op, {}).values()):
            emits = source_emits(source_op, source_port_label)
            if any(emit[-1] in received_keys for emit in emits):
                continue
            port_path = f'{source_port_label} -> ?'
            orphan_key = ('orphan', source_op, port_path)
            grouped[orphan_key]['path_str'] = f'{source_op} -> ?'
            grouped[orphan_key]['port_path'] = port_path
            set_source_events(grouped[orphan_key], source_op, source_port_label)

    return [
        {
            'path_str': values['path_str'],
            'port_path': values['port_path'],
            'source_emits': values.get('source_emits', []),
            'source_callbacks': values.get('source_callbacks', []),
            'samples': values['samples'],
        }
        for values in grouped.values()
    ]


def _validate_selection_options(
    *, responses, warmup, window, window_unit, expected_period_ns, latency_budget_ns
):
    """Reject invalid response and window options before processing any paths."""
    if responses not in ('all', 'first'):
        raise ValueError(f'Unsupported response selection: {responses}')
    if warmup < 0:
        raise ValueError('Warmup must be non-negative.')
    if window is not None and window <= 0:
        raise ValueError('Window must be positive.')
    if window_unit not in ('messages', 'ns'):
        raise ValueError(f'Unsupported window unit: {window_unit}')
    if expected_period_ns is not None and expected_period_ns <= 0:
        raise ValueError('Expected period must be positive.')
    if latency_budget_ns is not None and latency_budget_ns <= 0:
        raise ValueError('Latency budget must be positive.')


def _message_window(source_emits, callbacks, warmup, window):
    """Select source identities by count and include callbacks between their starts."""
    window_end = None if window is None else warmup + window
    eligible_emits = source_emits[warmup:window_end]
    if not eligible_emits:
        return [], []
    callback_start_ns = eligible_emits[0][0]
    callback_end_ns = eligible_emits[-1][0]
    source_callbacks = [
        callback_ns
        for callback_ns in callbacks
        if callback_start_ns <= callback_ns <= callback_end_ns
    ]
    return eligible_emits, source_callbacks


def _select_source_window(entry, warmup, window, window_unit, time_origin_ns):
    """Select source events and flag paths starting beyond a common time window."""
    emits_by_key = {}
    for callback_ns, emit_ns, source_key in entry['source_emits']:
        emits_by_key.setdefault(source_key, (callback_ns, emit_ns, source_key))
    source_emits = sorted(emits_by_key.values())
    if window_unit == 'messages':
        eligible_emits, source_callbacks = _message_window(
            source_emits, entry['source_callbacks'], warmup, window
        )
        return source_emits, eligible_emits, source_callbacks, False

    window_start_ns = time_origin_ns + warmup
    window_end_ns = None if window is None else window_start_ns + window
    eligible_emits = [
        emit
        for emit in source_emits
        if emit[0] >= window_start_ns and (window_end_ns is None or emit[0] < window_end_ns)
    ]
    starts_after_window = bool(
        window_end_ns is not None
        and not eligible_emits
        and source_emits
        and source_emits[0][0] >= window_end_ns
    )
    source_callbacks = [
        callback_ns
        for callback_ns in entry['source_callbacks']
        if callback_ns >= window_start_ns and (window_end_ns is None or callback_ns < window_end_ns)
    ]
    return source_emits, eligible_emits, source_callbacks, starts_after_window


def _first_responses(samples):
    """Keep the earliest sink completion for each source identity."""
    by_source = {}
    for sample in samples:
        previous = by_source.get(sample.source_key)
        if previous is None or sample.sink_end_ns < previous.sink_end_ns:
            by_source[sample.source_key] = sample
    return sorted(
        by_source.values(),
        key=lambda sample: (sample.source_start_ns, sample.sink_end_ns, sample.source_key),
    )


def _select_entry(
    entry,
    source_emits,
    eligible_emits,
    source_callbacks,
    *,
    responses,
    expected_period_ns,
    latency_budget_ns,
):
    """Collect responses and cadence for the selected source events on one path."""
    eligible_keys = {source_key for _, _, source_key in eligible_emits}
    eligible_emit_times = [emit_ns for _, emit_ns, _ in eligible_emits]
    all_samples = [sample for sample in entry['samples'] if sample.source_key in eligible_keys]
    # Every row has one sink operator, so its callback range identifies the callback.
    sink_callbacks = {(sample.sink_start_ns, sample.sink_end_ns) for sample in all_samples}
    sink_completions = sorted(end_ns for _, end_ns in sink_callbacks)
    matched_keys = {sample.source_key for sample in all_samples}
    selected_samples = all_samples
    if responses == 'first':
        selected_samples = _first_responses(all_samples)
    selected = {
        **entry,
        'samples': selected_samples,
        'response_count': len(all_samples),
        'source_identity_count': len(matched_keys),
        'eligible_source_count': len(eligible_keys),
        'unmatched_source_count': len(eligible_keys - matched_keys),
        'excluded_source_count': len(source_emits) - len(eligible_keys),
        'source_callback_count': len(source_callbacks),
        'source_callback_hz': _rate_hz(source_callbacks),
        'emit_hz': _rate_hz(eligible_emit_times),
        'sink_completion_count': len(sink_callbacks),
        'sink_completion_hz': _rate_hz(sink_completions),
    }
    if expected_period_ns is not None:
        period_errors = [
            abs(current - previous - expected_period_ns)
            for previous, current in zip(sink_completions, sink_completions[1:])
        ]
        selected['sink_period_error_p99_9_ns'] = (
            _percentile(sorted(period_errors), 99.9) if period_errors else None
        )
    if latency_budget_ns is not None:
        selected['latency_budget_exceedance_count'] = sum(
            sample.latency_ns > latency_budget_ns for sample in selected_samples
        )
    return selected


def select_series(
    series,
    *,
    port_path=None,
    responses='all',
    warmup=0,
    window=None,
    window_unit='messages',
    expected_period_ns=None,
    latency_budget_ns=None,
):
    """Filter exact port paths and select all or first response per source identity."""
    _validate_selection_options(
        responses=responses,
        warmup=warmup,
        window=window,
        window_unit=window_unit,
        expected_period_ns=expected_period_ns,
        latency_budget_ns=latency_budget_ns,
    )

    source_starts = [start_ns for entry in series for start_ns, _, _ in entry['source_emits']]
    time_origin_ns = min(source_starts) if source_starts else 0

    selected_port_path = None
    if port_path is not None:
        selected_port_path = port_path.replace(' -> ', '->')
    selected_series = []
    paths_starting_after_window = []
    for entry in series:
        entry_port_path = entry['port_path']
        if selected_port_path is not None and (
            entry_port_path is None or entry_port_path.replace(' -> ', '->') != selected_port_path
        ):
            continue
        source_emits, eligible_emits, source_callbacks, starts_after_window = _select_source_window(
            entry, warmup, window, window_unit, time_origin_ns
        )
        if starts_after_window:
            paths_starting_after_window.append(entry_port_path or entry['path_str'])
        selected_series.append(
            _select_entry(
                entry,
                source_emits,
                eligible_emits,
                source_callbacks,
                responses=responses,
                expected_period_ns=expected_period_ns,
                latency_budget_ns=latency_budget_ns,
            )
        )
    if paths_starting_after_window:
        sys.stderr.write(
            'WARNING: the common time window ends before the first source message for: '
            + ', '.join(paths_starting_after_window)
            + '. Time windows start from the earliest source message across all paths.\n'
        )
    return selected_series


# ── Statistics (pure Python; nsys' embedded interpreter has no numpy) ─────────


def _percentile(sorted_vals, q):
    """Linear-interpolation percentile (matches numpy.percentile default 'linear')."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = q / 100.0 * (n - 1)
    lo = int(rank)  # floor (rank >= 0)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (rank - lo) * (sorted_vals[hi] - sorted_vals[lo])


def stats_ns(values_ns):
    """Nanosecond latency statistics from a list of ns values ({} if empty)."""
    if not values_ns:
        return {}
    a = sorted(float(v) for v in values_ns)
    n = len(a)
    mean = sum(a) / n
    std = (sum((x - mean) ** 2 for x in a) / n) ** 0.5  # population std
    return {
        'count': n,
        'mean': mean,
        'min': a[0],
        'max': a[-1],
        'p50': _percentile(a, 50),
        'p90': _percentile(a, 90),
        'p99': _percentile(a, 99),
        'p99_9': _percentile(a, 99.9),
        'std': std,
    }


# Statistics columns of the table, in order ('avg' == mean; all raw ns).
_STAT_KEYS = ('count', 'avg', 'min', 'max', 'p50', 'p90', 'p99', 'p99_9', 'std')


def _stat_cols(values_ns):
    """Raw-nanosecond stats dict keyed for the table ('avg' == mean), or None.

    No unit conversion: durations stay in nanoseconds so nsys' --timeunit
    (default nsec) can rescale the ':dur_ns'-typed columns.
    """
    s = stats_ns(values_ns)
    if not s:
        return None
    return {
        'count': s['count'],
        'avg': s['mean'],
        'min': s['min'],
        'max': s['max'],
        'p50': s['p50'],
        'p90': s['p90'],
        'p99': s['p99'],
        'p99_9': s['p99_9'],
        'std': s['std'],
    }


def _row(
    seq,
    path_str,
    port_path,
    *,
    response_count,
    eligible_source_count,
    source_identity_count,
    unmatched_source_count,
    excluded_source_count,
    source_callback_count,
    source_callback_hz,
    emit_hz,
    sink_completion_count,
    sink_completion_hz,
    stats=None,
    sink_period_error_p99_9_ns=None,
    latency_budget_exceedance_count=None,
):
    """One table row tuple in temp.ocl column order.

    `stats` is a _stat_cols() dict, or None for a path that never completed --
    which reports an explicit zero Selected count with NULL statistics rather
    than being dropped, so a broken path stays visible.
    """
    if stats is None:
        stats = dict.fromkeys(_STAT_KEYS)
        stats['count'] = 0
    return (
        (
            seq,
            path_str,
            port_path,
            response_count,
            eligible_source_count,
            source_identity_count,
            unmatched_source_count,
            excluded_source_count,
            source_callback_count,
            source_callback_hz,
            emit_hz,
            sink_completion_count,
            sink_completion_hz,
        )
        + tuple(stats[k] for k in _STAT_KEYS)
        + (sink_period_error_p99_9_ns, latency_budget_exceedance_count)
    )


def sample_rows(series):
    """Selected samples in machine-readable table order."""
    rows = []
    for entry in series:
        for sample in entry['samples']:
            source_identity, source_sequence = sample.source_key
            rows.append(
                (
                    len(rows),
                    entry['path_str'],
                    entry['port_path'],
                    source_identity,
                    source_sequence,
                    sample.source_start_ns,
                    sample.sink_start_ns,
                    sample.sink_end_ns,
                    sample.latency_ns,
                )
            )
    return rows


def _validate_trace(events, compute_idx):
    """Report trace health and reject missing or ambiguous correlation data."""
    if not compute_idx:
        return (
            "No ':compute' ranges were found in the OCL domain. End-to-end latency "
            'is measured between operator compute cycles, so the compute annotation '
            'is required in that domain.'
        )
    overlapping_ops = _overlapping_compute_operators(compute_idx)
    if overlapping_ops:
        return (
            'Overlapping or touching OCL compute ranges were found for the same operator name: '
            + ', '.join(overlapping_ops)
        )

    (
        missing_payloads,
        conflicting_emits,
        emits_outside_compute,
        receives_outside_compute,
        receive_without_emit,
    ) = _trace_health(events, compute_idx)
    health_warnings = []
    if emits_outside_compute:
        health_warnings.append(
            f'{emits_outside_compute} emit markers occurred outside operator callbacks'
        )
    if receives_outside_compute:
        health_warnings.append(f'receives outside operator callbacks={receives_outside_compute}')
    if receive_without_emit:
        health_warnings.append(
            f'receive identities with no captured emit={receive_without_emit} '
            '(possible leading capture boundary, not proof of a drop)'
        )
    if health_warnings:
        sys.stderr.write('WARNING: OCL trace health: ' + ', '.join(health_warnings) + '.\n')

    if missing_payloads:
        marker_names = ', '.join(sorted(set(missing_payloads)))
        return (
            f'{len(missing_payloads)} OCL emit/receive marker(s) have no decodable '
            f'message identity: {marker_names}'
        )
    if conflicting_emits:
        identity_preview = ', '.join(repr(identity) for identity in conflicting_emits[:3])
        if len(conflicting_emits) <= 3:
            identity_detail = identity_preview
        else:
            identity_detail = f'first three: {identity_preview}'
        return (
            f'{len(conflicting_emits)} OCL message identity value(s) were emitted more '
            f'than once: {identity_detail}. Each identity must name one source-port '
            'emit. This can happen when a marker is recorded before a failed '
            'emit is retried.'
        )

    return None


def _resolve_port_path(port_path, port_path_file):
    """Read one UTF-8 selector without interpreting its NVTX backslash escapes."""
    if port_path is not None and port_path_file is not None:
        raise ValueError('Specify either port-path or port-path-file, not both.')
    if port_path_file is None:
        return port_path
    try:
        with open(port_path_file, encoding='utf-8') as selector_file:
            path = selector_file.read().removesuffix('\n')
    except (OSError, UnicodeError) as error:
        raise ValueError(f'Unable to read port path file {port_path_file!r}: {error}') from error
    if not path.strip() or '\n' in path or '\r' in path:
        raise ValueError(
            f'Port path file {port_path_file!r} must contain exactly one non-empty line.'
        )
    return path


# ── The nsys stats report ────────────────────────────────────────────────────


class Ocl(nsysstats.StatsReport):
    """Nsight Systems report for Holoscan operator-centric latency paths."""

    display_name = 'Holoscan Operator-Centric Latency'
    # pylint: disable-next=line-too-long
    usage = """{SCRIPT}[:port-path=<path>|:port-path-file=<file>][:responses=all|first][:warmup=<N>][:window=<N>][:window-unit=messages|ns][:expected-period-ns=<N>][:latency-budget-ns=<N>][:view=summary|samples] -- {DISPLAY_NAME}

    Per-path end-to-end latency for a Holoscan (HSDK 5) NVTX capture. One table
    row per exact port path. A fan-out yields one row per downstream branch.

    REQUIRED pre-export -- this report reads an NVTX extended payload, which is
    only present when the SQLite export includes blobs:

        nsys export --type sqlite --include-blobs=true --force-overwrite=true \\
            -o <report>.sqlite <report>.nsys-rep
        nsys stats --report-dir <this_dir> -r {SCRIPT} <report>.nsys-rep

    The SQLite that `nsys stats` generates on its own defaults to
    --include-blobs=false and can never work; this report stops and prints the
    command above rather than emitting an empty table.

    Shape the output with nsys' own options:
        -f <fmt>       column|table|csv|tsv|json|hdoc|htable (default: console table)
        -o <path>      write to a file; nsys names it <path>_{SCRIPT}.<ext>
        --timeunit U   nsec (default) | usec | msec | sec -- rescales time columns

    Table columns (one row per exact port path):
        Path       "src_op -> dst_op -> ... -> sink_op"
        Port Path  complete output/input port lineage for the row
                   or '<source port> -> ?' when an enumerated operator path
                   has no completed source-to-sink sample, or when a source
                   output port is never received
                   or NULL when no first hop was observed inside compute ranges
        Responses  trace-visible responses from eligible sources before
                   all/first response selection
        Eligible Sources
                   distinct source messages emitted inside the requested window
        Matched Sources
                   distinct source message identities that completed this path
        Unmatched Sources
                   eligible sources with no visible response on this exact path
        Excluded Sources
                   source messages emitted before or after the requested window
        Source Callbacks, Source Callback Hz
                   source operator callbacks in the time window, or spanning
                   the first through last emit in a message window; rows with
                   the same source operator share these operator-level values
        Emit Hz
                   eligible source-port message rate from emit timestamps
        Sink Completions, Sink Completion Hz
                   distinct sink callbacks reached by eligible sources, before
                   all/first response selection
        Selected   samples used for the latency statistics
        Avg, Min, Max, P50, P90, P99, P99.9, StdDev
                   durations, raw ns, rescaled by --timeunit
        Sink Period Error P99.9
                   shown with expected-period-ns; percentile of absolute interval
                   error over the distinct sink completions, not path latency
        Latency Budget Exceedances
                   shown with latency-budget-ns; selected latency strictly over budget

    With view=samples, Source Start, Sink Start, and Sink End are absolute trace
    positions; Latency is an elapsed duration. All are raw ns before --timeunit
    scaling.

    For port-path selection, separate the displayed port labels with '->', for
    example 'left/stage/source.out->right/stage/sink.in'. The displayed ' -> '
    separator is also accepted. Keep graph slashes and backslash escapes intact.
    The former '/' port separator is no longer accepted: it is now part of graph
    paths. Quote the report argument in the shell to protect '>' and backslashes.

    For arbitrary names or long paths, use port-path-file=path.txt instead.
    The UTF-8 file contains one exact Port Path, with an optional final newline;
    its backslashes and spaces are preserved. Relative filenames resolve from
    the current working directory. Use a short filename without ':' or ',':
    nsys parses those delimiters in report arguments and includes arguments in
    generated output filenames. The path inside the file bypasses that parsing.
    port-path and port-path-file are mutually exclusive. Omit both to show all
    reconstructed paths.

    responses=first keeps the earliest completed sink callback for each distinct
    source message identity on each exact port path. The source unit is a message
    identity, not an inference invocation.

    view=samples emits one row per selected response with its source identity,
    sequence, source start, sink start/end, and start-to-end latency. Use
    view=summary for source accounting when the selected window is empty.
    Period-error and latency-budget aggregates appear only in summary view.

    warmup skips source messages before measurement. window keeps the next
    source messages when window-unit=messages. With window-unit=ns, the common
    origin is the callback that emitted the earliest traced source message
    across all paths. Measurement starts at origin + warmup and ends at
    start + window.
    Bounds are half-open, and responses from eligible sources may finish after
    the window. A time window counts all source callbacks inside its explicit
    bounds, including callbacks that emit nothing. A message window has no
    later time bound, so its callback count ends at its last emit.
    Eligible Sources always equals Matched Sources plus Unmatched Sources. A
    receive whose emit is outside the capture is not counted in any of these
    columns; detecting leading capture truncation is a separate check.

    Trace-health failures stop the report for multiple OCL payload-schema
    registrations, missing message identities, repeated emit identities,
    overlapping or touching same-name compute ranges, or negative path latency. One stderr
    warning summarizes non-zero counts for emit/receive marks outside operator
    callbacks and receives whose emit is outside the capture. A warning also
    names selected sources that begin after a common time window ends. Preserve
    stderr when writing formatted output; these warnings are not table columns.

    End-to-end is start_to_end: source ':compute' start -> sink ':compute' end,
    for the compute cycles chained by one message. Messages are correlated by
    the structured payload (source_identity, sequence) carried on the
    '<op>.<port>:emit' and '<op>.<port>:receive' marks. Receive-to-emit causality
    is inferred when both marks occur inside the same operator ':compute' range;
    emit/receive marks are only ever attributed to a ':compute' range of the SAME
    operator, so parallel branches cannot cross-assign.
"""

    _arg_opts = [
        [['port-path'], {'type': str, 'default': None}],
        [['port-path-file'], {'type': str, 'default': None}],
        [['responses'], {'choices': ('all', 'first'), 'default': 'all'}],
        [['warmup'], {'type': int, 'default': 0}],
        [['window'], {'type': int, 'default': None}],
        [['window-unit'], {'choices': ('messages', 'ns'), 'default': 'messages'}],
        [['expected-period-ns'], {'type': int, 'default': None}],
        [['latency-budget-ns'], {'type': int, 'default': None}],
        [['view'], {'choices': ('summary', 'samples'), 'default': 'summary'}],
    ]

    table_checks = {
        'StringIds': '{DBFILE} file does not contain StringIds table.',
        'NVTX_EVENTS': '{DBFILE} does not contain NV Tools Extension (NVTX) data.',
    }

    def _nvtx_columns(self):
        """Return NVTX_EVENTS column names."""
        return {row[1] for row in self._dbcon.execute('PRAGMA table_info(NVTX_EVENTS)')}

    def _blob_export_error(self):
        """The one remedy message for a SQLite exported without NVTX blobs.

        Built from the actual database basename and this script's directory so
        the printed commands are literally copy-pasteable.

        --force-overwrite=true matters: if `nsys stats` already auto-generated a
        blob-less <report>.sqlite, that file persists and is reused (only
        --force-export re-exports it), and a plain `nsys export -o` refuses to
        overwrite it. Without the flag the user is stuck in a loop where nothing
        they do changes the result.
        """
        stem = self._dbfile_basename
        report_dir = os.path.dirname(os.path.abspath(__file__))
        return (
            f'No NVTX extended payload data found in {stem}.sqlite.\n'
            '\n'
            'This report reads the OCL correlation key from an NVTX extended payload,\n'
            'which is only present when the SQLite is exported with --include-blobs.\n'
            'The export that `nsys stats` generates on its own never has it.\n'
            '\n'
            'Re-export with blobs, overwriting any existing export, using the same\n'
            'basename as the .nsys-rep:\n'
            '\n'
            f'    nsys export --type sqlite --include-blobs=true --force-overwrite=true \\\n'
            f'        -o {stem}.sqlite {stem}.nsys-rep\n'
            '\n'
            'Then re-run:\n'
            '\n'
            f'    nsys stats --report-dir {report_dir} -r data_flow_tracking {stem}.nsys-rep\n'
            '\n'
            'To use an export under a different name, pass it directly:\n'
            '\n'
            f'    nsys stats --report-dir {report_dir} -r data_flow_tracking '
            '--sqlite <file>.sqlite\n'
        )

    def _load_events(self):
        """Read NVTX rows -> ([(name, start, end, message_key)], error).

        Rejects any database lacking decodable structured payloads before doing
        any analysis, so a blob-less export can never produce a misleading table.
        """
        if 'binaryData' not in self._nvtx_columns():
            return None, self._blob_export_error()

        payload_tables = {
            'NVTX_PAYLOAD_SCHEMAS',
            'NVTX_PAYLOAD_SCHEMA_ENTRIES',
        }
        found_tables = {
            row[0]
            for row in self._dbcon.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if not payload_tables <= found_tables:
            return None, self._blob_export_error()

        message_schema_registrations = self._dbcon.execute(
            'SELECT COUNT(*) FROM NVTX_PAYLOAD_SCHEMAS WHERE name = ?',
            (OCL_MESSAGE_PAYLOAD_SCHEMA,),
        ).fetchone()[0]
        if message_schema_registrations > 1:
            return None, (
                f'{message_schema_registrations} {OCL_MESSAGE_PAYLOAD_SCHEMA!r} schema '
                'registrations were found. This report analyzes one OCL registration; '
                'profile each process separately.'
            )

        message_layout = _payload_layout(
            self._dbcon,
            OCL_MESSAGE_PAYLOAD_SCHEMA,
            {
                'source_identity': REGISTERED_STRING_HANDLE_TYPE,
                'sequence': UINT64_TYPE,
            },
        )
        if message_layout is None:
            return None, (
                f'The trace does not contain the expected {OCL_MESSAGE_PAYLOAD_SCHEMA!r} schema.'
            )
        message_domain_id = message_layout[0]
        string_ids = dict(self._dbcon.execute('SELECT id, value FROM StringIds'))
        # HSDK records its OCL compute and message markers in the payload schema's domain.
        cur = self._dbcon.execute(
            """
            SELECT COALESCE(e.text, s.value) AS name,
                   e.start, e.end, e.binaryData
            FROM NVTX_EVENTS AS e
            LEFT JOIN StringIds AS s ON e.textId == s.id
            WHERE e.domainId = ?
            ORDER BY e.start
            """,
            (message_domain_id,),
        )

        events = []
        saw_ocl_marker = False
        saw_message_mark = False
        saw_payload = False
        for name, start_ns, end_ns, binary_data in cur.fetchall():
            if not name:
                continue
            is_message = name.endswith(EMIT_SUFFIX) or name.endswith(RECEIVE_SUFFIX)
            if is_message or name.endswith(COMPUTE_SUFFIX):
                saw_ocl_marker = True
            corr = None
            if is_message:
                saw_message_mark = True
                corr = _message_payload(binary_data, string_ids, message_layout)
                if corr is not None:
                    saw_payload = True
            events.append((name, start_ns, end_ns, corr))

        if not events:
            return None, 'No NVTX events were found in the OCL domain.'

        if not saw_ocl_marker:
            return None, (
                "No Holoscan OCL markers (':compute', ':emit', ':receive') were found "
                'in this capture. Profile an application built with OCL NVTX '
                'annotations enabled, then re-run this report.'
            )

        # Marks are present but carry no payload -> the export dropped the blobs.
        if saw_message_mark and not saw_payload:
            return None, self._blob_export_error()

        if not saw_message_mark:
            return None, (
                "Only ':compute' ranges were found -- no ':emit' or ':receive' marks. "
                'Per-path latency needs the message marks to correlate operators.'
            )

        return events, None

    def _build_rows(self):
        """Build the per-path table (raw-ns durations), or (None, error)."""
        try:
            port_path = _resolve_port_path(
                self.parsed_args.port_path, self.parsed_args.port_path_file
            )
        except ValueError as error:
            return None, str(error)
        events, err = self._load_events()
        if err is not None:
            return None, err

        compute_idx = build_range_index(events, COMPUTE_SUFFIX)
        err = _validate_trace(events, compute_idx)
        if err is not None:
            return None, err

        adj, sources, sinks, all_ops = build_topology(events)
        if not all_ops:
            return None, (
                'OCL emit/receive marks were decoded, but no emit key was ever matched '
                'to a receive key, so no message path could be reconstructed. This '
                'usually means the capture covers only part of the graph.'
            )

        records, receive_lookup, emit_port_lookup = build_compute_records(
            events, compute_idx, all_ops
        )

        paths = enumerate_paths(adj, sources, sinks)
        if not paths:
            return None, (
                'A message graph was reconstructed but it contains no source-to-sink '
                'path (every operator both sends and receives, so the graph is cyclic).'
            )

        try:
            all_series = path_series(paths, records, receive_lookup, emit_port_lookup, compute_idx)
        except ValueError as error:
            return None, str(error)
        try:
            series = select_series(
                all_series,
                port_path=port_path,
                responses=self.parsed_args.responses,
                warmup=self.parsed_args.warmup,
                window=self.parsed_args.window,
                window_unit=self.parsed_args.window_unit,
                expected_period_ns=self.parsed_args.expected_period_ns,
                latency_budget_ns=self.parsed_args.latency_budget_ns,
            )
        except ValueError as error:
            return None, str(error)
        if port_path is not None and not series:
            available = [
                entry['port_path']
                for entry in all_series
                if entry['port_path'] is not None and not entry['port_path'].endswith(' -> ?')
            ]
            available_paths = ', '.join(sorted(available)) or 'none'
            return None, (
                f'No Port Path exactly matched {port_path!r}. '
                f'Available completed paths: {available_paths}.'
            )
        series.sort(key=lambda p: (p['path_str'], p['port_path'] or ''))

        if self.parsed_args.view == 'samples':
            return sample_rows(series), None

        rows = []
        for entry in series:
            rows.append(
                _row(
                    len(rows),
                    entry['path_str'],
                    entry['port_path'],
                    response_count=entry['response_count'],
                    eligible_source_count=entry['eligible_source_count'],
                    source_identity_count=entry['source_identity_count'],
                    unmatched_source_count=entry['unmatched_source_count'],
                    excluded_source_count=entry['excluded_source_count'],
                    source_callback_count=entry['source_callback_count'],
                    source_callback_hz=entry['source_callback_hz'],
                    emit_hz=entry['emit_hz'],
                    sink_completion_count=entry['sink_completion_count'],
                    sink_completion_hz=entry['sink_completion_hz'],
                    stats=_stat_cols([sample.latency_ns for sample in entry['samples']]),
                    sink_period_error_p99_9_ns=entry.get('sink_period_error_p99_9_ns'),
                    latency_budget_exceedance_count=entry.get('latency_budget_exceedance_count'),
                )
            )
        return rows, None

    def start_query(self):
        rows, err = self._build_rows()
        if err is not None:
            return err

        # Materialize the table in the temp DB, then let the base class run
        # `self.query` against it so nsys' own -f / --output / --timeunit shape the
        # output. (Temp writes work even though the main DB is opened read-only.)
        con = self._dbcon
        con.execute('DROP TABLE IF EXISTS temp.ocl')
        if self.parsed_args.view == 'samples':
            con.execute("""
                CREATE TEMP TABLE ocl (
                    seq INTEGER, path TEXT, port_path TEXT,
                    source_identity TEXT, source_sequence INTEGER,
                    source_start INTEGER, sink_start INTEGER, sink_end INTEGER,
                    latency INTEGER
                )
            """)
            con.executemany('INSERT INTO ocl VALUES (?,?,?,?,?,?,?,?,?)', rows)
            self.query = """
                SELECT path AS "Path",
                       port_path AS "Port Path",
                       source_identity AS "Source Identity",
                       source_sequence AS "Source Sequence",
                       source_start AS "Source Start:ts_ns",
                       sink_start AS "Sink Start:ts_ns",
                       sink_end AS "Sink End:ts_ns",
                       latency AS "Latency:dur_ns"
                FROM temp.ocl ORDER BY seq
            """
            return super().start_query()

        con.execute("""
            CREATE TEMP TABLE ocl (
                seq INTEGER, path TEXT, port_path TEXT,
                response_count INTEGER, eligible_source_count INTEGER,
                source_identity_count INTEGER, unmatched_source_count INTEGER,
                excluded_source_count INTEGER,
                source_callback_count INTEGER, source_callback_hz REAL,
                emit_hz REAL, sink_completion_count INTEGER,
                sink_completion_hz REAL,
                selected_count INTEGER, avg REAL, "min" REAL, "max" REAL,
                p50 REAL, p90 REAL, p99 REAL, p99_9 REAL, std REAL,
                sink_period_error_p99_9 REAL, latency_budget_exceedance_count INTEGER
            )
        """)
        con.executemany(
            'INSERT INTO ocl VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows
        )

        # Duration columns carry the ':dur_ns' type suffix (raw ns): nsys strips the
        # suffix from the header and rescales them per --timeunit. nsys' --timeunit
        # DEFAULTS TO NANOSECONDS, so with no flag the durations print as raw ns;
        # pass --timeunit=usec|msec|sec to rescale. Rates and counts stay untyped.
        period_projection = ''
        if self.parsed_args.expected_period_ns is not None:
            period_projection = (
                ',\n                   sink_period_error_p99_9 AS "Sink Period Error P99.9:dur_ns"'
            )
        budget_projection = ''
        if self.parsed_args.latency_budget_ns is not None:
            budget_projection = (
                ',\n                   latency_budget_exceedance_count '
                'AS "Latency Budget Exceedances"'
            )
        self.query = f"""
            SELECT path  AS "Path",
                   port_path AS "Port Path",
                   response_count AS "Responses",
                   eligible_source_count AS "Eligible Sources",
                   source_identity_count AS "Matched Sources",
                   unmatched_source_count AS "Unmatched Sources",
                   excluded_source_count AS "Excluded Sources",
                   source_callback_count AS "Source Callbacks",
                   source_callback_hz AS "Source Callback Hz",
                   emit_hz AS "Emit Hz",
                   sink_completion_count AS "Sink Completions",
                   sink_completion_hz AS "Sink Completion Hz",
                   selected_count AS "Selected",
                   avg   AS "Avg:dur_ns",   "min" AS "Min:dur_ns",   "max" AS "Max:dur_ns",
                   p50   AS "P50:dur_ns",   p90   AS "P90:dur_ns",   p99   AS "P99:dur_ns",
                   p99_9 AS "P99.9:dur_ns", std   AS "StdDev:dur_ns"
                   {period_projection}{budget_projection}
            FROM temp.ocl ORDER BY seq
        """
        return super().start_query()


if __name__ == '__main__':
    Ocl.Main()
