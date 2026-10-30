# Data flow tracking report

`data_flow_tracking.py` is an `nsys stats` report that reconstructs source-to-sink port paths from
Holoscan SDK 5 NVTX operator markers and reports end-to-end latency and message rates. It requires
only Nsight Systems and its bundled `nsysstats` module; no Python dependencies are needed.

Export the capture to SQLite with binary payloads:

```bash
nsys export --type sqlite \
  --include-blobs=true --force-overwrite=true \
  -o <capture_path>.sqlite \
  <capture_path>.nsys-rep
```

Generate the report:

```bash
nsys stats \
  --report-dir tools/profiling \
  --report data_flow_tracking \
  --timeunit msec \
  <capture_path>.sqlite
```

See the report options with:

```bash
nsys stats --report-dir tools/profiling --help-reports=data_flow_tracking
```

For the full workflow, see [Profile an Application](../../docs/how-to/profile_an_application.mdx)
and the [GR00T benchmark](../../applications/gr00t_n1_7_wbc_g1/BENCHMARK.md).

## Measurement boundaries

Path latency runs from source operator compute start to sink operator compute end,
including time between operators. The report gives whole-path statistics, not a
per-operator latency breakdown. Application metrics that end at sink start use a
different boundary and cannot be compared as exact parity.

Transport is correlated by `(source_identity, sequence)`. Receive-to-emit continuation
is inferred within the same operator compute callback. Retained state, fan-in,
conditional forwarding, and reused messages can make causality ambiguous. Validation
taps and fan-in can produce overlapping rows; do not add their counts together.

Unmatched sources mean no visible response on that path, not proven drops. Keep
incomplete paths and zero-sample rows. Missing statistics are unavailable, not zero.
Check command success and saved stderr before interpreting results. Trace-health
warnings are not stored in summary columns; preserve them with the measurements.

## Measurement options

See [Customize the report](../../docs/how-to/profile_an_application.mdx#customize-the-report)
for option syntax, defaults and examples.

The source unit is a message identity, not an inference invocation. Windows select
source messages; their responses may finish after the window if they remain within
the capture. For `window-unit=ns`, measurement starts at `origin + warmup` and ends
at `start + window`, where the origin is the callback that emitted the earliest
traced source message across all paths.

Use latency and period targets supplied by the user or established in the
application configuration, and record their provenance. Do not invent targets.

## Exact path selection

Follow [Measure a specific path](../../docs/how-to/profile_an_application.mdx#measure-a-specific-path)
for inline and file-based selectors, escaping and Nsight argument restrictions.
Both `port-path` and `port-path-file` select rows after path reconstruction;
neither reduces the work needed to reconstruct a large graph.

## Sample export

`view=samples` exports each selected response's source identity, sequence, source
start, sink start/end, and latency. Samples omit source-accounting and
period-error/budget aggregates, which remain in the summary.

```bash
nsys stats \
  --report-dir tools/profiling \
  --report 'data_flow_tracking:view=samples' \
  --format csv \
  --timeunit nsec \
  --output build/benchmarks/<profile_name>_samples \
  build/benchmarks/<profile_name>.sqlite \
  2> build/benchmarks/<profile_name>_samples.stderr
```

Append the requested summary's exact path, window, and response options to the
`--report` argument. Use `nsec` to preserve raw trace timestamp units. Link the
actual generated filenames; Nsight includes report arguments in output filenames.

## Output columns

Typical JSON row:

```json
{
  "Path": "mock_joint_state_source -> agile_wbc -> mock_joint_command_sink",
  "Port Path": "mock_joint_state_source.out_joint_state -> agile_wbc.in_joint_state -> agile_wbc.out_joint_command -> mock_joint_command_sink.in_joint_command",
  "Responses": 20000,
  "Eligible Sources": 20000,
  "Matched Sources": 20000,
  "Unmatched Sources": 0,
  "Excluded Sources": 1000,
  "Source Callbacks": 20000,
  "Source Callback Hz": 50.0,
  "Emit Hz": 50.0,
  "Sink Completions": 20000,
  "Sink Completion Hz": 50.0,
  "Selected": 20000,
  "Avg (ms)": 0.2414,
  "Min (ms)": 0.0547,
  "Max (ms)": 1.5323,
  "P50 (ms)": 0.2291,
  "P90 (ms)": 0.2954,
  "P99 (ms)": 0.4592,
  "P99.9 (ms)": 1.2599,
  "StdDev (ms)": 0.0823,
  "Sink Period Error P99.9 (ms)": 0.2,
  "Latency Budget Exceedances": 0
}
```

Column definitions:

- `Path`: reconstructed operator chain, grouped by operator names for readability.
- `Port Path`: exact output/input port lineage used for this row and for selection.
- `Responses`: trace-visible responses from eligible sources before `all`/`first` selection.
- `Eligible Sources`: distinct source messages inside the requested window.
- `Matched Sources`: eligible source identities with at least one visible response on this path.
- `Unmatched Sources`: eligible sources with no visible response on this path. This does not by
  itself prove a dropped message.
- `Excluded Sources`: known source messages outside the requested window.
- `Source Callbacks`, `Source Callback Hz`: source operator runs in the requested time window, or
  from the first through last eligible source-message callback in a message window. This includes
  callbacks that emit nothing. A time window keeps its explicit end; a message window ends its
  callback count at the last selected emit because it has no later time boundary. Rows sharing a
  source operator share these operator-level values; emit counts and rates remain per output port.
- `Emit Hz`: rate of eligible source-port emit events. `Eligible Sources` is the emit
  count.
- `Sink Completions`, `Sink Completion Hz`: distinct sink callbacks reached by eligible source
  messages, before response selection.
- `Selected`: responses used for latency statistics.
- `Avg`, `Min`, `Max`, `P50`, `P90`, `P99`, `P99.9`, `StdDev`: path latency stats.
- `Sink Period Error P99.9`: percentile of absolute interval error over the same distinct sink
  completions, not a percentile of path latency. It appears only when `expected-period-ns` is
  supplied.
- `Latency Budget Exceedances`: selected responses whose path latency is strictly over the supplied
  budget. It appears only when `latency-budget-ns` is supplied.

Rates use `(N - 1) / (last - first)` and are unavailable for fewer than two events or a zero span.

An incomplete path remains visible as `<source port> -> ?` with zero selected samples. This also
includes a source output port whose messages are never received. Use the eligible, matched, and
unmatched counts to distinguish an empty response set from an empty source window. Receives whose
source emit is outside the capture are not included in those counts.

Graphs with retained state, fan-in, or conditional forwarding can produce several overlapping grouped
`Path` rows in one report. For example, a GR00T+WBC graph may show rows with and without an
intermediate combiner stage. Interpret each row as one reconstructed completed operator path; multiple
rows sharing some operators are expected and are not automatically a report bug.

## Marker format

The Holoscan SDK runtime emits these NVTX data-flow annotations in the `holoscan` domain:

```text
<operator>:compute             range
<operator>.<output-port>:emit   instantaneous mark
<operator>.<input-port>:receive instantaneous mark
```

For nested graphs, operator labels include the complete root-to-leaf graph instance path, omitting
the root graph's own name. Subgraph segments and the operator name are separated with `/`:

```text
outer/inner/operator:compute
outer/inner/operator.port:emit
outer/inner/operator.port:receive
```

The operator-to-port separator is `.`. Literal `\`, `.`, `:`, and `/` characters inside
subgraph, operator, or port names are escaped with a preceding `\`. Preserve these escapes in
reported labels and path selectors. For example, `left\/stage/camera\.front.rgb\.out:emit`
belongs to operator `left\/stage/camera\.front`; the escaped slash is part of one graph name,
distinct from two nested graphs named `left/stage`.

The `:emit` and `:receive` marks carry a structured NVTX payload named:

```text
Holoscan OCL Message Payload
```

The report uses these fields to correlate messages:

```text
source_identity
sequence
```

The report correlates message transport by matching `(source_identity, sequence)` from emit marks to
receive marks. An emit marks an accepted send or one whose delivery may already have started;
a cleanly rejected send has no emit marker. A matching receive can precede its emit in the trace,
so correlate by payload identity rather than timestamp order.

## Capture and report errors

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| No Holoscan data-flow markers found | Profile did not capture NVTX annotations | Profile with `-t nvtx` |
| `Only ':compute' ranges were found` | No message transport was captured | Capture a window in which operators emit and receive messages |
| `structured payloads were not decoded` / blob error | SQLite was exported without payload BLOBs | Re-run `nsys export --include-blobs=true` |
| Some message markers have no decodable identity | Partial or corrupt marker payloads | Treat the capture as invalid; inspect the marker names in the error and re-export if needed |
| More than one message payload-schema registration | The capture may contain more than one Holoscan SDK process or an ambiguous duplicate registration | Profile each process separately |
| A message identity was emitted more than once | Duplicate or corrupt message identities in the trace | Treat the capture as invalid and inspect the identities in the error |
| Overlapping or touching compute ranges for one operator | Duplicate operator names, an unsupported multi-process capture, or a marker exactly on a shared callback boundary | Use unique operator names and do not claim cross-process results from this trace; touching ranges have not occurred in validated captures |
| Negative path latency | Invalid timestamp order or incorrect path attribution | Treat the capture as invalid and inspect the named path and its markers |
| `no emit key was ever matched to a receive key` | Capture covers only part of the graph, payload schema mismatch, or source/sequence changed | Inspect NVTX Events View for matching `:emit` and `:receive`; verify `source_identity` and `sequence` |
| Counts differ from app CSV | Different source unit, warmup/window, response selection, path, or timing boundary | Select the exact `Port Path`; align `warmup`, `window`, and `responses`; compare eligible, matched, and selected counts |
| Report latency is higher than source-sink CSV | The report measures source compute start to sink compute end | Compare as rough parity, not exact parity |
| Trace-health warning on emits outside callbacks | Capture boundary or malformed operator markers | Verify that the emit has its own operator's compute range |
| Trace-health warning on receives without captured emits | Capture began after those source emits | Treat this as a leading capture boundary, not proof of a dropped message |
| A selected source begins after the common time window ends | That source starts later than the requested interval | Check the shared time origin and window settings; preserve the warning and empty row |

## Inspect markers manually in Nsight Systems

Open the `.nsys-rep` in Nsight Systems. To view data-flow annotations:

1. Find the `holoscan` NVTX domain.
2. Right-click the domain outside the thread dropdown.
3. Select **Show in Events View**.
4. Search for `:compute`, `:emit`, `:receive`, an operator name, or a port name.

Use Events View to confirm that expected emit/receive/compute labels exist before debugging the
Python report.

## Testing

From the repository root, run:

```bash
python3 -m pytest tools/profiling/tests
```

The test stubs `nsysstats`; only pytest is required.
