# Nsight diagnostic queries

Use these recipes when a CUDA trace needs closer inspection. Capture and export
using [Profile an Application](../../docs/how-to/profile_an_application.mdx)
and the [CUDA profiling guidance](../../docs/how-to/optimize_an_application.mdx#profile-your-application).
NVTX-only captures do not contain the CUDA activity queried here.

Open the export read-only with `sqlite3 -readonly <capture.sqlite>`. Inspect
`.tables` and `.schema <table>` first: tables and columns depend on the Nsight
version and captured activity. Consult NVIDIA's
[SQLite schema reference](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
for field definitions. Missing tables are not evidence that no waits occurred.

Timestamps below are nanoseconds; durations are converted to milliseconds.
Scope stream analysis by process, device, and context as well as stream ID.

## Long CUDA API calls

```sql
SELECT r.globalTid, s.value AS api, r.start,
       (r.end - r.start) / 1e6 AS duration_ms
FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
JOIN StringIds AS s ON s.id = r.nameId
ORDER BY duration_ms DESC
LIMIT 30;
```

Inspect allocation, transfer, and synchronization calls around the critical
workload's outliers. A long call shows where host time was spent; inspect the
timeline and dependencies to determine why. Calls on other threads ending at
the same time can guide investigation but do not establish mutex ownership.

## Synchronization by stream

```sql
SELECT globalPid, deviceId, contextId, streamId, syncType,
       COUNT(*) AS count, MAX(end - start) / 1e6 AS max_ms,
       AVG(end - start) / 1e6 AS mean_ms
FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
WHERE end - start > 5000000
GROUP BY globalPid, deviceId, contextId, streamId, syncType
ORDER BY max_ms DESC;
```

Adjust the example 5 ms threshold to the workload's latency budget. Inspect
`ENUM_CUPTI_SYNC_TYPE` to interpret `syncType`; some synchronization events are
not tied to a particular stream.

## Launch-return to kernel-start gaps

```sql
SELECT k.globalPid, k.deviceId, k.contextId, k.streamId,
       s.value AS kernel, r.start AS launch_start,
       (k.start - r.end) / 1e6 AS gap_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
  ON k.correlationId = r.correlationId
 AND (k.globalPid >> 24) = (r.globalTid >> 24)
JOIN StringIds AS s ON s.id = k.shortName
ORDER BY gap_ms DESC
LIMIT 30;
```

The process check prevents matching equal correlation IDs from different
processes. For driver API launches, inspect `CUPTI_ACTIVITY_KIND_DRIVER` instead;
CUDA Graph correlation also depends on capture settings and exported graph data.
A kernel can begin before its launch call returns, producing a negative gap.
A positive gap can include stream dependencies and queued work; it does not by
itself identify SM contention, shared work queues, or front-end scheduling.

## Kernel activity by stream

```sql
SELECT k.globalPid, k.deviceId, k.contextId, k.streamId,
       s.value AS kernel, COUNT(*) AS launches
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON s.id = k.shortName
GROUP BY k.globalPid, k.deviceId, k.contextId, k.streamId, s.value
ORDER BY launches DESC;
```

Use `TARGET_INFO_CUDA_STREAM` and `ENUM_CUPTI_STREAM_TYPE`, when present, to
identify legacy-default-stream activity. Do not assume stream ID zero denotes
the default stream in the export. Map activity to operator NVTX ranges before
changing stream use; counts alone do not establish interference.
