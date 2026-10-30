---
name: data-flow-profiling
description: >
  Profile data flow in Isaac OS / Holoscan applications using Nsight Systems.
  Use for application profiling focused on end-to-end (E2E) source-to-sink latency, message
  counts and rates, or exporting path statistics and per-response samples
  from .nsys-rep/.sqlite captures. Also use for general "profile my application"
  requests when no specific profiling domain is named. For CPU sampling and GPU
  kernel profiling, use a normal Nsight workflow.
license: Apache-2.0
metadata:
  author: "Connor Smith <cosmith@nvidia.com>"
---

# Data Flow Profiling

Follow [Profile an Application](../../docs/how-to/profile_an_application.mdx)
for capture, export, and report commands. Use the
[data-flow report reference](../../tools/profiling/README.md) for options,
column definitions, measurement boundaries, and errors.
Performance diagnosis, optimization, and unrequested follow-up experiments are
outside this skill's scope.

## Choose the starting point

- **New profile:** identify the application, run mode, configuration, and requested
  measurements, then follow the capture workflow.
- **Existing `.nsys-rep`:** start at SQLite export. Reuse a valid export with blobs
  when available.
- **Existing `.sqlite`:** start at report generation. Missing blobs require the
  original capture for re-export; report a missing capture as a blocker.
- **Existing report:** surface available rows, columns, settings, and artifacts;
  generate additional views only as needed. Do not rerun the app to display results.

Infer the application and settings from the conversation, metadata, and available
artifacts. For project or mode discovery, follow
[Build and run](../../docs/how-to/build_and_run.mdx) and the
[CLI reference](../../docs/reference/cli.mdx).
Ask only for missing information needed to capture or select measurements.

## Choose measurement settings

- By default, report **all reconstructed port paths**, with `responses=all`, no
  warmup exclusion, and the full captured source window. State these settings.
  Apply requested filters explicitly and keep the unfiltered summary available.
- Plan the capture to cover startup, the measurement interval, and completion of
  responses from that interval. For steady-state or shared windows and supplied
  targets, use the [measurement options](../../tools/profiling/README.md#measurement-options).
- For a new data-flow capture, use the guide's low-overhead NVTX tracing settings.
  Preserve additional tracing when requested and record the flags used.

## Capture and report

1. Run the needed capture/export/report steps from the guide, following repository
   cache-mount instructions for builds. Use the application's container or a host
   with compatible Nsight Systems for export and reporting.
2. Prefer `build/benchmarks/` for artifacts, with a distinct output prefix for each
   capture to retain. Use the app's supported run limit or stop it normally, then
   wait for Nsight to finish writing the capture before exporting. Overwrite only
   when intentionally replacing an artifact; otherwise omit `--force-overwrite=true`
   from the guide's commands.
3. Export SQLite with blobs. When replacing a blob-less export, use the guide's
   export command with `--force-overwrite=true`.
4. Generate the complete machine-readable summary without path selectors and save
   stderr. For a requested subset, produce an additional summary with its own prefix;
   copy exact paths from the complete report, preserving nested names and escapes.
5. Export samples when requested using the
   [sample export recipe](../../tools/profiling/README.md#sample-export).
   Match the summary's path, window, and response settings, and retain the summary
   for source accounting and aggregate metrics.

## Check results before reporting

- Check command success and read saved stderr. Surface failures rather than treating
  partial output as valid statistics; preserve warnings with the results.
- Apply the reference's [measurement boundaries](../../tools/profiling/README.md#measurement-boundaries).
  Do not equate sink-end latency with metrics ending at sink start. Preserve
  causality limits for retained state and fan-in when interpreting results.
- Identify the requested output port. Do not add counts from overlapping rows, label
  unmatched sources as proven drops, or turn unavailable statistics into zero.
  Keep incomplete paths and zero-sample rows.

## Surface the results

- Lead with the requested paths, latency statistics, sample count, measurement
  settings, and relevant warnings. Include counts, rates, and target aggregates
  when requested. Label any displayed subset and state the total path count.
- Link the capture, SQLite export, complete machine-readable summary, saved stderr,
  and any selected summaries or sample exports. Preserve exact port paths, all
  available columns, empty rows, and unavailable values in those artifacts.
- Save app/mode/configuration, capture duration, available code/tool versions,
  exact commands, tracing flags, units, and selection settings with the artifacts;
  mark unavailable provenance as unknown. A custom HTML report is optional, not a
  default deliverable.

## Links

The links in this skill are paths in an Isaac OS checkout. Without a checkout, find
docs pages in [llms.txt](https://docs.nvidia.com/isaac-os/llms.txt) for your Isaac OS
version, and other files in the
[source repository](https://github.com/nvidia-isaac/isaac-os) at your tag or commit.
If your version is not in `llms.txt`, read `docs/` in the source repository.
