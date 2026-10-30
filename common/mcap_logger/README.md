# MCAP Logger

Records the outputs of a Holoscan graph to a single `.mcap` file. It is **not** a
`holoscan::Operator` — `HsdkMcapLogger` is a plain C++ object used in two phases around your own
`holoscan::Graph`, the same shape as `Ros2Bridge` (see `common/ros2_bridge/README.md`):

```text
                    compose()                    compile()              run_async()
                        |                             |                      |
  App Graph::compose()  |  attach<T>()                |                      |
       calls  --------->|  export_topic<T> + add_flow  |                      |
                        |  starter recorded, deferred  |                      |
                        v                             v                      v
                  [ Graph authoring ] ---------> [ ExecutionPlan ] -----> [ RunSession ]
                                                        |
                                                        | plan.endpoint_catalog()
                                                        v
                                            HsdkMcapLogger::start(catalog)
                                                        |
                                       owns: ChannelRuntime
                                             one poll thread per attached topic
                                             drop-oldest queue per topic (on its edge)
                                             McapSink (one shared .mcap file)
```

Every `attach<T>()` call mints its own exported HoloMQ topic and wires your operator's `Output<T>`
onto it — the logger never taps an existing topic or subscribes passively. Each attached topic
gets its own dedicated poll thread once `start()` runs, parked in a blocking wait that the
runtime's kernel event wake releases as soon as a sample arrives (no polling interval). The thread
writes each sample before receiving the next, so a topic's samples are written in the order they
were received. Each topic's queue is the edge into its exported output, bounded and drop-oldest, so
a slow or bursty topic can only drop its own samples: it cannot stall another topic's poll thread or
get another topic's samples dropped.

## Basic usage

```cpp
#include "common/mcap_logger/mcap_logger.h"

class MyGraph final : public holoscan::Graph {
 public:
  explicit MyGraph(isaac::os::mcap_logger::HsdkMcapLogger& logger)
      : Graph("my_graph"), logger_(logger) {}

 protected:
  void compose() override {
    auto my_op = op<MyOperator>("my_op");
    // Plain (flatbuffer) message type -- three-argument attach<T>().
    logger_.attach<isaac::JointCommandCpuT>(*this, "joint_command", my_op->joint_command_out);
  }

 private:
  isaac::os::mcap_logger::HsdkMcapLogger& logger_;
};

int main() {
  using isaac::os::mcap_logger::HsdkMcapLogger;
  using isaac::os::mcap_logger::HsdkMcapLoggerOptions;

  // output_path is a directory here, not a file: timestamp_output_path defaults to true, so
  // the logger names the actual file mcap_log_<local-timestamp>.mcap inside it. That default
  // exists so forgetting to change output_path between runs still can't silently overwrite a
  // previous recording. logger.output_path() returns the resolved path after construction.
  HsdkMcapLogger logger(HsdkMcapLoggerOptions{.output_path = "/data"});
  MyGraph graph(logger);

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  logger.start(plan.endpoint_catalog());
  auto run_session = holoscan::run_async(plan);

  // ... application runs ...

  run_session.request_stop();
  run_session.wait();
  // logger.shutdown() runs implicitly here via ~HsdkMcapLogger(), or call it explicitly first
  // to finalize the file (close it for reading) before the logger object itself goes away.
}
```

**Teardown ordering matters, the same way it does for `Ros2Bridge`:** stop the `RunSession`
(`request_stop()`/`wait()`) before, or concurrently with, shutting the logger down — never after.
The logger's poll threads keep calling into the graph's channels for as long as they're running,
so shutting the logger down first while the graph is still live tears those channels out from
under it.

`HsdkMcapLoggerOptions` also takes `enable_write_file` (defaults to `true`) and `enable_stream`
(defaults to `false`) — independent toggles, at least one of which must be `true`. With streaming
enabled, every attached topic also streams live to any Foxglove client connected to
`websocket_host`:`websocket_port` (defaults `"127.0.0.1"`:`8765`, matching the Foxglove SDK's own
default). `output_path()` returns `std::nullopt` when `enable_write_file` is `false` — there is
no file in that mode. If the running host's `127.0.0.1` isn't reachable from wherever the Foxglove
client runs (e.g. the app runs inside a container, or on a remote machine), either forward the port
deliberately (e.g. via SSH local port-forwarding: `ssh -L 8765:localhost:8765 <user>@<host>`, then
connecting Foxglove Studio to `ws://localhost:8765`) or override `websocket_host` to `"0.0.0.0"` or
a specific reachable interface — see the "Foxglove live streaming" section of the repository's
`SECURITY.md` before doing so.

`TopicLogOptions`, passed to `attach()` just before the optional `ConnectionOptions`, on both
overloads, sets what one topic asks for:

```cpp
struct TopicLogOptions {
  std::optional<VideoEncodeOptions> encode;  // record as video; tensor-schema image topics only
  std::uint32_t queue_depth = 16;            // samples held on the topic's queue
};

logger_.attach<isaac::JointCommandCpuT>(*this, "joint_command", source->command_out,
                                        TopicLogOptions{.queue_depth = 64});
```

`queue_depth` is how many of a topic's samples the transport holds while the topic's poll thread is
busy writing. When the poll thread falls behind and the queue is full, the oldest sample is dropped
to make room and the logger warns, for the first drop and then every tenth, naming the topic. The
producer is never refused or slowed: the logger only observes the graph. A topic that should not
lose samples gets a deeper queue, sized to how long its poll thread can stall, in samples. For a
tensor-schema topic each queued sample is a frame, so the depth is also how many frames of memory
the topic may hold.

The logger sets the edge into each exported topic itself: `attach()` overrides the `queue_policy` and
`queue_depth` of any `ConnectionOptions` you pass, and keeps its other fields. It does not set the
export's own subscriber bounds, which would silently override the edge's depth.

Each attached topic costs one thread, which receives, writes and releases one sample at a time.

**Set `timestamp_output_path = false`** to use `output_path` as the exact literal file to write
instead of a directory — e.g. a test or script that wants a fixed, predictable filename. In that
mode, `output_path` behaves exactly as it did before this option existed: the logger writes to
that path verbatim and does not create a timestamped name.

## Error handling

Failures that would otherwise let a run complete while silently recording nothing (or dropping a
topic) throw instead:

- **Constructing `HsdkMcapLogger`** throws `std::runtime_error` if neither `enable_write_file` nor
  `enable_stream` is `true`, if the output file can't be opened (bad directory, permissions, etc.),
  or if streaming is enabled but the websocket server can't bind (e.g. `websocket_port` is already
  in use) — a run that can't actually record or stream anything fails at construction rather than
  completing normally with every sample silently dropped.
- **`attach<T>()`** throws `std::invalid_argument` if `port_name` was already attached — without
  this, a duplicate `port_name` would silently overwrite the earlier attach's starter, and that
  earlier topic would simply never get logged, with no error anywhere. It also throws
  `std::invalid_argument` for a video-encoding request it could never satisfy: `encode` on any
  tensor-schema type other than `ImageT`, or a zero bitrate or frame rate — see "Video encoding"
  below for why those two throw while an absent hardware encoder does not.
  Finally, it throws `std::invalid_argument` if a topic's name collides with the
  `<topic>/metadata` channel of a video-encoded topic, in either order — attaching `camera` with
  encoding and `camera/metadata` as an ordinary topic would otherwise put two different schemas
  on one channel, and a channel keeps whichever schema its first write used, so the other
  topic's messages would be recorded under the wrong schema and read back as garbage with nothing
  logged. The `<topic>/metadata` name is claimed for the whole run by an encoded `attach()`,
  whether or not the host turns out to have a usable hardware encoder.
- **`start()`** throws `std::runtime_error` if called a second time (or after `shutdown()`), or if
  a subscriber can't be created for one of the topics recorded by `attach()`.

None of this is optional/configurable — a caller that wants softer failure handling wraps the
constructor/`attach()`/`start()` calls in its own `try`/`catch`.

## Compression

`HsdkMcapLoggerOptions::compression` sets how each chunk of the `.mcap` file is compressed. It takes
the Foxglove SDK's `foxglove::McapCompression` and applies to the whole file:

| Value | Use it for |
| --- | --- |
| `Lz4` (default) | A smaller file at little CPU cost. |
| `None` | The highest write throughput, at the cost of the largest file. Use it when the disk's speed and space allow. |
| `Zstd` | The smallest file, but it compresses large frames more slowly than a camera produces them, so a raw image topic can fall behind and drop samples. |

A raw 1080p stream at 30 fps is about 190 MB/s uncompressed. For camera images, prefer the logger's
hardware video encoding (see "Video encoding"), which writes a compact stream instead of raw frames.
A reader needs no setting for compression, because each chunk records its own.

## Live streaming

`enable_stream` fans every attached topic out to a live `foxglove::WebSocketServer` in addition to
(or instead of) the `.mcap` file — see "Basic usage" above for the options. Both sinks receive the
exact same call — `channel->log(payload, log_time_ns)`, or
`CompressedVideoChannel::log(message, log_time_ns)` for an encoded topic — so a live client and
the file always see identical timestamps for every sample; there is no separate "receive time"
path for the live sink.

**Before overriding `websocket_host` to bind beyond loopback, read the "Foxglove live streaming"
section of the repository's `SECURITY.md`.** The server authenticates no client and encrypts
nothing, so anyone who can reach the port can subscribe to every topic the application exports. The
default bind, `127.0.0.1`, limits that to clients on the same host (or reaching it via a port
forward); binding `0.0.0.0` or another externally reachable interface exposes the port to anyone
who can route to the machine. The exposure is read-only either way (clients cannot publish, set
parameters, or call services), but the telemetry itself is fully readable.

**Known limitation: the last ~few seconds of a run may not reach a connected live client, even
though the file is always complete.** `McapSink`'s destructor calls `writer_->close()` (synchronous,
local, always finishes before returning) and then `server_->stop()` immediately after, with nothing
in between that waits for outstanding network sends to already-connected clients to flush. Verified
against the vendored Foxglove SDK (0.14.3): its public API (`foxglove/channel.hpp`,
`foxglove/server.hpp`, and the raw C header) has no `flush()`/`drain()`/backlog-wait primitive
anywhere, and `stop()`'s own documentation ("gracefully shut down") makes no drain guarantee.
Observed empirically on this branch: a 90s run's `.mcap` file completed to ~89.9s while a connected
Foxglove Studio client visibly stopped updating a few seconds earlier (~86s) in the same run. This
is a limitation of the current Foxglove SDK shutdown API.

**Practical implication:** treat the live view as best-effort near-real-time monitoring, not a
substitute for the recording. If you need to be sure you've seen every sample a run produced
(especially near the very end of a run), read it back from the `.mcap` file, not from what a live
client happened to render before the server stopped.

A separate, less-investigated observation: a live Foxglove Studio client's plot rendering looked
slightly choppier than the same data played back from the finished file, despite both receiving
identical timestamps on the wire. This is more likely a Foxglove Studio client-side live-rendering
behavior than anything on the logging side, but it hasn't been root-caused.

## Tensor-schema messages (e.g. `holoscan::schema::ImageT`)

A type whose tensor data is transported out-of-band — `ImageT` is the built-in example — needs a
`DescriptorTensorTopicContract<T>` describing that tensor field's bounds, and uses a different
`attach<T>()` overload, which takes that contract as its fourth argument:

```cpp
#include "common/image_schema/image_schema.h"

void compose() override {
  auto camera = op<MyCameraOp>("camera");
  logger_.attach<holoscan::schema::ImageT>(
      *this, "camera", camera->image_out,
      isaac::os::image_schema::ImageTensorFieldContract(
          1920, 1080, holoscan::schema::ImageEncoding_RGB8));
}
```

Passing a tensor-schema type to the plain three-argument `attach<T>()` instead is a compile error
(a `static_assert` names the correct overload), not a runtime surprise. Conversely, the SDK's own
`export_topic<T>` constraints make the reverse mistake — a plain type with a `contract` argument —
a hard compile error too, so the two overloads can't be crossed either way.

`isaac::os::image_schema::ImageTensorFieldContract()` (`common/image_schema/`, shared
with `common/ros2_bridge/tensor_converter.h`'s own `ImageT` contract) is a ready-made helper
for `ImageT` specifically — it computes the tensor's byte bound as `width * height *
bytes_per_pixel(encoding)` and throws `std::invalid_argument` for an encoding it doesn't recognize
or a dimension/encoding combination that overflows a 64-bit byte count. A caller that wants an
explicit byte bound instead calls `ImageTensorFieldContract(max_tensor_bytes, memory_kind)`
directly. A custom tensor-schema type with no equivalent helper builds its own
`DescriptorTensorTopicContract<T>` directly.

## Video encoding

An `ImageT` topic can record as hardware-encoded (NVENC) video instead of raw tensor bytes. Set
`encode` in the `TopicLogOptions` passed to the tensor-schema `attach<T>()`:

```cpp
#include "common/mcap_logger/video_encode_options.h"

logger_.attach<holoscan::schema::ImageT>(
    *this, "camera", camera->image_out,
    isaac::os::image_schema::ImageTensorFieldContract(
        1920, 1080, holoscan::schema::ImageEncoding_RGB8),
    isaac::os::mcap_logger::TopicLogOptions{
        .encode = isaac::os::mcap_logger::VideoEncodeOptions{}});
```

```cpp
enum class VideoCodec : std::uint8_t { kH264, kH265 };  // H.265 and HEVC are the same codec.

struct VideoEncodeOptions {
  VideoCodec codec = VideoCodec::kH264;
  std::uint32_t bitrate = 10'000'000;
  std::uint32_t frame_rate = 30;
  std::uint32_t cuda_device = 0;
};
```

The parameter defaults to `std::nullopt`, which is the raw `TensorFrame` path described above,
byte for byte unchanged. There is no width/height in the options: they are read from each frame's
own `holoscan.schema.Image` metadata, so they cannot drift from the
`DescriptorTensorTopicContract` that already bounds the tensor field.

The motivation is write throughput — fewer bytes per frame through one serializing MCAP writer —
and native Foxglove rendering: a `foxglove.CompressedVideo` channel renders in an Image panel with
no converter extension installed, where a raw `TensorFrame` needs one. It is not archival
compression; see "Every frame is a keyframe" below for why the ratio is modest.

`codec` is an enum rather than a string because H.264 and HEVC are the only codecs NVENC's
`CODEC_GUIDS` and NVDEC's `CODEC_IDS` both carry: asking for anything else is a compile error, so
there is no runtime rejection to handle. Two total mappings turn it into the two spellings its
consumers need — `FoxgloveCodecFormat()` for the lowercase `foxglove.CompressedVideo` `format`
field, `NvencCodecName()` for the uppercase keys of NVENC's codec table.

`attach<T>()` throws `std::invalid_argument` for `encode` passed with any tensor-schema type
other than `ImageT` (no other payload carries the width/height/encoding an encoder needs), and
for a zero `bitrate` or `frame_rate`. These throw rather than degrade because they are caller
errors — wrong on every machine, on every run. An absent hardware encoder is a different thing —
a property of the host, not of the call — but it also fails loudly, from `start()`, instead of
recording raw tensors in place of the video that was asked for. See the next section.

### The raw-vs-encoded decision is permanent per topic

`start()` builds each encoded topic's encoder with `FrameEncoder`'s constructor before any channel
exists. The constructor probes the `cuda_device` named in the options it was handed — the same
device the encoder would encode on, so the probe and the encoding target cannot name different
GPUs — and throws `std::runtime_error` when that GPU has no usable hardware encoder. An A100 or
H100 ships a video decoder but no encoder, and a `cuda_device` this host does not have counts the
same way. The message names the topic and the device and says what to change.

It throws rather than recording raw `TensorFrame`s in place of the video: a run that silently
substitutes one for the other looks healthy, and a single warning line is easy to miss. A caller
that wants the fallback asks `FrameEncoder::hardware_available(cuda_device, codec)` first and
attaches the topic without `encode` — the reference app does exactly that, and warns. A zero bitrate
or frame rate throws `std::invalid_argument` from the same constructor, before the probe, keeping a
caller error distinct from a host that cannot encode. The probe opens and closes a real encoding
session, about 160 ms on the build host, and asks it which codecs it supports, so its answer is
cached for the life of the process, per device and codec: a graph with several encoded topics pays
for it once per device and codec, however many loggers or topics ask. A probe that fails stays
failed until the process restarts.

That decision can never be revisited, because an MCAP channel carries exactly one schema for its
whole life. Once a topic's channel exists as `foxglove.CompressedVideo`, a later failure — an
encode error, a mid-run resolution change, or a source encoding other than RGB8/RGBA8 — drops that
frame with a rate-limited warning. It cannot write a `TensorFrame` into a `CompressedVideo`
channel, so there is no per-frame fallback.

One failure is louder than the rest: a device that passes the probe can still fail to give this
topic a session of its own — NVENC caps how many sessions a GPU will hold open at once, for
instance — and that is only discovered when the first frame arrives, too late to choose the raw
path. That failure is permanent: the topic records nothing at all, and its channel is never even
created. It is therefore reported once at error level, in full, outside the rate-limited
dropped-frame budget, and the session is never reopened for later frames.

### An encoded topic writes two channels

| Channel | `message_encoding` | `schema.name` |
| --- | --- | --- |
| `<topic>` | `protobuf` | `foxglove.CompressedVideo` |
| `<topic>/metadata` | `flatbuffer` | `holoscan.schema.Image` |

Both are written by the same poll thread at the identical `log_time`, so a reader joins them
frame for frame on `log_time` alone. `attach()` claims the `<topic>/metadata` name, so no other
topic can be attached under it — see "Error handling" above.

The sidecar exists because `foxglove.CompressedVideo` carries only `timestamp`, `frame_id`,
`data` and `format`, while the raw `TensorFrame` path preserves every field of
`holoscan.schema.Image`. Writing only `CompressedVideo` would silently discard `encoding`,
`color_space`, `significant_bits`, `roi_offset_x`/`roi_offset_y`, `gain`, `exposure_time_ns` and
`header.capture_timestamp_ns`. Two of those are load-bearing rather than bookkeeping:
`capture_timestamp_ns` is normatively the start of sensor integration and `exposure_time_ns` is
its companion, so a consumer fusing against an IMU reconstructs the true frame instant as
`capture_timestamp_ns + exposure_time_ns / 2`. `CompressedVideo::timestamp` cannot stand in for
that — it is `publish_timestamp_ns`, the graph-emit time, a different fact under a similar name.
`color_space` matters because the image schema states its images are not self-describing without
it.

The sidecar writes the `Image` table as its channel's own schema rather than as opaque nested
bytes, so generic MCAP tooling decodes it with no unwrapping step.

### Every frame is a keyframe

The encoder forces intra-only IDR frames with in-band SPS/PPS on every single frame,
unconditionally. That is deliberate and deliberately not configurable. Every frame is
independently decodable, which is what makes random-access seeking, a client joining mid-stream,
and out-of-order tolerance free rather than requiring keyframe tracking and seek-to-keyframe logic
on the decode side. Foxglove also rejects B-frames outright, so the compression the relaxation
would buy is compression Foxglove cannot use.

The honest consequence: this is all-intra video, not inter-predicted video. Expect a modest ratio
against the raw tensor bytes — roughly what a stream of individually compressed stills gives you,
not what a long-GOP archival encode gives you. If your goal is the smallest possible file rather
than write throughput and native rendering, this is not that feature.

### Pixels are the one genuinely lossy part

With the sidecar, metadata survives encoding intact. Pixels do not. NVENC consumes ARGB and NVDEC
returns NV12, so 4:2:0 chroma subsampling plus bitrate quantization make the round trip lossy and
never bit-exact. That is inherent to lossy video and is the point of the feature — but it is a real
loss, and the raw `TensorFrame` path remains bit-exact if you need it. An encoded topic replays
through `common/mcap_replayer` (see its README) at the layout its metadata names (RGB8 or RGBA8),
with alpha always 255. Because only RGB8 and RGBA8 can be staged into the
encoder's input pixels, a frame arriving under any other source encoding is checked and dropped
rather than mangled. The loss is at least visible rather than silent: `<topic>/metadata` still
reports the original `encoding`.

An encoded recording is viewable in Foxglove Studio, and `common/mcap_replayer` replays it back
into a graph as host images (see its README). `tests/test_mcap_logger_encoded_video_integration.cc`
decodes a written file back through NVDEC to prove the bytes are readable.

## Reading a recording

**A reader must use `mcap::ReadMessageOptions::ReadOrder::LogTimeOrder`.** Messages are not
written to the file in `log_time` order: `log_time` is stamped on a topic's poll thread from
`SampleMetadata::publish_timestamp_ns` *before* the sample is written. A
topic's own samples are written in the order they were received, but every topic has its own poll
thread, so records of different topics land in the file in whatever order the threads finish.
Reading in file order therefore returns messages of different topics out of time order.

This is a property of the logger, not of video encoding — it applies identically to every plain
flatbuffer and raw tensor channel. MCAP's `IndexedMessageReader` falls back to a scan when a file
has no chunk index, so `LogTimeOrder` works on every file this logger produces.

## Design

| Component | File | Responsibility |
| --- | --- | --- |
| `HsdkMcapLogger` | `mcap_logger.h` | Facade: `attach<T>()`, `start()`, `shutdown()`. Owns the `ChannelRuntime`, the pollers, and the sink. |
| `PollerBase` | `topic_poller.h` | What every poller shares: the topic's name, reporting dropped samples from sequence gaps, and keeping one failed write from ending the topic. |
| `TopicPoller<T>` | `topic_poller.h` | One dedicated poll thread per plain-flatbuffer topic; serializes via `T`'s `SchemaPackageDescriptor` and writes each sample before receiving the next. |
| `TensorTopicPoller<Payload>` | `tensor_topic_poller.h` | Tensor-schema counterpart to `TopicPoller<T>`; materializes each received tensor (CUDA or host) directly into a `TensorFrame` FlatBuffers message (one copy per tensor, no staging buffer) and writes it before receiving the next. A topic holding a `FrameEncoder` takes the other branch instead: the tensor goes to the encoder, and the poller writes the video frame plus its metadata sidecar rather than a `TensorFrame`. |
| `TopicLogOptions` | `topic_log_options.h` | What one attached topic asks for, passed to `attach()`: `encode` and `queue_depth`, plus `EdgeOptionsFor()`, which sets the topic's edge to drop-oldest at that depth. |
| `FrameEncoder` | `frame_encoder.h`/`.cc` | One topic's NVENC session, created only for a topic that opted into video encoding, and whose constructor probes the options' own `cuda_device` and throws when that GPU cannot encode. Every encoder that exists has therefore passed its probe, which is still not a promise that encoding will succeed: the per-frame failure latch stays for what a probe cannot foresee, such as a frame size the encoder rejects or a device that stops granting sessions. Sized lazily from the first frame, which is also when its device staging buffers are allocated: they are kept for the session and reused by every later frame rather than allocated and freed per frame. Every call serialized on one mutex, because an NVENC session mutates a shared input-buffer ring and is not thread-safe. |
| `McapSink` | `mcap_sink.h`/`.cc` | Wraps the Foxglove MCAP writer. One channel per attached topic — two for an encoded topic, the video channel plus its `<topic>/metadata` sidecar — each created lazily and reused thereafter, mutex-guarded only around channel lookup/append. |
| `isaac::os::image_schema` | `common/image_schema/` | `ImageT`-specific tensor-contract helpers, shared with `ros2_bridge` — not part of this component, but the one every `ImageT` caller here uses. |

Each topic's queue is the edge into its exported output, owned by the transport, so the bound on how
much one topic may occupy cannot be spent by another. The poll thread writes one sample at a time;
the queue drops the oldest when it is full.

**MCAP `log_time`** always comes from `SampleMetadata::publish_timestamp_ns` — the runtime's own
graph-emit timestamp, read once per publication and stable across every subscriber — which is what
drives replay pacing. Nothing else (capture/acquisition time, wall-clock time at write) is used.
If a message needs its own capture time, that belongs in the message's own schema/payload, not
carried by this logger out of band.

**On-disk framing:** every channel this logger writes carries one of exactly two well-known
`message_encoding`/`schema.name` pairs:

| Channel kind | `message_encoding` | `schema.name` |
| --- | --- | --- |
| Plain flatbuffer, raw tensor, and encoded-video metadata sidecar | `flatbuffer` | the attached type's own schema, or `isaac.mcap_logger.TensorFrame` for raw tensors |
| Encoded video (opt-in, see "Video encoding") | `protobuf` | `foxglove.CompressedVideo` |

Both fields are MCAP channel-level metadata, so a reader knows which kind a channel is at
file-open time, before it parses a single message — there is no per-message sniffing and no
ambiguity about how to decode a channel. Any generic MCAP/Foxglove tool decodes either directly,
with no special-casing. Unless a topic opts into video encoding, every channel is a FlatBuffers
message, which is what every topic did before that option existed. Tensor-schema topics all share
one root schema, `isaac.mcap_logger.TensorFrame`
(`common/mcap_logger/tensor_frame.fbs`) — a `tensors: [TensorEntry]` vector (each entry:
`shape`, `dtype`, `data`) plus an opaque `metadata`/`metadata_schema_name` pair carrying the
attached type's own metadata table (e.g. `holoscan.schema.Image`'s `width`/`height`/`encoding`).
One shared schema means one Foxglove converter can cover every tensor-schema type this logger ever
writes, not just `ImageT`.

**Concurrency/type-safety discipline:** everything is resolved at compile time on the attached type
`T` (or tensor `Payload`) — there is no `std::type_index` or runtime type registry anywhere in this
component. A poller's `stop()` joins its poll thread, which first writes what its queue still
holds, so every sample published before the stop has been written (unless it was dropped), and
every read claim released, before the poller's subscriber is destroyed.
`HsdkMcapLogger::shutdown()` destroys every poller before it resets the `ChannelRuntime` and drops
the sink, so nothing relies on a claim outliving its subscriber or the runtime.

## Current capabilities

- Attach any plain flatbuffer-schema message type (`attach<T>()`) or any tensor-schema type with a
  `DescriptorTensorTopicContract<T>` (`attach<T>(..., contract)`) from a graph's `compose()`.
- One poll thread per attached topic, so every topic's samples are written in the order they were
  received and topics write in parallel, each behind its own drop-oldest queue.
- One `.mcap` output file per logger, with per-topic channel reuse.
- Deterministic, explicit `shutdown()` (also runs implicitly via the destructor) that finalizes the
  file even if the owning graph is torn down before it finishes a full run.
- Hard failures (bad output path, duplicate `attach()` under the same `port_name`, calling
  `start()` twice) throw rather than silently dropping data — see "Error handling" above.
- Timestamped output filenames by default (`mcap_log_<timestamp>.mcap` inside a given directory),
  so forgetting to change `output_path` between runs can't silently overwrite a previous
  recording; opt out with `timestamp_output_path = false` for a fixed, predictable path.
- `ImageT`-specific contract helper (`image_schema::ImageTensorFieldContract`) shared with the ROS2
  bridge, so both components size the same tensor field the same way.
- Configurable chunk compression for the recording (`HsdkMcapLoggerOptions::compression`), LZ4 by
  default — see "Compression" above.
- Opt-in NVENC video encoding for an `ImageT` topic (`attach<T>(..., contract,
  TopicLogOptions{.encode = VideoEncodeOptions{}})`), writing a `foxglove.CompressedVideo` channel
  plus a `<topic>/metadata` sidecar — see "Video encoding" above, including what it costs.

## Tests

`tests/` covers each piece independently: `test_poller_base.cc` (drop counting from sequence gaps
and the warning's schedule), `test_mcap_sink.cc` (channel reuse, exact byte-for-byte framing for
both message shapes), `test_mcap_logger_wiring.cc` (compose-time `attach<T>()` wiring for both
overloads, `start`/`shutdown` lifecycle, including that `shutdown()` is terminal), and
`test_mcap_logger_integration.cc`/`test_mcap_logger_tensor_schema_integration.cc` (full
attach-compile-run-shutdown integration tests against a real running graph, driven by
`NotificationSource`/`NotificationSender` for an exact, deterministic sample count rather than a
sleep-and-hope wall-clock guess, with exact byte-for-byte verification of the resulting `.mcap`
file's content on read-back).

Video encoding adds `test_video_encode_options.cc` (both codec spellings and the rejected
configurations), `test_frame_encoder.cc` (per-topic encoder behavior, including reading a
device-resident frame where it lies and staging a host-resident one), and
`test_mcap_logger_encoded_video_integration.cc` (a full run written to an
`.mcap`, read back, and decoded through NVDEC — including a check that red and blue come home in
the order they left, which is the one defect a perfectly valid, perfectly decodable bitstream can
still hide).

`test_mcap_logger_tensor_schema_integration.cc` also has one CUDA-device-only test
(`LogsImageFramesFromACudaDeviceTensor`) that emits a real device-resident Tensor to exercise
`TensorTopicPoller`'s `cudaMemcpy` (`kDLCUDA`) materialization branch specifically, rather than
the host-memory branch every other test exercises. It self-skips (`GTEST_SKIP()`, reported as
SKIP rather than FAIL) on a node with no usable CUDA device.
