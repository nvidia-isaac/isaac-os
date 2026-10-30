# mcap_replayer

Replays a recorded `.mcap` file into a live Holoscan graph, reproducing the wall-clock pacing the
recording was made with. It is **not** a `holoscan::Operator`: `McapReplayer` is a plain C++ object
used around your own `holoscan::Graph`, in two phases:

```text
  McapReplayer(path, Topic<T>{...}...)
        |
  Graph::compose()                    holoscan::compile(graph)       holoscan::run_async(plan)
        |                                      |                              |
  attach<T>() per topic                        |                              |
  (import_topic + add_flow)                    |                              |
        |                                      v                              v
        +--------------------------->  ExecutionPlan  ------------->  RunSession
                                               |                              |
                                 plan.endpoint_catalog()                      |
                                               +--> McapReplayer::start(catalog)  (after run_async)
                                                      owns, per topic, a publisher, plus a
                                                      prefetch thread and a pacing thread
```

Every topic is wired to its consumer with `attach<T>()`, which imports the topic into your graph and
calls `add_flow()` on your behalf, so a consumer cannot tell replayed data from live data by how it
is wired. A topic you name but never `attach<T>()` is left out of the run: it is not imported, read
or counted.

## Basic usage

1. Construct the replayer, naming every topic to replay with the type its recorded messages decode
   to. It needs no graph, so build it first and hand it to the graph that wires it:

   ```cpp
   isaac::os::mcap_replayer::McapReplayer replayer(
       path, isaac::os::mcap_replayer::Topic<isaac::PoseCpuT>{"pose"});

   class MyGraph final : public holoscan::Graph {
    public:
     explicit MyGraph(isaac::os::mcap_replayer::McapReplayer& replayer)
         : Graph("my_graph"), replayer_(replayer) {}

    protected:
     void compose() override {
       set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
       auto sink = op<MySinkOp>("sink");
       replayer_.attach<isaac::PoseCpuT>(*this, "pose", sink->in_pose);
     }

    private:
     isaac::os::mcap_replayer::McapReplayer& replayer_;
   };
   ```

   The constructor also takes an optional `ReplayOptions` before the topics (see below). It throws
   `std::runtime_error` if the recording cannot be opened, and `std::invalid_argument` if the
   options are invalid, a topic is named twice, a named topic is absent from the recording, or its
   recorded schema does not match the type it was named with. All of this is checked before a
   message is read, so a wrong topic name or type fails before the graph is built.
   A topic that only some recordings hold can be checked first with
   `McapReplayer::recording_has_topic(path, topic)`, which opens the file and reports whether the
   topic is in it.

   `attach<T>()` calls `add_flow()` itself with a `kDropOldest` queue policy, and takes no options
   to change it. Replay keeps pace with the recording rather than waiting on a slow consumer: a
   publish that finds the consumer's queue full overwrites the oldest unread sample instead of
   blocking, so the pacing thread never stalls and the recording's timing is kept. A consumer that
   cannot keep up loses the oldest unread samples, and has to be made faster or the playback speed
   lowered.

2. Compile, run, and start the replayer, calling `start()` **after** `run_async()`:

   ```cpp
   MyGraph graph(replayer);
   const holoscan::ExecutionPlan plan = holoscan::compile(graph);
   auto run_session = holoscan::run_async(plan);
   replayer.start(plan.endpoint_catalog());

   // ... application runs; replayed messages reach sink as they become due ...

   replayer.stop();  // or let ~McapReplayer() do it; either logs the run's summary once
   run_session.request_stop();
   run_session.wait();
   ```

   `start()` attaches the publisher of every `attach<T>()`ed topic against the catalog
   synchronously, so the run is ready to publish when it returns. It throws `std::runtime_error`
   if an attachment fails and `std::logic_error` if called twice; replay a recording again with a
   new `McapReplayer`.
   `start()` and `stop()` belong to the thread that constructs and destroys the replayer.

## Options

`ReplayOptions`, passed to the constructor:

- `playback_speed` (default `1.0`, must be greater than zero): recorded seconds per wall-clock
  second. `2.0` replays twice as fast.
- `loop` (default `false`): start the recording over at its end. Without it the recording plays
  once and the replayer stops producing.
- `buffer_messages` (default 256, must be at least one): how many decoded messages may wait ahead
  of the playback cursor.
- `start_stall_timeout` (default 5 s, must be greater than zero): how long reading the recording
  may go without reading a new message while the run is starting before the run gives up.
- `operator_name` (default `"mcap_replayer"`): prefix of each topic's external graph endpoint,
  `"<operator_name>/<topic>"`. A graph has one flat endpoint namespace for imports and exports, so
  set this when the same graph also records the topic (for example with `mcap_logger`) or runs
  more than one replay.

## How it replays

Two threads, because reading and decoding may run ahead of the recording's timeline but publishing
must not:

- **The prefetch thread** reads the recording in log-time order, builds a message from each one on
  an `attach<T>()`ed topic and pushes it into a bounded `ReplayBuffer` (`buffer_messages`). It
  blocks while the buffer is full, which is the only backpressure it gets. Messages on any other
  topic are skipped.
- **The pacing thread** asks the buffer for the front message with `ReplayBuffer::release_if_due()`,
  which inspects the front entry, decides whether it is due and pops it under one lock. A due
  message is published at once, on this thread: its topic is looked up by the message's channel id,
  and the topic's port sends it on a `ChannelPublisher<T>` (or `ChannelTensorSchemaPublisher` for a
  tensor-schema topic). Otherwise the thread sleeps until the message's due time or a stop request.

Due times come from `PacingClock`: each is an absolute offset from the pass's origin (the first
message's recorded time mapped to a wall-clock instant), divided by `playback_speed`, never an
increment on the previous due time, so rounding error does not accumulate. With `loop`, when the
reader wraps to the start of the recording, `release_if_due()` rebases the clock onto the new pass.

Before it sends anything, the pacing thread passes a start-up gate once per run (not again on loop
wrap). It waits for a little to be decoded ahead and for every `attach()`ed topic's consumer to
subscribe: the imported topic's edge takes wall-clock time to come up after `run_async()`, and
publishing earlier would send the opening messages to nobody.

- **Buffering** is done once up to 32 messages (or `buffer_messages`, if fewer) are ready. In a
  recording played once, if the `attach()`ed topics hold fewer messages than that, having all of
  them counts, and so does the reader finishing. A reader that reads no new message for
  `start_stall_timeout` (5 s by default) is fatal: the run logs an error, sends nothing and stops
  its own threads, and `stats().start_stalled()` is true. Slow reading that keeps making progress,
  including past messages on topics nobody attached, is never a stall.
- **Subscribing:** a consumer that has not subscribed within `kSubscriberWaitDeadline` (5 s, shared
  by all topics) is also fatal: the run logs an error naming each such topic, sends nothing and
  stops its own threads. `stats().never_subscribed(topic)` is then true for those topics.

Either way the summary says the run never began. Still call `stop()` afterwards. A topic that was
never `attach()`ed is neither waited on nor read.

## Statistics

`stats()` returns a `ReplayStats`; `stop()` logs its summary once. Under the `kDropOldest` policy
`attach<T>()` always uses, a slow consumer's queue silently drops its own oldest unread sample, so
`stats()` counts what was published, not what was consumed. Per topic it reports:

- `emitted`, `refused`, `decode_failures`: messages published; messages whose publish failed (no
  subscriber is connected, for example because the topic was never wired or its consumer's graph
  stopped; not "the consumer was too slow"); and recorded messages that could not be decoded as
  the topic's type.
- `worst_lag` and `mean_lag`: how late each message went out relative to its due time (never
  negative). `late_arrivals` counts messages that went out more than `ReplayStats::kLateThreshold`
  (10 ms) after they were due, whatever the topic's rate or the playback speed: it points at the
  machine delaying the replay between waking up and publishing. Their numbers among the messages
  delivered on the topic in the pass (1 is the first message delivered) are kept, as the first ten
  runs of consecutive messages, so a few late messages at the start of a run can be told from
  lateness throughout it. A message skipped as unreadable or refused takes no number, so the
  numbers match the recording's own order only while none were skipped; the topic's unreadable and
  refused counts show when some were. A pass ends where the topic's recorded time steps backwards,
  so a topic with one message per pass, or whose messages all share one recorded time, has no pass
  boundary to report at: its numbers run through the whole run. The summary lists them for the
  final pass, which is the whole run unless it looped. When a looping
  recording starts over, the late messages of the pass that just ended are logged, at most once per
  `ReplayStats::kWarnInterval` (1 s) per topic, noting how many further late messages the quiet
  stretch held back. Nothing is logged as each late message goes out, so a run that falls behind is
  not slowed further by its own logging.
- `start_stalled()` and `read_failed()`: run-wide flags. The first is set when the start-up wait
  gave up on a reader making no progress. The second is set when reading failed partway through
  the recording: the run ends there with an error and does not wrap when looping, and the summary
  says only the part read was replayed. `message_count(topic)` on the replayer gives the number of
  messages the recording holds on a topic's replayed channel, for waiting on completion.
  `finished()` on the replayer says whether the run has ended: everything the reader read has gone
  out, or the run gave up, failed to read or was stopped (the flags above say which). It stays false
  while a looping replay keeps going, so it replaces counting messages against `message_count()`
  for a recording played once.
- `recorded_hz` and `replayed_hz`, and `typical_schedule_error` and `worst_schedule_error`:
  readings taken against the recording's own timestamps that never consult a due time, so they
  can notice a due time that was itself wrong. `replayed_hz` is in recorded time, so a faithful run
  matches `recorded_hz` at any speed.
  The schedule error is a signed series within each pass over the recording: each message's
  delivered time since the pass's first message, less its recorded time since that message divided
  by `playback_speed`. Only the spread within a pass counts, so a delay common to a whole pass
  (even on its first message alone) is excluded. `typical_schedule_error` is the standard
  deviation about each pass's own mean, pooled over all passes. `worst_schedule_error` is the
  largest peak-to-peak spread of one pass (earliest to latest landing relative to the recording).
  The rates are in recorded time while the errors are in delivered (wall-clock) time. All four are
  `std::optional`, empty until a pass has delivered two messages and for a recording with no
  cadence (every message at one recorded time, or one message per pass of a loop).

The summary prints, per topic, a delivered count, then lines for refused, unreadable and
never-subscribed topics where applicable, then:

```text
  Messages arrived 0.1 ms late on average and 1.2 ms late at worst.
  Replayed at 100.00 Hz, recorded at 100.00 Hz; messages kept the recording's own spacing to within 0.2 ms typically and 1.4 ms at worst.
```

The second line is omitted where the topic has no cadence, and at a playback speed other than 1 it
instead reads `Replayed at <r> Hz in recorded time (<d> Hz as delivered at <s>x speed), recorded at
<r> Hz; as delivered, messages kept ...`. Closing lines explain refusals, lateness, decode
failures or a run that never began, or report that every message was delivered on schedule.

## Tensor-schema topics

A topic recorded in `mcap_logger`'s tensor envelope (wire schema `isaac.mcap_logger.TensorFrame`,
see `common/mcap_logger`), for example `holoscan::schema::ImageT`, is published differently from a
plain topic. The recorded metadata bytes are passed unchanged to
`ChannelTensorSchemaPublisher<Payload>::emit()` together with tensors rebuilt in host memory; no
`T` value is ever materialized. This works because `mcap_logger` writes those bytes verbatim in
the wire format the publisher expects. Every tensor is rebuilt in host memory because a recording
does not carry the memory kind or device it came from. A damaged message, or one whose tensor
count, shape or size disagrees with what it declares, is counted as a decode failure rather than
published.

A tensor-schema topic is named with the contract its imported edge is built from, as for
`mcap_logger`'s `attach()`: `Topic<T>{name, contract}`, where `contract` is a
`DescriptorTensorTopicContract<T>` giving the bounds, memory kind and transfer policy of each tensor
field. The caller supplies it because what is right depends on the frames the topic carries, and
there is no `Topic<T>` constructor without one, so leaving it out does not compile. For
`holoscan::schema::ImageT`, `isaac::os::image_schema::ImageTensorFieldContract()` builds it, for
example `Topic<holoscan::schema::ImageT>{"image", ImageTensorFieldContract(8 * 1024 * 1024)}`; a
frame larger than the bound is refused when it is published. A plain type is named with just
`Topic<T>{name}`. Such a topic must be recorded in the tensor envelope; one recorded flat is
rejected at construction.

## Encoded video

A topic `mcap_logger` recorded as hardware-encoded video (`foxglove.CompressedVideo` plus its
`<topic>/metadata` sidecar) is declared exactly like a raw image topic,
`Topic<holoscan::schema::ImageT>{name, contract}`; the recording decides how it is read. Each
frame is decoded on NVDEC, converted on the GPU, copied to host memory and published as the same
rank-1 `uint8` tensor under the sidecar's metadata bytes unchanged.

- The pixel layout follows the sidecar's `encoding`: `RGB8` replays as three channels and `RGBA8` as
  four. H.264 and H.265 carry no alpha, so replayed alpha is always 255 whatever was recorded.
- Video is lossy: replayed colors match the recorded ones to within compression loss (a few
  levels on flat color), not bit for bit as a raw recording does. The stream is treated as BT.601
  limited range, the way the logger's encoder writes it.
- Only even frame sizes replay: video is 4:2:0, so a frame with an odd width or height is dropped
  and counted, with the size in the reason.
- `ReplayOptions::video_decoder_cuda_device` (default 0) picks the GPU that decodes. Construction
  throws `std::invalid_argument` if that GPU cannot decode the topic's codec (H.264 or H.265; no
  hardware decoder for it, or a build without the NVIDIA Video Codec SDK), if the topic's first
  video message is missing, damaged or names no codec (the reason is in a warning), or if the
  `<topic>/metadata` channel is missing or is not image metadata. Name the video topic, not its
  `<topic>/metadata` channel, which is rejected.
- The video and metadata messages are joined on their shared log time. A message whose partner
  never arrives at that log time is left out of the replay, so the replay can deliver fewer frames
  than the recording's `message_count`. Each such message is counted per topic, and one warning
  names the topic and how many were left out when a pass of the recording ends; a looping replay
  warns once per topic, not on every pass. These are not part of the run's unreadable-message
  count. A frame that fails to decode, or decodes to a size other than the metadata's, is dropped
  and counted like any unreadable message.
- Decoded frames are held in host memory in the replay buffer (`buffer_messages` deep), so a
  high-resolution topic needs `buffer_messages` frames of RAM.

## Tests

`tests/` covers each piece on its own, then the whole replayer against a live graph:

- `test_mcap_file.cc` -- opening and reading a recording: topic lookup, log-time order, looping
  without the caller seeing the wrap, per-channel counts, damaged or unindexed recordings reported
  with their reason rather than looking empty, tensor-envelope type checks, and the lowest-channel
  rule when one topic is on several channels.
- `test_decoder.cc` -- building the published message from a flat payload, a bad payload
  rejected with a reason, and emitting before attach reported rather than crashing.
- `test_tensor_decoder.cc` -- rebuilding tensors from the envelope, rejecting malformed ones
  (wrong marker, missing tensor, shape not matching its data, shape overflow), and replaying
  pixels intact through a real graph alongside a flat topic.
- `test_pacing.cc` -- `PacingClock`: absolute targets that cannot drift, speed scaling, rebase
  on wrap, and the sleep cutoff.
- `test_replay_buffer.cc` -- `ReplayBuffer`: the reader is reported finished only after every
  pushed entry has been released.
- `test_replay_stats.cc` -- `ReplayStats`: per-topic counts, lateness, rate and schedule-error
  measures, rate-limited warnings, and the end-of-run summary, including that a run which never
  began, failed or fell behind is not called healthy.
- `test_mcap_replayer.cc` -- construction checks, wiring, the start-up and stop lifecycle, and
  live runs held to the recorded schedule: uneven cadence, each new pass, a slow consumer, a
  recording many times the buffer size, and a consumer that never subscribes.
- `test_compressed_video_builder.cc` -- the encoded-video builder refusing, with a reason, image
  metadata that is not an image and an encoding that video is never recorded from.
- `test_compressed_video_replay.cc` -- on a GPU with NVENC and NVDEC, a recording written by
  `mcap_logger` as encoded video replayed through a live graph as RGB, RGBA and H.265, with the
  image metadata passed through and the colors checked against what was recorded.

The helpers `mcap_test_support.h`, `replay_test_harness.h` and `scoped_log_level.h` are shared by
several files.
