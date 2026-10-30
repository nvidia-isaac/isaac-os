# mcap_replay_reference

Minimal reference application for `common/mcap_replayer`. It opens an `.mcap` file written by
`mcap_logger_reference` and replays seven of its topics, and an eighth when the recording holds
it and this host can decode it, back into a live Holoscan graph at the timing they were recorded
with, then prints a per-topic summary of the run.

- `sine_joint_state`, `cosine_joint_state` -- `isaac.JointStateCpu`
- `circular_pose` -- `isaac.PoseCpu`
- `circular_twist` -- `isaac.TwistCpu`
- `circular_transform` -- `isaac.TransformCpu`
- `camera_info` -- `isaac.CameraInfoCpu`
- `counter_image` -- a tensor-schema `holoscan.schema.Image` stream
- `counter_image_encoded` -- the same kind of stream at 1920x1080, which the logger records as
  H.264 video (plus a `counter_image_encoded/metadata` sidecar) on a host with a hardware encoder.
  It is declared like `counter_image`, with its own contract sized for the decoded frames, and the
  replayer decodes it on the GPU, so the graph receives the same host image tensor either way.
  Optional: a recording from a host without an encoder does not hold it, and a host without a
  hardware decoder cannot replay it. The app logs why and replays the other seven topics.

Topics are named **with their type**, all in one `McapReplayer` constructor call
(`make_replayer()`). Each `Topic<T>{"<topic>"}` checks the name against the recording's channels
and `T` against the schema the channel was recorded with (for `counter_image`, the payload type
nested in the tensor envelope). A missing topic or mismatched type is reported when the replayer is
constructed, before the graph is built and before any message is read.

A tensor-schema topic also takes the contract for its tensors in the topic itself, as
`Topic<holoscan::schema::ImageT>{"counter_image", ImageTensorFieldContract(kMaxImageBytes)}`; the
contract bounds the frame size, and the topic does not compile without one.

## Record, then replay

```bash
./isaac_os run mcap_logger_reference -- --duration-s=5
```

That prints the path of the file it wrote. Hand it back:

```bash
./isaac_os run mcap_replay_reference -- --input-path=<path printed above>
```

The app has one run mode, `replay` (the default), so `./isaac_os run mcap_replay_reference replay`
is equivalent. Expect every replayed topic (all eight when the encoded stream is replayed) in full
with nothing refused, and the run to take about as long as the recording did, since replay follows
the recorded timeline.

### Options

- `--input-path=<file>` (required) -- the recording to replay.
- `--playback-speed=<positive number>` (default `1.0`) -- recorded seconds per wall-clock second
  (`ReplayOptions::playback_speed`). At `2.0` the file replays in half the time, with every
  interval between messages halved.
- `--loop` -- sets `ReplayOptions::loop`: the recording starts over at its end. The run then has
  no message count to end on, so it continues until Ctrl+C, which stops the graph and prints the
  summary as a normal exit does.
- `--visualize` -- shows `counter_image`, and `counter_image_encoded` when it is replayed, each in
  a window of its own. Requires a display; without it, leave the flag off and the streams go to
  plain sinks. The windows and image layers are sized for
  the RGB8 frames `mcap_logger_reference` records, 640x480 and 1920x1080 (shown in a half-size
  window), because layer bounds must be declared before any frame is read.

  Rendering the window shares the machine with the replay, so with `--visualize` the summary's
  worst-case lag and spacing error can reach tens of milliseconds at 1x, on every topic at once
  because they share one pacing thread, and opening the window adds a small start-up delay. Leave
  the flag off when measuring timing. Replaying with `--visualize` at around 10x or faster
  (`--playback-speed=10`, 200 Hz of images from a 20 Hz stream) may also stop the graph partway
  through with a "receive borrow capacity is exhausted" error.
- `--help`, `-h` -- prints usage.

Without `--loop`, the app exits when replay reaches the end of the recording or fails.

## Files

- **`replay_graph.h`** -- the shortest thing to read before writing a graph of your own.
  `make_replayer()` is one `McapReplayer` constructor call naming every topic with its type, and
  `ReplayGraph::compose()` makes one `replayer_.attach<T>(*this, topic, sink->in)` call per topic
  (which imports the topic and does the `add_flow()`); `main` owns the replayer.
  Also holds `DrainOp`, the trivial consumer each non-visualized topic is wired to, and
  `ImageToHwcOp`, which gives a replayed frame the height x width x 3 shape Holoviz's RGB8 image
  layer reads. Recorded images are one flat byte buffer, so it only re-views the same bytes: no
  pixels are copied and no alpha channel is added.
- **`main.cc`** -- parses the flags, builds the replayer, compiles and runs the graph, then calls
  `replayer.start(plan.endpoint_catalog())`. Replay
  holds its first message back until every attached topic's consumer has subscribed; if one never
  does within 5 seconds the replayer logs an error naming it and sends nothing, and this app exits
  with a failure. If reading the recording fails partway through, the replayer logs an error and
  this app also exits with a failure, saying the replay is incomplete. Otherwise it waits for
  `replayer.finished()` (or Ctrl+C with `--loop`), and `replayer.stop()` joins the replay threads
  and logs the summary.

## Reading the summary

The summary logged on exit gives, per topic, the number of messages delivered, how late they were
on average and at worst, and how many were refused or could not be read back from the recording.
These are the only two ways a message fails to be published; a full buffer is not one, because the
reader blocks instead of discarding. A refusal means a message could not be published, usually
because its consumer's graph stopped partway through the run. The replayer will not slow down for a
consumer that is too slow to keep up; by default it drops the consumer's oldest unread messages
instead. This keeps the replay faithful to the recording and relies on the consumer to handle the
requested throughput. A consumer that never subscribed is reported
separately: the run never begins, and the summary says so instead of counting refusals.

Each topic with a recorded cadence also gets a line giving the rate it was replayed at beside the
rate it was recorded at, and how closely messages kept the recording's spacing, typically (the
standard deviation within a pass) and at worst (the largest spread within a pass, earliest to
latest landing). A delay common to a whole pass, including one on its first message alone, is not
counted there; lateness reports that. Unlike lateness, these compare the run against the
recording's timestamps rather than against the schedule replay computed, so a wrong schedule shows
up even when nothing was late. At a playback speed other than 1x the replayed rate is given in
recorded time with the delivered rate beside it, while the spacing is in delivered (wall-clock)
time. A topic with no recorded cadence, such as a transform recorded once, gets no such line.

A few messages reported late right after a run starts are expected. The first messages on a topic
can go out slower than the rest while replay warms up. A topic with late messages gets a line in
the summary giving how many went out more than 10 ms after they were due and their message numbers
on that topic, for example `2 went out more than 10 ms after they were due (messages #1, #4)`.
The numbers count the messages delivered on that topic, so they match the recording's own order
only while none were skipped; the topic's unreadable and refused counts show when some were. If
the numbers are only the first few and the topic's average lateness and spacing stay low, treat
it as start-up noise. Many late messages, or numbers spread through the run, mean the replay fell
behind during steady state: use a lower playback speed, a faster consumer or a less busy machine.
With `--loop` the summary only comes when the run ends, so each time the recording starts over the
replay also logs the late messages of the pass that just ended, at most once per second per topic.
A topic with a single message per pass has no restart to report at, so its late messages are listed
for the whole run.

The counts describe what this run did; a replay never learns how many messages the recording holds,
so the summary does not reconcile against the file.

A healthy run reports every topic delivered, nothing refused and nothing unreadable.
