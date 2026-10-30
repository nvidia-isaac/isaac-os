# ROS2 Bridge

Exchanges data between a Holoscan graph and an external ROS2 node, in-process. It is
**not** a `holoscan::Operator` — `Ros2Bridge` is a plain C++ object used in two phases around
your own `holoscan::Graph`:

```text
                    compose()                    compile()              run_async()
                        |                             |                      |
  App Graph::compose()  |  from_ros2 / to_ros2       |                      |
       calls  --------->|  import_topic/export_topic  |                      |
                        |  created lazily, memoized   |                      |
                        v                             v                      v
                  [ Graph authoring ] ---------> [ ExecutionPlan ] -----> [ RunSession ]
                                                        |
                                                        | plan.endpoint_catalog()
                                                        v
                                              Ros2Bridge::start(catalog)
                                                        |
                                          owns: rclcpp::Node + executor thread
                                                ChannelPublisher/Subscriber per wired topic
                                                relay threads
```

Topics wired via `from_ros2` flow **in**: an internal ROS2 subscription converts each message
and emits it onto a Holoscan port your graph's operators consume. Topics wired via `to_ros2`
flow **out**: your graph emits onto a port, a poll thread converts it and republishes onto ROS2.

## Basic usage

1. Construct the bridge, giving it the name its `rclcpp::Node` should run under:

   ```cpp
   Ros2Bridge<> bridge("my_app_bridge");
   ```

2. Give your `Graph` a reference to the bridge as a constructor member, then wire ports to your
   operators in `compose()` with `from_ros2`/`to_ros2`: both take the ROS2 message type and the
   HSDK type as explicit template arguments (`from_ros2<Ros2T, HsdkT>`/`to_ros2<Ros2T, HsdkT>`),
   resolved against `ConverterT::SupportedPairs` entirely at compile time -- an unregistered or
   ambiguous `(Ros2T, HsdkT)` pair is a compile error, not a runtime throw. `from_ros2` takes an
   `Input<HsdkT>` and wires ingress, `to_ros2` takes an `Output<HsdkT>` and wires egress. Both
   take the same argument order -- graph, port, a `Ros2TopicSpec{port_name, ros2_topic,
   max_tensor_bytes}` -- so a call reads the same either way. Each call lazily creates (and
   memoizes) the underlying HoloMQ topic on first use for its `port_name`:

   ```cpp
   class MyGraph final : public holoscan::Graph {
    public:
     explicit MyGraph(Ros2Bridge<>& bridge) : Graph("my_graph"), bridge_(bridge) {}

    protected:
     void compose() override {
       auto my_op = op<MyOperator>("my_op");
       bridge_.from_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
           *this, my_op->joint_state_in,
           {.port_name = "joint_state", .ros2_topic = "/joint_states"});
       bridge_.to_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
           *this, my_op->joint_command_out,
           {.port_name = "joint_command", .ros2_topic = "/joint_commands"});
     }

    private:
     Ros2Bridge<>& bridge_;
   };
   ```

   Calling `from_ros2` again for an ingress `port_name` already wired (with the same `HsdkT`)
   reuses the same imported topic (memoization) rather than re-importing it, but the resulting
   plan does not compile as-is: an `ImportedTopic`'s inherited external-channel contract admits
   exactly one boundary-to-`Operator` relation, so fanning it out to two consumers is rejected at
   `compile()` with `EXTERNAL_CHANNEL_PLAN_INVALID` — confirmed by
   `test_ros2_bridge_wiring.cc`'s `RejectsFanOutWithoutAManuallyAuthoredPartition`, which asserts
   on that exact diagnostic token rather than just "compile failed." There is no manual-partition
   workaround: binding a `Graph::partition()` around the consumers does not avoid this rejection.
   Egress has no such path to even attempt: `Graph::add_flow` itself rejects a second edge into
   an already-wired `Input` outright (`GRAPH_INPUT_FANIN_UNSUPPORTED`, thrown during `compose()`
   and surfaced to the caller as a generic wrapping `GRAPH_COMPOSE_FAILED` — the specific inner
   token isn't visible past that point, per the equivalent egress test), so a `port_name` wired
   via `to_ros2` supports exactly one producer.

   The egress direction supports its own kind of fan-out instead: publishing the same `Output`
   onto multiple ROS2 topics. Call `to_ros2` once per distinct `port_name`, all with the same
   `source`:

   ```cpp
   bridge_.to_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
       *this, my_op->joint_command_out,
       {.port_name = "joint_command_a", .ros2_topic = "/joint_commands_a"});
   bridge_.to_ros2<sensor_msgs::msg::JointState, isaac::JointStateCpuT>(
       *this, my_op->joint_command_out,
       {.port_name = "joint_command_b", .ros2_topic = "/joint_commands_b"});
   ```

   Each `port_name` gets its own independently created `ExportedTopic`, so there's no
   memoization or partition concern here at all: this is an ordinary Holoscan `Output` fanning
   out to multiple `Input`s, and the producing `Operator` already owns one execution thread for
   that `Output` regardless of how many edges leave it.

   Construct the graph with the bridge you made in step 1:

   ```cpp
   MyGraph graph(bridge);
   ```

3. After `holoscan::compile()`, start the bridge's ROS2-facing runtime and run the graph:

   ```cpp
   const holoscan::ExecutionPlan plan = holoscan::compile(graph);
   bridge.start(plan.endpoint_catalog());
   auto run_session = holoscan::run_async(plan);

   // ... application runs ...

   run_session.request_stop();
   run_session.wait();
   // bridge.stop() runs implicitly here via ~Ros2Bridge(), or call it explicitly first.
   ```

**Teardown ordering matters:** stop the Holoscan `RunSession` (`request_stop()`/`wait()`)
before, or concurrently with, stopping the bridge (`bridge.stop()`, or letting it go out of
scope) — never after. The bridge's poll threads and ROS2 subscription callbacks keep calling
into the graph's channels for as long as it's started, so stopping it first while the graph
is still running tears those channels out from under a live graph. See
`tests/test_ros2_bridge_roundtrip.cc` for a complete, working example of this whole sequence.

**The bridge owns the process's ROS2 context.** `start()` initializes `rclcpp` for the process if
nothing has yet, and `~Ros2Bridge()` shuts it down after `stop()`. The shutdown is required, not
just tidy: with the Lyrical CUDA buffer backend loaded, a process that leaves it to the static
default context at exit crashes while unloading the backend plugin. So use one bridge per process,
and create any other `rclcpp` node in the process (such as an observer node) after the bridge so it
is destroyed first or just have other nodes in another process. Other ROS2 processes are unaffected,
since the context is per process. `Ros2Bridge` can be neither copied nor moved.

## Built-in converters

`DefaultRos2Converter` registers these `(ROS2 type, hsdk type)` pairs out of the box:

| ROS2 type | hsdk type |
| --- | --- |
| `sensor_msgs::msg::JointState` | `isaac.JointStateCpu` |
| `sensor_msgs::msg::Imu` | `holoscan.schema.Imu` |
| `geometry_msgs::msg::Pose` / `PoseStamped` | `isaac.PoseCpu` |
| `geometry_msgs::msg::Twist` / `TwistStamped` | `isaac.TwistCpu` |
| `geometry_msgs::msg::Transform` / `TransformStamped` | `isaac.TransformCpu` |
| `nav_msgs::msg::Odometry` | `isaac.OdometryCpu` |
| `sensor_msgs::msg::CameraInfo` | `isaac.CameraInfoCpu` |
| `sensor_msgs::msg::Image` (tensor-backed) | `holoscan.schema.Image` |

The hsdk type column gives the schema name. In C++ code, pass the generated FlatBuffers object
type, which is the schema name with a `T` suffix: for example `holoscan::schema::ImageT` for
`holoscan.schema.Image`, and `isaac::PoseCpuT` for `isaac.PoseCpu`.

The non-`Stamped` variants leave the hsdk type's optional `header` field unset on `to_hsdk()`
and ignore it on `to_ros2()`; the `Stamped` variants populate/consume it.

## Adding a custom converter

The set of `(ROS2 type, hsdk type)` pairs a bridge understands is fixed at compile time via
its `ConverterT` template parameter (`Ros2Bridge<ConverterT = DefaultRos2Converter>`). Adding
a new pair never requires editing this library's files:

1. Write a `Converter<Ros2T, HsdkT>` specialization (`converter.h` has the built-in
   `sensor_msgs::msg::JointState <-> isaac.JointStateCpu` pair as a model). It needs
   `Ros2Type`/`HsdkType` aliases (what the compile-time match in step 3 keys on) and
   `to_hsdk`/`to_ros2` conversion functions:

   ```cpp
   namespace isaac::os::ros2_bridge {

   template <>
   struct Converter<my_msgs::msg::Custom, my::CustomT> {
     using Ros2Type = my_msgs::msg::Custom;
     using HsdkType = my::CustomT;

     static my::CustomT to_hsdk(const my_msgs::msg::Custom& msg) { /* ... */ }
     static my_msgs::msg::Custom to_ros2(const my::CustomT& msg) { /* ... */ }
   };

   }  // namespace isaac::os::ros2_bridge
   ```

2. Extend `DefaultRos2Converter`'s registered set with `tuple_cat`, rather than editing it:

   ```cpp
   struct MyConverter : isaac::os::ros2_bridge::DefaultRos2Converter {
     using SupportedPairs = decltype(std::tuple_cat(
         DefaultRos2Converter::SupportedPairs{},
         std::tuple<isaac::os::ros2_bridge::Converter<my_msgs::msg::Custom, my::CustomT>>{}));
   };
   ```

3. Use it as the bridge's template argument:

   ```cpp
   Ros2Bridge<MyConverter> bridge("my_app_bridge");
   ```

Your `from_ros2`/`to_ros2` calls' `<Ros2T, HsdkT>` template arguments then resolve against
`MyConverter`'s full set (the built-ins plus yours) at compile time via a `static_assert` in
`create_graph_topic` (`ros2_bridge.h`). An unregistered or ambiguous pair fails compilation,
not at runtime — a compile error is clearer than a runtime throw and ensures type safety is
verified before the graph runs.

Whether you write a plain or a tensor converter, the converter is checked with C++ concepts
(`converter_concepts.h`) when it is wired, so a mistake fails with a short message instead of a
template backtrace. A plain converter (the HSDK type is a flatbuffer) must provide `Ros2Type` and
`HsdkType` aliases, `to_hsdk(const Ros2Type&)` and `to_ros2(const HsdkType&)` (which returns the
message, or a `holoscan::expected` of it when the conversion can fail). A tensor converter (the HSDK
type is a descriptor-tensor-schema type) must provide those aliases plus a `Payload` alias,
`contract(const Ros2TopicSpec&)`, `to_tensors(const std::shared_ptr<Ros2Type>&, const
Ros2TopicSpec&)` and `from_tensors(const ChannelResolvedTensorBundle<Payload>&, const
Ros2TopicSpec&)`. Registering a plain converter for a tensor type, or the reverse, is the usual
mistake and has its own message.

A tensor converter does not have to work out where memory should go. In a CUDA build, two helpers
(`ros_buffer_ingress.h` and `ros_buffer_egress.h`) place the data as the topic's spec asks:

```cpp
// ROS to Holoscan: `msg` is the std::shared_ptr to the message, `msg->data` its rosidl::Buffer.
auto tensor = RosBufferToTensor(msg, msg->data, spec);

// Holoscan to ROS: copies `tensor` into the message's buffer.
auto status = TensorToRosBuffer(tensor, spec, ros_msg.data);
```

Both return a `holoscan::expected`, so a placement that cannot work comes back as an error you can
return from the converter. For example, `TensorToRosBuffer` refuses a device tensor that has no
readiness stream, because nothing would order the read after the producer's work. Call them from
the converter functions, which the bridge runs on threads that already have the pool GPU current.
A conversion that needs the CUDA pool (a copy into a pooled buffer, or any CUDA ingress) returns an
error instead of guessing a GPU if the pool has not been bound yet (the bridge binds it in
`start()`). The rules behind them are two pure functions in `memory_transfer.h`, `PlanIngress` and
`PlanEgress`, which you can also call to ask what a given combination of source and topic kind
would do.

A tensor converter's `contract()` for a device topic must allow the transport to copy:
`ImageTensorFieldContract` already sets `.transfer = TensorTransferPolicy{.foreign_emit_policy =
InterfaceTensorEmitPolicy::kCopyIfNeeded}`, so use it (or set that policy yourself). A tensor the
bridge wraps around ROS-owned memory is foreign to the Holoscan transport, which can only share
memory from its own pool, so under the default policy (`kRequireExportable`) the device topic is
rejected when the graph is compiled, with `PHYSICAL_TENSOR_LOGICAL_REQUIREMENTS_INVALID`, whose
diagnostic asks for a copy-capable transfer policy such as `kCopyIfNeeded`.

## Tensor-carrying messages

Some `(ROS2 type, hsdk type)` pairs carry pixel/tensor data rather than a plain flatbuffer
struct. These are registered with a `TensorConverter<Ros2T, HsdkT>` specialization
(`tensor_converter.h`) instead of the plain `Converter<Ros2T, HsdkT>` from the previous
section.

The two trait shapes share the `Ros2Type`/`HsdkType` aliases, but
`TensorConverter` replaces `Converter`'s single-value `to_hsdk()`/`to_ros2()` pair with a
different set of members shaped for the tensor-schema transport:

- **`contract()`** — the most structurally distinctive part: a per-field
  `TensorBoundaryContract` describing each tensor field's expected representation (memory
  kind, dtype, rank, stride policy) and size bound. This is what lets the bridge attach
  through `ChannelRuntime::create_tensor_schema_publisher`/`create_tensor_schema_subscriber`
  and `Graph::import_topic`/`export_topic`'s descriptor-tensor-schema overload, instead of the
  plain scalar-payload path a `Converter` pair uses.
- **`to_tensors()`** — builds one or more `holoscan::Tensor`s plus the schema's serialized
  metadata bytes from a ROS2 message, in place of `to_hsdk()`'s single returned value.
- **`from_tensors()`** — the reverse: reads a resolved tensor bundle (metadata plus tensor
  handles) back into a ROS2 message, in place of `to_ros2()`'s single-value conversion.

`TensorConverter` specializations register into the same
`DefaultRos2Converter::SupportedPairs`/`tuple_cat` extension mechanism as plain `Converter`
pairs (see "Adding a custom converter" above) — no separate registration path, and
`dispatch_by_type_pair` doesn't care which trait shape it resolves to.

`sensor_msgs/msg/Image <-> holoscan.schema.Image` is the one built-in tensor-backed pair, and
serves as the model (`tensor_converter.h`) for registering another. `holoscan.schema.Image`
(`holoscan::schema::ImageT`) isn't a project-local schema — it's the Holoscan SDK's own
descriptor-tensor-schema type from `holoscan::sensor_io`, so this bridge reuses the SDK's own
tensor-carrying wire format rather than defining its own. It only populates the fields ROS2's
`Image` message actually has (`height`/`width`/`encoding`/`data`); `holoscan::schema::ImageT`'s
other fields (color space, exposure, gain, region of interest, plane layouts) are left at
their SDK-provided defaults.

Tensor-backed topics can also set an optional `max_tensor_bytes` field on `Ros2TopicSpec`. This
is not a per-message copy limit — HoloMQ never serializes tensor bytes into the schema envelope,
only the descriptor crosses — it sizes the transport's preallocated backing capacity for the
tensor field, checked against the peer's own declared contract when they connect. It defaults to
8 MiB when omitted and has no effect on plain (non-tensor) topics:

```cpp
bridge_.from_ros2<sensor_msgs::msg::Image, holoscan::schema::ImageT>(
    *this, my_op->image_in,
    {.port_name = "image_in", .ros2_topic = "/camera/image", .max_tensor_bytes = 16'777'216U});
```

## QoS

`Ros2TopicSpec` has a `qos` field (an `rclcpp::QoS`) applied to the ROS2 subscription created by
`from_ros2()` or the publisher created by `to_ros2()`. It defaults to `rclcpp::QoS(10)`:
reliable, volatile, keep-last 10. Set it to match the peer:

```cpp
bridge_.from_ros2<sensor_msgs::msg::Image, holoscan::schema::ImageT>(
    *this, my_op->image_in,
    {.port_name = "camera_in",
     .ros2_topic = "/camera/image_raw",
     .qos = rclcpp::SensorDataQoS()});
```

A reliable subscription does not match a best-effort publisher (typical of camera drivers), and
the topic stays silent; rclcpp logs an "incompatible QoS" warning in that case. Use
`rclcpp::SensorDataQoS()` or `.best_effort()` for such peers. If the same ingress port is wired
by more than one `from_ros2()` call, the last call's `qos` is used.

## Capture time (`capture_time_clock`)

Event-time selectors such as `LatestWithin`, `Exact`, `OnAligned` and `BracketedWindow` only select
samples that carry a capture time. By default `from_ros2()` emits none, so a graph that samples an
imported topic with one of them fails at `compile()` with `TEMPORAL_CAPTURE_TIME_REQUIRED`.
Arrival-driven triggers (`OnEach`, `OnAnyInput`) are unaffected.

Set `capture_time_clock` on the `Ros2TopicSpec` to the clock key that `add_clock` returned. The
topic then requires a capture time in that clock, and every message is emitted with its
`header.stamp` as the capture time. A message whose stamp is not a valid time (negative seconds, or
nanoseconds of a second or more) is dropped with a warning:

```cpp
void compose() override {
  const holoscan::ClockKey clock = add_clock<holoscan::RealtimeClock>(
      "clock", holoscan::RealtimeClock::Config{.use_time_since_epoch = true});
  set_default_clock(clock);
  // ...
  bridge.from_ros2<geometry_msgs::msg::PoseStamped, isaac::PoseCpuT>(
      *this, consumer->pose_in,
      {.port_name = "pose", .ros2_topic = "/pose", .capture_time_clock = clock});
}
```

The bridge cannot check that the clock matches the publisher's. A ROS2 `header.stamp` carries no
clock identity, so the clock you name must count the same time. For wall-clock stamps that is a
`RealtimeClock` with `use_time_since_epoch = true`; the default `RealtimeClock` counts from its
own construction, so every stamp looks far in the future and is never selected, with no error. The
same applies to `use_sim_time`.

A message type with no `header.stamp` (for example `Pose`) can set the option too. The bridge then
stamps each message with the ROS2 time it was received and logs a warning when the topic is wired,
since that is arrival time, not capture time. This lets a consumer using `DrainAll` bind on the
import without stamped messages.

## GPU-resident images (`memory_kind`)

`sensor_msgs/msg/Image` carries its pixels in a `rosidl::Buffer`, which can live in GPU memory.
Set `memory_kind = Ros2MemoryKind::kCudaDevice` on a topic's `Ros2TopicSpec` to move those pixels
between ROS 2 and a Holoscan graph without a host round trip. The default (`kHost`) is
unchanged, and `memory_kind` only applies to tensor-schema topics such as `ImageT`. It only pays
off when both ends are on the GPU: a GPU ROS producer feeding GPU operators, or the reverse. The
CUDA buffer backend must be present, in the build and in every ROS process involved, for device
topics to work as described here.

```cpp
Ros2Bridge<> bridge("my_bridge", /*pool_device=*/0);  // the one GPU every device topic uses

bridge.from_ros2<sensor_msgs::msg::Image, holoscan::schema::ImageT>(
    *this, detector->image_in,
    {.port_name = "camera_in", .ros2_topic = "/camera/image",
     .memory_kind = Ros2MemoryKind::kCudaDevice});
```

What each direction does:

| Direction | Data | Topic kind | Result |
| --- | --- | --- | --- |
| ROS to Holoscan | CUDA buffer on the pool GPU | `kCudaDevice` | the tensor wraps the message's own buffer (the bridge copies nothing) |
| ROS to Holoscan | host buffer | `kCudaDevice` | copied to a pooled CUDA buffer first |
| ROS to Holoscan | CUDA buffer | `kHost` | copied down to host memory (only if a CUDA buffer still reaches a host topic; the bridge's own subscription already asks ROS for host buffers) |
| Holoscan to ROS | device tensor | `kCudaDevice` | copied device-to-device into a pooled CUDA buffer, then published |
| Holoscan to ROS | host tensor | `kCudaDevice` | copied host-to-device into a pooled CUDA buffer, then published |
| Holoscan to ROS | device tensor | `kHost` | copied down to host memory, then published |

On a `kHost` topic the subscription asks ROS for host buffers, so a GPU publisher's frames are
copied down by the ROS transport before the bridge sees them.

### Where the ROS publisher puts the pixels (`publisher_memory`)

A `from_ros2` device topic also needs to know what its ROS publisher sends. The default,
`publisher_memory = Ros2MemoryKind::kCudaDevice`, suits a publisher that sends GPU memory. For a
publisher that sends host memory, such as a typical camera driver, set `kHost`:

```cpp
bridge.from_ros2<sensor_msgs::msg::Image, holoscan::schema::ImageT>(
    *this, detector->image_in,
    {.port_name = "camera_in", .ros2_topic = "/camera/image_raw",
     .memory_kind = Ros2MemoryKind::kCudaDevice,
     .publisher_memory = Ros2MemoryKind::kHost});
```

The bridge then asks ROS for host buffers and copies each frame to the GPU once. We recommend
setting `publisher_memory = kHost` whenever the publisher sends host memory. Without it the
subscription asks ROS for CUDA buffers, and receiving a host publisher's frames that way is much
slower: in our 4K measurements several times the latency and about three times the CPU, and the
publisher pays part of that CPU.

The penalty is largest when the publisher is in another process, which is the usual case for a
camera driver: in a 4K test on one machine, 10 Hz from a separate process took about 67 to 83 ms
per frame with the default, against 19 to 22 ms with `kHost`. A publisher in the bridge's own
process, with intra-process communication on both nodes (see the note on nodes in the same process
under "Things to note"), mostly avoids it, but `kHost` is still the setting that matches a host publisher.
Small frames (640x480) showed no meaningful difference.

A wrong value never changes the result, only the speed. If a topic keeps the default but its
publisher sends host buffers, the bridge logs one warning for that topic naming the field to
change; the same warning appears when a publisher meant to send GPU memory lacks the CUDA buffer
backend, so check that too. The opposite mistake, `kHost` for a publisher that sends GPU memory,
is not reported: the subscription then receives host buffers, so ROS copies each frame down before
the bridge sees it and the bridge copies it back up. `publisher_memory` applies to `from_ros2`
only. A `to_ros2` topic publishes the memory kind in `memory_kind`, and each ROS subscriber
chooses what it accepts.

In a build without the CUDA buffer backend (the `cuda_buffer` package was not found at configure
time), wiring a `kCudaDevice` topic throws and names the topic.

In current SDK versions the transport stages one device-to-device copy of each frame. That is a
copy on the GPU, not a trip through host memory.

### One GPU per bridge

The ROS CUDA buffer pool is one process-wide singleton bound to a single GPU, and the ROS buffer
backend can only import a CUDA buffer when it is on the GPU that the receiving thread has current.
So the bridge puts every device topic on one GPU, the `pool_device` you pass to the constructor
(default 0), and makes it current on the threads it starts.

- A device topic's ROS publisher must publish from that GPU. A buffer from another GPU reaches
  the bridge as an empty CUDA buffer and is dropped with a log naming the topic.
- A host message on a device topic is copied to that GPU. A device tensor on a different GPU is
  copied across GPUs before it is published.
- If other code in the process created CUDA buffers on a different GPU first, `start()` throws and
  names both GPUs. Start the bridge before that code, or construct the bridge with that GPU.

Your graph's operators are not limited to that GPU. The bridge only fixes where its own edges
live: an operator on another GPU can copy a frame it receives from the bridge, and the operator
that produces an egress tensor can run on any GPU (its output binding says which).

### Compile options a device topic needs

The bridge never calls `holoscan::compile()`, and the SDK refuses to guess a CUDA device, so every
`kCudaDevice` topic needs a device binding in the `CompileOptions` you pass to `compile()`;
without it `compile()` fails with `IMPLICIT_DEVICE_ZERO_REJECTED`.

- **Ingress** (`from_ros2`): bind the imported topic by its endpoint name, which is the `port_name`
  from the spec, to the bridge's pool device:
  `options.deployment.bind_imported_topic_device({.endpoint_name = "camera_in",
  .device = holoscan::DeviceId{bridge.pool_device()}})`.
- **Egress** (`to_ros2`): bind the operator whose output feeds the exported topic yourself, as for
  any device tensor producer:
  `options.deployment.bind_tensor_output_device({.operator_path = "my_op", .output_port = "image",
  .device = holoscan::DeviceId{0}})`.

### Things to note

- **External ROS subscribers must opt in.** This is not the bridge's doing: any `rclcpp`
  subscription of your own to a bridge-published device topic receives host memory unless it sets
  `SubscriptionOptions::acceptable_buffer_backends` to `"cuda"` (or `"any"`).
- **ROS nodes in the same process as the bridge need intra-process communication.** A test or a
  demo that publishes or subscribes CUDA buffers from its own node in the bridge's process sends
  them through the ROS CUDA buffer backend's same-process IPC path, which blocks every publish on
  the GPU and logs "Same-process CUDA IPC requires cudaEventSynchronize". Construct the bridge with
  `rclcpp::NodeOptions().use_intra_process_comms(true)` (its third constructor argument) and give
  the other nodes the same option; the buffers are then handed over in place. Without it, release
  a node's publisher before the subscription it feeds in the same process, or the ROS context
  prints "Failed to delete participant" at shutdown. Peers in other processes need neither.
- **Only pooled buffers stay on the GPU.** A buffer must come from
  `cuda_buffer_backend::allocate_buffer()`. A foreign `cudaMalloc` pointer wrapped in a
  `CudaBuffer` may be converted to host memory in transit.
- **A publisher on the wrong GPU.** It is dropped with a log, not delivered (see above).
- **A large host frame can delay the other topics.** All ingress topics share one executor
  thread. A very large host frame on a device topic (4K) takes a few milliseconds on that thread
  for the copy to the GPU, so other ingress topics on the same bridge can see a delay: the
  measured p99 latency of a 640x480 topic went from about 2 ms to about 5 ms. The delay is much
  larger, about 37 ms in the same test, when the publisher sends host memory but the topic's
  `publisher_memory` is left at the default `kCudaDevice`; the bridge was not told it is a host
  publisher (see `publisher_memory` above). A frame published from the GPU does not cause this.
- **Forwarding an imported device frame straight to an exported topic can stall.** If an operator
  emits the very frame it received from a device import on a device export, without producing a
  new one, the import's staging pool can run out of free slots. The ingress then drops every
  following frame, with only a log message for the first few. Operators that write a new output
  frame, as most processing does, are not affected. For a pure passthrough, copy the frame into
  memory the operator owns before emitting it.
- **One ROS context per process.** The bridge initializes ROS and shuts it down when it is
  destroyed; do not initialize and shut down ROS repeatedly in one process with CUDA buffers.

## Tests

`tests/` covers each piece independently: `test_converter.cc` and `test_tensor_converter.cc`
(conversion correctness, plus the converter concept `static_assert`s in `test_converter.cc`),
`test_ros2_bridge_wiring.cc` (compose-time `from_ros2`/`to_ros2`
wiring, including lazy/memoized construction and that an unpartitioned fan-out is rejected — see
the caveat above), `test_ros2_bridge_session.cc` (start/stop lifecycle, including that `stop()`
actually tears down ROS2-side wiring, not just that construction succeeds), and
`test_ros2_bridge_roundtrip.cc` (a full ROS2 publish -> Holoscan graph -> ROS2 subscribe
integration test).

The GPU-resident image support adds:

- `test_memory_transfer.cc`: the ingress and egress placement rules, and the check that a device
  topic needs the CUDA buffer backend.
- `compile_fail_plain_missing_to_ros2.cc` and `compile_fail_plain_on_tensor_type.cc`: targets that
  must not compile, run as tests that pass only when the build prints the expected concept message.
- `test_cuda_device_utils.cc`: scoped device switching and per-thread streams.
- `test_cuda_pool_device.cc`: binding the CUDA buffer pool to a GPU.
- `test_ros_buffer_egress.cc`: Holoscan tensor to ROS buffer for each placement.
- `test_ros_buffer_ingress.cc`: ROS buffer to Holoscan tensor for each placement.
- `test_ros2_bridge_cuda_roundtrip.cc`: a CUDA image through a graph and back out as a CUDA buffer.

The five CUDA tests (`test_cuda_device_utils.cc` through `test_ros2_bridge_cuda_roundtrip.cc`) are
built only when the CUDA buffer backend is found and skip without a CUDA device. The tests that
need two GPUs (in `test_cuda_device_utils.cc`, `test_cuda_pool_device.cc` and
`test_ros_buffer_egress.cc`) skip below two devices.
