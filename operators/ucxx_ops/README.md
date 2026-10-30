# UCX operators

Typed UCX sender/receiver operators compatible with Isaac Sim's wire protocol.

## Wire format and graph payloads

The UCX boundary is separate from the graph schema identity:

| UCX wire message/tag name | Graph payload |
| --- | --- |
| `isaac.JointState` | Existing `isaac::JointStateCpuT` |
| `isaac.JointCommand` | Existing `isaac::JointCommandCpuT` |
| `isaac.Imu` | `holoscan::schema::ImuT` |
| `isaac.Image` | `holoscan::schema::ImageT` |
| `isaac.Time` | `holoscan::schema::TimeT` |

`messages/` generates only FlatBuffers wire tables/builders. Joint and IMU
wire envelopes import the SDK's `tensor.fbs`; Image and Time reuse SDK tables
directly. There is no Isaac tensor schema, generated-header patch, pack
forwarder, or generated graph descriptor for the wire envelopes.

`message_traits_inl.h` selects the wire root and codec;
`wire_codec.cc` implements the conversions. Joint codecs copy contiguous CPU
float32 bytes into/out of owned vectors using the SDK Tensor table. This avoids
unaligned float access and does not require optional acceleration/temperature
tensors that Isaac Sim omits. IMU converts its wire quaternion tensor to the SDK
quaternion; Image uses SDK Pack/UnPack and transfers CUDA pixels separately.
Capture time is populated from the wire timestamp; joint encoders put that
capture time back into the wire header. Messages without required timestamps
or with invalid joint tensors are rejected.

Legacy wire field order and UCX tag names are preserved independently of graph
schema names. The wire compatibility test decodes a fixture matching the removed
`isaac.Tensor` layout with `holoscan.schema.Tensor`.

## Runtime

`UcxxEndpoint` is a value-type handle sharing one UCX context/worker/endpoint.
`UcxxConnectOp` waits for the initial connection after graph startup. It defaults
to unlimited retries with a one-second interval; `UcxxConnectOptions` sets the
timeout and retry interval as `std::chrono::milliseconds`. Cancellation stops the wait, and
receiver initialization failures remain fatal. Established connections do not reconnect.
Receivers use a notification source for asynchronous readiness. Image receivers
copy completed UCX frames into SDK-owned CUDA tensor storage,
which manages consumer readiness and lifetime. This adds one device-to-device
copy per image; joint messages are unaffected. Graph outputs declare bounded
CUDA storage and applications explicitly bind the output device.
Image consumers must declare CUDA tensor input memory.

UCX/UCXX/RMM/rapids-logger come from the NVIDIA pip prebundle; its UCX runtime
must match the deployment image.

## Validation

The SO-101 sinus simulation app exercises joint receive/control/send and optional
camera receive/drop. Its tests cover bounded, continuous, and camera graph
compilation. Codec tests cover optional joint fields, materialized command bytes,
timestamps, malformed dtype, IMU, and Time; message tests guard wire compatibility.
Live simulation is a separate integration check, not part of the offline tests.
