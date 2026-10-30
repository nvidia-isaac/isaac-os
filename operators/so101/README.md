# SO-101 hardware I/O

Shared hardware operators for the real modes of
[Sinus Motion](../../applications/so101_sinus_motion/README.md#real-hardware) and
[Reach](../../applications/so101_reach/README.md#real-hardware).
Their controllers and motion-control paths differ.

## Hardware driver

The source and sink share one `std::shared_ptr<FeetechSerialDriver>` and serial connection.
The [driver](../../common/so101_driver/feetech_serial_driver.h) handles calibration,
serial communication, and unit conversion. Positions use radians and velocities use radians/s.
The driver class has no Holoscan dependency; its build target also provides YAML configuration
decoding through `holoscan::config_io`.

`initialize()` configures and homes the arm. Opening the driver is not a passive state read.

## State source

[`So101StateSourceOp`](so101_io_ops.h) uses `OnClock` with `period` in seconds (default `0.01`).
The period must be positive and finite. Both source and sink require the expected six-joint
calibration order.

The source emits `StateSampleCpuT` with positions, velocities, capture time, a monotonic
`source_sequence`, and a `fault_epoch`. A failed hardware read skips publication and increments
`fault_epoch`. After two seconds of continuous read failures, measured with a steady wall clock,
the source requests application completion.

## Command sink

[`So101CommandSinkOp`](so101_io_ops.h) uses `OnEach` to process incoming `JointCommandCpuT`
messages. Each of `position`, `velocity`, `effort`, `kp`, and `kd` must contain six finite values.
The sink sends only `position` to the driver. Servo gains, speed, and acceleration come from
the driver configuration.

Cancellation and shutdown request a measured-position hold. The hold can fail if feedback or
communication is unavailable; it is not a guaranteed stop mechanism.

> **Warning:** A successful hold leaves torque enabled. Support the arm, then switch off servo
> power before moving or recalibrating the joints by hand.

See [so101_io_ops.cc](so101_io_ops.cc) for validation, fault handling, and shutdown behavior.

## Application wiring

The [Sinus Motion real graph](../../applications/so101_sinus_motion/real/main.cc) connects the
shared I/O to a sine generator and bounded motion planner. The
[Reach real graph](../../applications/so101_reach/real/main.cc) uses a learned policy and
[delta motion controller](so101_delta_motion_control_op.h). Their application READMEs contain
the run commands, configuration, and hardware setup instructions.
