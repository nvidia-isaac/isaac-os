# SO-101 Sinus Motion

Moves all six SO-101 joints with a sine target. Mock and real modes run at 100 Hz;
`SimpleMotionPlannerOp` clamps each command to the measured position ± `max_step_size`.

| Mode | Behavior |
| --- | --- |
| [`mock`](#mock) | Synthetic joint state and command validation |
| [`sim`](#isaac-sim) | Isaac Sim over UCX; separate graph without the motion planner |
| [`real`](#real-hardware) | Feetech joint-state reads and position commands |
| [`profile_real`](#measure-servo-speed-and-acceleration) | Bounded servo speed/acceleration diagnostic |

See [application modes](../../docs/explanation/application_modes.mdx) for clock and I/O differences.

[Benchmark](BENCHMARK.md)

## Run

Run from the repository root:

```bash
./isaac_os build so101_sinus_motion
./isaac_os modes so101_sinus_motion
```

<Tabs>

<Tab title="Mock" id="mock-run-tab">

### Mock

```bash
./isaac_os run so101_sinus_motion mock
```

The default run stops after 100 validated commands, about one second at 100 Hz.
For a longer run:

```bash
./isaac_os run so101_sinus_motion mock \
  --run-args='--message-count=11000'
```

`--message-count` is the number of completed sink commands, one per joint-state
sample, so 11,000 commands take about 110 s at 100 Hz. The default timeout is
10 seconds plus three times the expected duration; override it with
`--completion-timeout-seconds`.

Measure latency and cadence with [Profile an Application](../../docs/how-to/profile_an_application.mdx)
and the app [Benchmark](BENCHMARK.md).

</Tab>

<Tab title="Isaac Sim" id="isaac-sim-run-tab">

### Isaac Sim

Sends sinusoidal joint-position commands to the SO-101 arm over UCX. The sine phase
follows incoming simulator timestamps, so it pauses and scales with simulator time.
Messages without a header or timestamp are rejected.

In one terminal, start Isaac Sim with the SO-101 UCX scene:

```bash
./isaac_sim run --scene operators/so101/assets/so101_ucx_with_env.usda
```

Launch the application from the repository root in another terminal:

```bash
./isaac_os run so101_sinus_motion sim
```

The simulator starts playback when the application connects. Expect sinusoidal
joint motion and a Holoviz camera window. Stop the controller with Ctrl+C before
stopping the simulator. See the
[Isaac Sim guide](../../docs/how-to/set_up_isaac_sim.mdx) for additional launcher
options. The `sim` mode connects to `sim_so101.hostname` and
`sim_so101.port`; if you change the port in a custom YAML, also pass that port to
`./isaac_sim run` through `-- --port`.

The mode's metadata supplies
`applications/so101_sinus_motion/sim/config/so101_sinus_motion_sim_config.yaml`
automatically, so no `--config-path` is needed for the default configuration.
Pass application options through `--run-args`.

<Note>
Holoviz requires a working display and does not disable itself automatically.
For headless runs, including hosts configured with
[Run without a desktop](../../docs/how-to/optimize_your_host.mdx#run-without-a-desktop),
set `sim_so101.enable_camera: false` in the YAML before launching the application.
</Note>

#### Validation

Compile the graph without connecting to Isaac Sim:

```bash
./isaac_os run so101_sinus_motion sim \
  --run-args='--no-connect'
```

Stop after 100 joint-state samples:

```bash
./isaac_os run so101_sinus_motion sim \
  --run-args='--message-count=100'
```

#### Custom configuration

Copy the default YAML into the repository, edit it, and pass its
path through `--run-args`. The explicit path overrides the metadata default:

```bash
cp applications/so101_sinus_motion/sim/config/so101_sinus_motion_sim_config.yaml \
  so101_sim_custom.yaml
# Edit so101_sim_custom.yaml for your simulation.
./isaac_os run so101_sinus_motion sim \
  --run-args='--config-path=so101_sim_custom.yaml'
```

Relative paths resolve from the repository root inside the container. Files
outside the repository need to be mounted into the container and referenced by
their container paths.

<Warning>

Isaac Sim up to 6.2.1.0 ignores PD gains (`kp` and `kd`) received through UCX.
Changing command gains does not change simulated stiffness or damping; tune the USD
articulation drives instead. See [joint-command gains](../../docs/how-to/set_up_isaac_sim.mdx#configure-joint-gains)
for details.

</Warning>

</Tab>

<Tab title="Real robot" id="real-hardware-run-tab">

### Real hardware

1. Calibrate the follower arm. This application cannot calibrate the arm; it only
   consumes the JSON produced by the
   [LeRobot SO-101 calibration guide](https://huggingface.co/docs/lerobot/so101).
   Follow the calibration section of that guide before continuing.

   > **Note:** The file lands in
   > `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<robot.id>.json`.
   > The directory is `so_follower`, not `so101_follower`: LeRobot 0.6.1 shares one
   > calibration directory across the SO-100 and SO-101 followers.

   Pose the five arm joints at true mid-travel and hold the gripper roughly
   half-open before pressing Enter. During the range step, move every joint through
   its full usable travel, including the gripper. The calibrated center becomes
   encoder count 2048 for each arm joint; the driver instead maps the gripper's
   swept closed endpoint to coordinate `0`, regardless of motor direction. A
   non-gripper joint centered near one of its hard stops has its sine clamped on
   that side and visibly flattens. The driver also clamps any `home_position` that
   falls outside a joint's calibrated range.

   Calibrate *before* running the application. After a successful run, torque stays
   enabled (see the warning below), and a torqued joint cannot be hand-posed.
   Power-cycle the arm before recalibrating.

2. Grant your user serial-port access on the **host**:

   ```bash
   sudo setfacl -m u:$USER:rw /dev/ttyACM0
   ```

   > **Note:** `/dev/ttyACM*` numbering is not stable. List the nodes with
   > `ls /dev/ttyACM*`, confirm which one is the follower, and use that node both in
   > the ACL command and for `real_robot.serial_port`. `udev` recreates the node on
   > reboot, unplug/replug, or a power-cycle of the arm, so re-run `setfacl` after
   > those events. A missing ACL fails immediately with `open:: Permission denied`.

3. Copy the config. Real mode accepts `--config-path` and `--message-count`, with
   no `--override`. It reads `real_robot` from YAML before graph compilation.
   Keep local paths in the copy:

   ```bash
   mkdir -p build
   cp applications/so101_sinus_motion/config/so101_sinus_motion_config.yaml \
     build/so101-sinus-real.yaml
   ```

4. In the YAML copy, set `real_robot.calibration_file` to the **container** path
   `/run/isaac-os/so101-follower-calibration.json`, and verify
   `real_robot.serial_port` against the node found above. For the first hardware pass
   also set `sin_amplitude: 0.05`. Use absolute paths: `./isaac_os` sets
   `HOME=/workspace/isaac_os` inside the container, so a host `$HOME/...` path in
   the YAML is not visible there. The tracked calibration value contains
   `PLACEHOLDER`, which the config decoder rejects outright.

5. Mount the calibration file and run. Keep the workspace clear for the initial
   homing move to the all-zero sine mean. Support the arm through startup.

   > **Warning:** If initialization fails after torque is enabled, the driver tries
   > to hold healthy servos at measured positions and verifies the hold. It reports
   > the original error and stops. If feedback or hold verification fails, torque
   > is disabled and the arm can go limp. A verified hold leaves torque enabled.

   ```bash
   export ISAAC_OS_EXTRA_DOCKER_RUN_ARGS="$ISAAC_OS_EXTRA_DOCKER_RUN_ARGS -v $HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/<robot.id>.json:/run/isaac-os/so101-follower-calibration.json:ro"

   ./isaac_os run so101_sinus_motion real \
     --run-args="--config-path=build/so101-sinus-real.yaml"
   ```

   The host shell expands `$HOME` before Docker sees the mount. Append to
   `ISAAC_OS_EXTRA_DOCKER_RUN_ARGS` rather than overwriting it, so existing cache
   mounts survive.

> **Warning:** After a successful run, or after failed initialization that ends in
> a verified hold, motors remain in torque mode so the arm does not free-fall. Do
> not move the joints by hand while torque is enabled. Support the arm before
> switching off servo power, then re-run `setfacl` before the next start.

Stop with Ctrl+C to hold the measured position. The first run uses `sin_amplitude: 0.05`;
expect small oscillations about the sine mean.

</Tab>

</Tabs>

## Architecture

Mock and real share this graph:

```mermaid
flowchart LR
    Src([Joint-state source<br/>100 Hz]):::hw
    SM([SinusMotionOp]):::op
    MP([SimpleMotionPlannerOp]):::op
    Sink([Command sink]):::hw

    Src ==>|Joint state| SM
    Src ==>|Joint state| MP
    SM ==>|Joint command| MP
    MP ==>|Clamped command| Sink

    classDef op fill:#76B900,stroke:#3F8500,stroke-width:2px,color:#fff
    classDef hw fill:#1DBBA4,stroke:#0D8473,stroke-width:2px,color:#fff
    linkStyle default stroke:#76B900,stroke-width:3px
```

Real mode creates one driver shared by source and sink. Its command edge uses a depth-one,
drop-oldest queue. See the [shared I/O contracts](../../operators/so101/README.md) and
[real graph](real/main.cc).

- **Joint-state source** - Mock mode publishes a synthetic 100 Hz state. Real mode
  reads the Feetech bus at the same period (`state_source.period: 0.01`).
- **SinusMotionOp** - For each of the six joints:

  ```text
  target = default_pos + amplitude * sin(2 * pi * frequency * t)
  ```

  Each incoming joint state triggers command generation.
- **SimpleMotionPlannerOp** - Receives the sine target and the current measured state.
  Each tick, every joint command is clamped to
  `[measured - max_step_size, measured + max_step_size]`. The bound is relative to
  **measured** position, not the previous command.
- **Command sink** - Mock mode validates the clamped command. Real mode writes
  the command to the Feetech servos; see [shutdown](#shutdown) for hold behavior.

## Configuration

Mock and real parameters:
[`config/so101_sinus_motion_config.yaml`](config/so101_sinus_motion_config.yaml).

| Key | Default | Meaning |
| --- | --- | --- |
| `state_source.period` | `0.01` | Joint-state period in seconds (100 Hz). |
| `sinus_motion_op.sin_frequency` | `0.25` | Sine frequency in Hz (one cycle every 4 s). |
| `sinus_motion_op.sin_amplitude` | `0.20` | Amplitude in radians about `default_pos`. |
| `sinus_motion_op.num_joints` | `6` | Joint count (all six SO-101 joints, including gripper). |
| `motion_planner_op.max_step_size` | `0.20` | Maximum per-tick joint delta in radians, vs measured state. |
| `real_robot.serial_port` | `/dev/ttyACM0` | Feetech USB serial device. |
| `real_robot.calibration_file` | placeholder JSON path | LeRobot follower calibration file. |
| `real_robot.position_p_gain` / `_i_gain` / `_d_gain` | `16` / `0` / `32` | Servo PID; **I = 0** matches the LeRobot defaults. |
| `real_robot.run_speed` / `run_acceleration` | `1500` / `60` | Servo-side limits used for graph commands; see the units and special zero values below. |
| `real_robot.home_position` | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | Homing pose, equal to the configured sine mean. |

`real_robot` keys match
[`so101_reach`](../so101_reach/config/so101_reach_config.yaml).

> **Tip:** For jerky motion, check `max_step_size` against amplitude and frequency.
> Very small steps may leave the arm nearly still: Feetech servos often ignore tiny offsets.

The gripper uses a different calibration convention from the other joints:
coordinate `0` is its calibrated closed endpoint. Because the one scalar sine
offset is applied to all six joints, the default gripper command opens during the
positive half-cycle and its negative half-cycle clamps at the closed limit. The
demo therefore does not sweep the gripper's full calibrated range.

### Servo gains and motion profile

<details>
<summary>❓ How do gains and motion limits affect the arm?</summary>

Sinus motion emits absolute, time-based targets, so it does not need the nonzero integral gain
used by the reach application's measured-plus-delta action law. Keeping `position_i_gain` at zero
also avoids integral wind-up causing the servos to hunt around a fixed hold target after shutdown.

Real mode homes the arm to the all-zero sine mean before starting the graph. Homing uses its own
fixed profile (`400` encoder steps/s and acceleration `50`), so changing `real_robot.run_speed`
or `real_robot.run_acceleration` does not change the initial homing move. Gravity-loaded
joints cannot meet the four-count homing gate with a proportional-only loop, so the
driver writes `I = 16` for that move and restores `real_robot.position_i_gain` (default
`0`) before the graph runs. A configured I gain of 16 or higher is left unchanged.

For graph commands, `run_speed` is measured in encoder steps/s (`0` means maximum
speed). A positive `run_acceleration` value is approximately 100 encoder steps/s²
per unit; a higher value permits sharper acceleration, while `0` selects
instant/unlimited acceleration. The default sine needs only about 205 encoder
steps/s and 322 encoder steps/s² at its peaks.
A continuously replaced sine target is not a reliable visual test of those limits.
Use the bounded `profile_real` diagnostic to compare measured responses to
isolated targets, keeping every other setting unchanged.

</details>

## Simulation configuration

Sample YAML: [sim/config/so101_sinus_motion_sim_config.yaml](sim/config/so101_sinus_motion_sim_config.yaml).

- `sim_so101.hostname` / `port` - UCX peer (Sim). Defaults `127.0.0.1:50008`.
- `sim_so101.joint_state_flatbuffer_buffer_size` - receiver `tagRecv` buffer
  for the JointState FBS blob. Increase it if you extend the schema.
- `sim_so101.enable_camera` - Enable the `image_receiver` +
  Holoviz chain. Defaults `true`; set to `false` for headless runs or scenes
  without `UCXCameraHelper` / `UCXPublishImage` OGN nodes.
- `sim_so101.camera_tensor_buffer_size` - device-buffer size the image
  receiver preallocates for the second (tensor) tagSend on the image tag.
  Must be at least `width * height * channels`; default `640 * 480 * 3` = `921_600` (RGB).
- `sim_so101.image_flatbuffer_buffer_size` - receiver `tagRecv` buffer for
  the Image metadata FBS blob.
- `sinus_motion_op.num_joints` - number of joints on the arm (SO-101 = 6).
- `sinus_motion_op.sin_frequency`, `sin_amplitude` - sine frequency in Hz and amplitude in radians.
- `sinus_motion_op.default_position`, `kp`, `kd` - a scalar or single-element list
  for broadcast, or a length-`num_joints` list.

### Wire format

Wire-compatible with Isaac Sim's `isaacsim.ucx.core` publishers/subscribers.
FBS schemas: `isaac.JointState`, `isaac.JointCommand` (and
`isaac.Image` when `enable_camera: true`). FNV-1a tag hashes: see `sim/main.cc`.
Joint envelopes reference the SDK's `holoscan.schema.Tensor`. Boundary codecs
translate their bytes to/from CPU vectors; wire timestamps become graph capture
time. Camera messages use SDK `ImageT` with an explicit CUDA device binding.

### Implementation notes

<details>
<summary>❓ How is the simulation graph scheduled?</summary>

- The graph has `connection`, `joint_state_rx` and `control_and_tx` partitions, plus
  `image_transport` and `holoviz` when `enable_camera: true`.
- UCX connects after graph startup so early receive notifications are accepted.
- Commands go directly to the sink because the sine amplitude is bounded.
  Teleoperation or learned policies should add a motion planner.
- Scheduling uses `RealtimeClock`. Sine phase and graph capture time use the
  received `JointState` timestamp, with no local-clock fallback.

</details>

## Measure servo speed and acceleration

Complete [real-hardware setup](#real-hardware) first. The app build includes this
diagnostic for comparing isolated targets rather than a continuously replaced sine.

```bash
./isaac_os run so101_sinus_motion profile_real \
  --run-args="--config-path=build/so101-sinus-real.yaml --output=build/profile.csv"
```

The output path must be new. The arm first performs the normal homing move, then
shoulder pan moves **+0.20 radians from its measured starting position and back,
three times**. The other five joints retain their measured starting targets.
Each target is written once and held for three seconds while position and velocity
are sampled about every 10 ms. This takes 18 seconds after homing. The diagnostic
bypasses the sine generator and measured-position clamp so neither can continually
replace the servo's target. It uses the same driver and configured run speed,
acceleration, and PID gains as real sinus mode. Sine amplitude, frequency, planner
step size, and state-source period do not affect this fixed diagnostic.

Keep the shoulder-pan path clear, including its configured home position plus
0.20 radians. The driver's calibrated joint limits still apply; a clamped target
invalidates a profile comparison. Normal completion, SIGINT/SIGTERM after homing,
and errors request a measured-position hold with torque enabled.
If feedback or communication prevents the hold, the diagnostic retries up to three
times. If all attempts fail, it exits with a warning that motion may continue.
Keep clear and switch off follower power before handling the arm. The unpowered arm
can fall; torque is not automatically disabled because that can drop the arm.
An initialization failure removes the newly created empty CSV so the same output
path can be retried. Initialization
retains the standard driver's homing and failure behavior described above.

Run separate CSV captures with only the indicated setting changed:

| Comparison | Fixed setting | Values to compare |
| --- | --- | --- |
| Acceleration | `run_speed: 1500` | `run_acceleration: 1` and `0` |
| Speed | `run_acceleration: 0` | `run_speed: 50` and `1500` |

Each CSV row contains the zero-based phase, elapsed seconds since its target write,
target radians, measured shoulder-pan radians, and measured velocity in radians/s.
Compare the time to first reach within 0.02 radians of each target across the six
phases, as well as the position/velocity traces. Confirm every phase reaches and
holds its target; do not interpret an incomplete or interrupted capture as a pass.
The diagnostic fails if a phase ends more than 0.02 radians from its target.
Homing is excluded from the CSV. A low positive acceleration should take longer
than unlimited acceleration, and a low positive speed should take longer than a
high speed. Higher positive acceleration does not mean a smoother ramp.

## Shutdown

<details>
<summary>❓ How does cancellation hold position?</summary>

In real mode, Ctrl+C requests a measured-position hold with torque enabled.
The driver rejects later motion commands after the hold is requested.

</details>
