# Isaac Sim UCX Python entrypoint

This directory contains the Python code mounted by the root `./isaac_sim`
launcher. User-facing launcher documentation lives in
[`docs/how-to/set_up_isaac_sim.mdx`](../../docs/how-to/set_up_isaac_sim.mdx).

## Files

- `isaac_sim_ucx.py`: Isaac Sim Python entrypoint. It loads an optional USD
  stage, starts a UCX listener, waits for a UCX client, plays the simulation
  while the client is connected, then resets the stage and waits for the next
  client.
- `isaac_sim_entrypoint.sh`: container entrypoint used by `./isaac_sim`. It
  handles Isaac Sim EULA/privacy prompts and forwards Ctrl+C cleanly to
  `/isaac-sim/python.sh`.
- `tests/`: lightweight shell and Python tests for the launcher contract and
  argument parsing.

## Python arguments

> **Note:** Don't invoke this script directly. Use the root `./isaac_sim`
> launcher, which starts the Isaac Sim container and mounts this entrypoint for
> you (see [`docs/how-to/set_up_isaac_sim.mdx`](../../docs/how-to/set_up_isaac_sim.mdx)). The
> raw invocation below documents the internal contract and the arguments the
> launcher forwards after `--`.

`isaac_sim_ucx.py` is intended to run inside an Isaac Sim Python environment:

```bash
/isaac-sim/python.sh tools/isaac-sim-ucx/isaac_sim_ucx.py \
  --scene /path/to/scene.usd \
  --port 50008 \
  --sim-dt 0.0166667
```

Common options:

- `--scene PATH`: USD scene file to load before waiting for a UCX client. If
  omitted, Isaac Sim starts with its default stage.
- `--port PORT`: UCX listener port, default `50008`.
- `--sim-dt SECONDS`: physics timestep. If omitted, the scene's
  `physxScene:timeStepsPerSecond` is used (falling back to `1 / 200` when unset),
  so a scene can pin its own physics rate. Pass explicitly to override.
- `--substeps N`: physics steps per loop iteration. Sets `inputs:step` on existing
  `IsaacSimulationGate` nodes named `physics_control_gate`; a control chain driven
  through that gate by `OnPhysicsStep` runs at `physics_rate / N`. Default `1`.
  See "Physics vs. control rate" below.
- `--headless`: start Isaac Sim without a GUI.
- `--require-scriptnode-consent`: don't pre-authorize Python ScriptNodes embedded in
  the USD scene; keep Isaac Sim's consent gate enabled. Use for scenes you don't
  trust. See [`SECURITY.md`](../../SECURITY.md#usd-scriptnodes).
- `--warmup`: perform one physics step and render, then exit without waiting for
  a controller. `./isaac_sim warmup` selects this mode. Normal runs skip warmup.
  Closing Isaac Sim waits for pending cache writes, which can take minutes with
  an empty cache.

The root `./isaac_sim` launcher mounts this file into the Isaac Sim container
and forwards any arguments after `--` to the Python entrypoint.

## Runtime behavior

At startup, `isaac_sim_ucx.py` enables the `isaacsim.ucx.nodes` extension and
sets conservative UCX defaults:

```text
UCX_TLS=tcp,self
UCX_NET_DEVICES=all
```

The scene opens after the runtime APIs initialize so its physics scene is
registered before the timestep is configured. Normal runs load it once.
Once the stage is loaded, the app creates a listener on `--port`. The simulator
is paused until a client connects. After disconnect, the stage is stopped and
reset so a later client gets a clean run.

## Physics vs. control rate

The launcher sets `inputs:step` to `--substeps` on existing nodes named
`physics_control_gate` whose type is `isaacsim.core.nodes.IsaacSimulationGate`.
The attribute must already exist. It does not create nodes or change connections.
In the G1 scene, `OnPhysicsStep` drives this gate, so its downstream control chain
runs at `physics_rate / substeps`:

- **Physics** runs at the scene's `physxScene:timeStepsPerSecond` (or `--sim-dt`;
  default 200 Hz) - the rate the USD position-drive PD tracks joint targets.
- **Control** (the gated path that publishes joint state / IMU / clock and applies
  the received `JointCommand`) runs at `physics_rate / substeps`. Between control
  steps the PhysX drive holds the last commanded target (zero-order hold).

`--substeps` defaults to `1` (control == physics). **The WBC apps run the sim with
`--substeps 4`**, giving 50 Hz control at 200 Hz physics - the policy's trained rate.
`simulation_manager.step(steps=substeps)` advances physics and control.
`rendering_manager.render()` updates the camera without republishing control
messages. The SO-101 scene uses `OnPlaybackTick` without this gate; the same ratio
does not establish its control rate.

The startup log calculates rates from the physics timestep and `--substeps`:

```text
Physics 200.0 Hz, control 50.0 Hz (4 physics substeps/control step).
```

This log does not verify that the gate exists or that the control chain uses it.
Configuration steps are in
[Physics and control rate](../../docs/how-to/set_up_isaac_sim.mdx#physics-and-control-rate).

Isaac Sim up to 6.2.1.0 does not apply `kp`/`kd` received through UCX; the supplied scenes use
USD articulation-drive gains. See [joint-command gains](../../docs/how-to/set_up_isaac_sim.mdx#configure-joint-gains)
for details.

Scene-specific assets and commands are documented by the applications that use
them. The root launcher docs live in
[`docs/how-to/set_up_isaac_sim.mdx`](../../docs/how-to/set_up_isaac_sim.mdx).

> The Isaac Sim UCX listener is intended for trusted local or isolated simulation
> networks; see [`SECURITY.md`](../../SECURITY.md) for UCX deployment guidance.
