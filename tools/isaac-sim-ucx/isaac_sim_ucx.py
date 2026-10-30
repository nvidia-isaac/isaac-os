# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Sim UCX.

Loads a USD scene into Isaac Sim, then waits for an incoming UCX connection
before starting the simulation.  When the client disconnects the stage is reset
and the listener waits for the next connection.

Usage (run via isaac-sim's python.sh):
    python.sh isaac_sim_ucx.py \\
        [--scene /path/to/scene.usd] [--port 50008] [--sim-dt 0.0167] [--headless] \\
        [--require-scriptnode-consent]
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from typing import Protocol

# -- logging -----------------------------------------------------------------

# Configure logging early with force=True so our format wins over whatever
# SimulationApp sets up internally.
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(name)s: %(message)s',
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# -- globals -----------------------------------------------------------------

_POLL_TIMEOUT_MS = 100  # how long wait_for_connection blocks before yielding to the render loop
_UCX_ENV_DEFAULTS = {
    'UCX_TLS': 'tcp,self',
    'UCX_NET_DEVICES': 'all',
}
_STARTUP_EXTENSIONS = ('isaacsim.ucx.nodes', 'omni.graph.scriptnode')

# Apply ScriptNode consent settings before opening the stage.
#   opt_in=true        -> consent granted for ScriptNodes to execute embedded Python
#   enable_opt_in=false -> disable the gate entirely so no consent dialog appears
_SCRIPTNODE_AUTO_CONSENT_SETTINGS = {
    '/app/omni.graph.scriptnode/opt_in': 'true',
    '/app/omni.graph.scriptnode/enable_opt_in': 'false',
}


class SimulationAppRuntime(Protocol):
    def update(self) -> None: ...
    def is_running(self) -> bool: ...
    def close(self) -> None: ...


class AppUtilsRuntime(Protocol):
    def play(self) -> None: ...
    def pause(self) -> None: ...
    def stop(self) -> None: ...


class RenderingManagerRuntime(Protocol):
    def render(self) -> None: ...


class SimulationManagerRuntime(Protocol):
    def step(self, *, steps: int = 1) -> None: ...
    def set_physics_dt(self, physics_dt: float) -> None: ...
    def get_physics_dt(self) -> float: ...


class UcxListenerRuntime(Protocol):
    def get_port(self) -> int: ...
    def is_connected(self) -> bool: ...
    def wait_for_connection(self, timeout_ms: int) -> bool: ...
    def stop_progress_thread(self) -> None: ...
    def shutdown(self) -> None: ...


@dataclass(frozen=True)
class IsaacSimRuntime:
    simulation_app: SimulationAppRuntime
    app_utils: AppUtilsRuntime
    rendering_manager: RenderingManagerRuntime
    simulation_manager: SimulationManagerRuntime
    add_listener: Callable[[int], UcxListenerRuntime]
    remove_listener: Callable[[int], None]
    is_stage_loading: Callable[[], bool]
    open_stage: Callable[[str], bool]


# -- arguments ---------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments without importing Isaac Sim."""
    parser = argparse.ArgumentParser(description='Run Isaac Sim with a UCX gate.')
    parser.add_argument(
        '-s',
        '--scene',
        default=None,
        help='Path to the USD scene file to load.',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=50008,
        help='UCX listener port (default: 50008).',
    )
    parser.add_argument(
        '--sim-dt',
        type=float,
        default=None,
        help=(
            'Physics timestep in seconds. If omitted, use the scene PhysicsScene '
            'physxScene:timeStepsPerSecond (falling back to 1/200 when unset). Pass '
            'explicitly to override the scene.'
        ),
    )
    parser.add_argument(
        '--substeps',
        type=int,
        default=1,
        help=(
            'Physics substeps per control step. The ActionGraph (state publish / '
            'command apply) ticks once per control step, so control runs at '
            'physics_rate / substeps. Default 1 (control == physics). The WBC apps '
            'pass --substeps 4 to get 50 Hz control at 200 Hz physics.'
        ),
    )
    parser.add_argument('--headless', action='store_true', help='Run without a GUI window.')
    parser.add_argument(
        '--require-scriptnode-consent',
        action='store_true',
        help=(
            'Do not pre-authorize Python ScriptNodes embedded in USD scenes; '
            'leave the Isaac Sim consent gate enabled.'
        ),
    )
    parser.add_argument(
        '--warmup',
        action='store_true',
        help='Populate caches with one physics step and render, then exit.',
    )
    args, _ = parser.parse_known_args(argv)
    if args.sim_dt is not None and args.sim_dt <= 0.0:
        parser.error('--sim-dt must be > 0')
    if args.substeps < 1:
        parser.error('--substeps must be >= 1')
    return args


# -- paths -------------------------------------------------------------------


def _set_ucx_env_defaults() -> None:
    """Set UCX defaults before Isaac Sim imports and loads UCX extensions."""
    for name, value in _UCX_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


def _resolve_scene_path(path: str) -> str:
    """Resolve and validate a USD scene path."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f'Scene not found: {p}')
    return str(p)


# -- isaac_sim ---------------------------------------------------------------


def _start_isaac_sim(
    headless: bool,
    auto_consent_scriptnodes: bool = True,
) -> IsaacSimRuntime:
    """Import Isaac Sim modules and create the SimulationApp."""
    extra_args = []
    for extension in _STARTUP_EXTENSIONS:
        extra_args.extend(['--enable', extension])
    if auto_consent_scriptnodes:
        for key, value in _SCRIPTNODE_AUTO_CONSENT_SETTINGS.items():
            extra_args.append(f'--{key}={value}')

    launch_config = {'headless': headless, 'extra_args': extra_args}
    simulation_app = _create_simulation_app(launch_config)

    # Keep Isaac Sim imports here so parse_args stays importable on a host
    # without Isaac Sim installed.
    # pylint: disable=import-outside-toplevel

    import isaacsim.core.experimental.utils.app as isaac_app_utils  # noqa: E402
    from isaacsim.core.rendering_manager import (
        RenderingManager as IsaacRenderingManager,  # noqa: E402
    )
    from isaacsim.core.simulation_manager import (
        SimulationManager as IsaacSimulationManager,  # noqa: E402
    )
    from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
    from isaacsim.core.utils.stage import is_stage_loading as isaac_is_stage_loading  # noqa: E402
    from isaacsim.core.utils.stage import open_stage as isaac_open_stage  # noqa: E402

    if not headless:
        enable_extension('omni.graph.window.action')

    from isaacsim.ucx.core import add_listener as isaac_add_listener  # noqa: E402
    from isaacsim.ucx.core import remove_listener as isaac_remove_listener  # noqa: E402

    return IsaacSimRuntime(
        simulation_app=simulation_app,
        app_utils=isaac_app_utils,
        rendering_manager=IsaacRenderingManager,
        simulation_manager=IsaacSimulationManager,
        add_listener=isaac_add_listener,
        remove_listener=isaac_remove_listener,
        is_stage_loading=isaac_is_stage_loading,
        open_stage=isaac_open_stage,
    )


def _create_simulation_app(launch_config: dict) -> SimulationAppRuntime:
    """Create SimulationApp without leaking this script's arguments to Kit."""
    # pylint: disable=import-outside-toplevel
    from isaacsim import SimulationApp  # noqa: E402

    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        return SimulationApp(launch_config)
    finally:
        sys.argv = original_argv


# -- stage -------------------------------------------------------------------


def _wait_for_stage_loaded(runtime: IsaacSimRuntime, scene_path: str) -> None:
    """Wait for the current USD stage to finish loading."""
    # pylint: disable=import-outside-toplevel
    import omni.usd

    while runtime.is_stage_loading():
        runtime.simulation_app.update()
    if omni.usd.get_context().get_stage() is None:
        logger.error('The scene could not be loaded: %s', scene_path)
        sys.exit(1)
    logger.info('Stage loaded.')


def _load_stage(runtime: IsaacSimRuntime, scene_path: str) -> None:
    """Open the USD scene and wait for it to finish loading."""
    logger.info('Opening scene: %s', scene_path)
    success = runtime.open_stage(scene_path)
    if not success:
        logger.error('The scene could not be opened: %s', scene_path)
        sys.exit(1)
    _wait_for_stage_loaded(runtime, scene_path)


_DEFAULT_PHYSICS_DT = 1.0 / 200.0


def _resolve_physics_dt(sim_dt: float | None) -> float:
    """Return sim_dt, or derive it from the scene's declared physics rate.

    An explicit --sim-dt always wins. Otherwise read physxScene:timeStepsPerSecond
    off the current stage's PhysicsScene so a scene can pin its own physics rate
    (H2 and G1 declare 200 Hz). Falls back to 1/200 when unset.
    """
    if sim_dt is not None:
        return sim_dt
    # pylint: disable=import-outside-toplevel
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is not None:
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Scene):
                continue
            attr = prim.GetAttribute('physxScene:timeStepsPerSecond')
            steps = attr.Get() if attr and attr.IsValid() else None
            if steps:
                dt = 1.0 / float(steps)
                logger.info('Using scene physics rate: %g Hz (dt %.5f s)', float(steps), dt)
                return dt
    logger.info('No scene physics rate; using default %g Hz.', 1.0 / _DEFAULT_PHYSICS_DT)
    return _DEFAULT_PHYSICS_DT


def _set_ucx_port(port: int) -> None:
    """Set inputs:port on every isaacsim.ucx.nodes.* prim to ``port``.

    The USDA leaves inputs:port unset on all UCX publish/subscribe nodes; we
    write it here so every node matches the listener opened by add_listener().
    Without this, nodes either use stale hardcoded ports or fail to bind.
    """
    # pylint: disable=import-outside-toplevel
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    for prim in stage.Traverse():
        type_name = prim.GetAttribute('node:type').Get()
        if not type_name or not type_name.startswith('isaacsim.ucx.nodes.'):
            continue
        attr = prim.GetAttribute('inputs:port')
        if attr.IsValid():
            attr.Set(port)


def _set_control_substeps(substeps: int) -> None:
    """Set control-rate decimation."""
    # pylint: disable=import-outside-toplevel
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    for prim in stage.Traverse():
        if prim.GetName() != 'physics_control_gate':
            continue
        type_attr = prim.GetAttribute('node:type')
        if not type_attr.IsValid() or type_attr.Get() != 'isaacsim.core.nodes.IsaacSimulationGate':
            continue
        step_attr = prim.GetAttribute('inputs:step')
        if step_attr.IsValid():
            step_attr.Set(substeps)


def _configure_stage(port: int, substeps: int) -> None:
    """Set the UCX port and control-rate decimation."""
    _set_ucx_port(port)
    _set_control_substeps(substeps)


def _configure_physics(runtime: IsaacSimRuntime, physics_dt: float, substeps: int) -> None:
    """Apply the resolved timestep to the current stage and report the active rate."""
    runtime.simulation_manager.set_physics_dt(physics_dt)
    active_dt = runtime.simulation_manager.get_physics_dt()
    logger.info(
        'Physics %.1f Hz, control %.1f Hz (%d physics substeps/control step).',
        1.0 / active_dt,
        1.0 / (active_dt * substeps),
        substeps,
    )


def _set_action_graphs_enabled(enabled: bool) -> None:
    """Enable or disable every ActionGraph on the current stage.

    Keeping the graph disabled until a UCX client connects prevents the
    articulation controller from yanking joints to a default-zero
    positionCommand on play() — otherwise the robot launches forward.
    """
    # pylint: disable=import-outside-toplevel
    try:
        import omni.graph.core as og  # type: ignore
        import omni.usd
    except ImportError as error:
        logger.warning(
            'Graph controls are unavailable; simulation actions cannot be toggled. Reason: %s',
            error,
        )
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.endswith('/ActionGraph'):
            continue
        graph = og.get_graph_by_path(path)
        if graph is None or not graph.is_valid():
            continue
        graph.set_disabled(not enabled)


def _on_connect(runtime: IsaacSimRuntime) -> None:
    """Enable the robot ActionGraph and start the timeline."""
    logger.info('UCX client connected - starting simulation.')
    _set_action_graphs_enabled(True)
    runtime.app_utils.play()


def _reset_stage(runtime: IsaacSimRuntime, scene_path: str) -> None:
    """Stop the timeline and reload the stage to its initial state."""
    logger.info('Resetting stage...')
    runtime.app_utils.stop()
    # Send a few frames so in-flight graph evaluations drain before we reload.
    for _ in range(3):
        runtime.simulation_app.update()
    _load_stage(runtime, scene_path)


def _initialize_idle_stage(runtime: IsaacSimRuntime) -> None:
    """Initialize and pause the stage."""
    runtime.app_utils.play()
    runtime.simulation_manager.step()
    runtime.rendering_manager.render()
    runtime.app_utils.pause()
    runtime.rendering_manager.render()
    logger.info('The simulator is initialized and waiting for a controller connection.')


# -- ucx_loop ----------------------------------------------------------------


def run(  # pylint: disable=too-many-positional-arguments
    runtime: IsaacSimRuntime,
    scene_path: str | None,
    port: int,
    sim_dt: float | None,
    warmup: bool = False,
    substeps: int = 1,
) -> None:
    """Run Isaac Sim with a UCX listener or one-shot cache warmup."""
    substeps = max(1, substeps)
    if scene_path is not None:
        _load_stage(runtime, scene_path)
        _configure_stage(port, substeps)
    else:
        logger.info('No stage path provided; using Isaac Sim default stage.')
    physics_dt = _resolve_physics_dt(sim_dt)
    _configure_physics(runtime, physics_dt, substeps)

    # Disable any /<robot>/ActionGraph until a UCX client connects.  Otherwise
    # the articulation controller ticks on default-zero positionCommand and
    # yanks every joint to zero on play() (= robot launches forward).
    _set_action_graphs_enabled(False)

    if warmup:
        logger.info('Warming Isaac Sim caches with one physics step and render.')
        runtime.app_utils.play()
        runtime.simulation_manager.step()
        runtime.rendering_manager.render()
        logger.info('Warmup complete; closing Isaac Sim to finish cache writes.')
        return

    if scene_path is not None:
        _initialize_idle_stage(runtime)

    listener = runtime.add_listener(port)
    logger.info('Listening for UCX connections on port %d', listener.get_port())
    logger.info('Waiting for UCX client to connect...')

    was_connected = False

    try:
        while runtime.simulation_app.is_running():
            connected = listener.is_connected()

            if not connected:
                if was_connected:
                    # Transition: connected to disconnected; reset the sim.
                    logger.info('UCX client disconnected - resetting simulation.')
                    if scene_path is not None:
                        _reset_stage(runtime, scene_path)
                        _configure_stage(port, substeps)
                        _configure_physics(runtime, physics_dt, substeps)
                    _set_action_graphs_enabled(False)
                    if scene_path is not None:
                        _initialize_idle_stage(runtime)
                    was_connected = False
                    logger.info('Waiting for UCX client to connect...')

                # Idle: block briefly so the render loop stays alive.
                connected = listener.wait_for_connection(timeout_ms=_POLL_TIMEOUT_MS)
                runtime.simulation_app.update()
                if connected:
                    was_connected = True
                    _on_connect(runtime)
            else:
                if not was_connected:
                    was_connected = True
                    _on_connect(runtime)
                # Advance physics and render the camera.
                try:
                    runtime.simulation_manager.step(steps=substeps)
                    runtime.rendering_manager.render()
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning('Simulation step error: %s - resetting.', exc)
                    if scene_path is not None:
                        _reset_stage(runtime, scene_path)
                        _configure_stage(port, substeps)
                        _configure_physics(runtime, physics_dt, substeps)
                        _set_action_graphs_enabled(False)
                        _initialize_idle_stage(runtime)
                    was_connected = False
                    logger.info('Waiting for UCX client to connect...')

    finally:
        listener.stop_progress_thread()
        listener.shutdown()
        runtime.remove_listener(port)


# -- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Start Isaac Sim and run the requested mode."""
    args = parse_args(argv)
    _set_ucx_env_defaults()
    scene_path = _resolve_scene_path(args.scene) if args.scene is not None else None
    runtime = _start_isaac_sim(
        args.headless,
        auto_consent_scriptnodes=not args.require_scriptnode_consent,
    )
    try:
        run(
            runtime,
            scene_path,
            args.port,
            args.sim_dt,
            args.warmup,
            args.substeps,
        )
    finally:
        runtime.app_utils.stop()
        runtime.simulation_app.close()


if __name__ == '__main__':
    main()
