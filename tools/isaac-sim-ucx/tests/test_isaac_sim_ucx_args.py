#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parser tests for the UCX entrypoint."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest import TestCase
from unittest.mock import patch

TOOL_DIR = Path(__file__).resolve().parents[1]
MODULE_NAME = 'isaac_sim_ucx'
MODULE_PATH = TOOL_DIR / f'{MODULE_NAME}.py'


def load_module():
    spec = importlib.util.spec_from_file_location('isaac_sim_ucx', MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load {MODULE_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warmup_arguments_are_parsed():
    module = load_module()

    args = module.parse_args(
        [
            '--scene',
            '/isaac-sim-ucx/assets/scene.usd',
            '--headless',
            '--warmup',
        ]
    )

    assert args.scene == '/isaac-sim-ucx/assets/scene.usd'
    assert args.headless is True
    assert args.warmup is True


def test_default_arguments_are_stable():
    module = load_module()

    args = module.parse_args([])

    assert args.scene is None
    assert args.port == 50008
    assert args.sim_dt is None
    assert args.headless is False
    assert args.warmup is False
    assert args.substeps == 1
    assert args.require_scriptnode_consent is False


def test_require_scriptnode_consent_argument_is_parsed():
    module = load_module()

    args = module.parse_args(['--require-scriptnode-consent'])

    assert args.require_scriptnode_consent is True


def _capture_startup_launch_config(module, auto_consent_scriptnodes):
    """Return the launch config before Isaac Sim imports are attempted."""
    captured = []

    class LaunchConfigCaptured(Exception):
        """Stop startup after capturing the SimulationApp configuration."""

    def capture_launch_config(launch_config):
        captured.append(launch_config)
        raise LaunchConfigCaptured

    module._create_simulation_app = capture_launch_config  # pylint: disable=protected-access
    try:
        module._start_isaac_sim(  # pylint: disable=protected-access
            True,
            auto_consent_scriptnodes=auto_consent_scriptnodes,
        )
    except LaunchConfigCaptured:
        pass

    assert len(captured) == 1
    return captured[0]


def test_scriptnode_auto_consent_settings_are_enabled_by_default():
    module = load_module()

    launch_config = _capture_startup_launch_config(module, auto_consent_scriptnodes=True)

    assert '--/app/omni.graph.scriptnode/opt_in=true' in launch_config['extra_args']
    assert '--/app/omni.graph.scriptnode/enable_opt_in=false' in launch_config['extra_args']


def test_require_scriptnode_consent_omits_consent_overrides():
    module = load_module()

    launch_config = _capture_startup_launch_config(module, auto_consent_scriptnodes=False)

    assert 'omni.graph.scriptnode' in launch_config['extra_args']
    assert not any(
        argument.startswith('--/app/omni.graph.scriptnode/')
        for argument in launch_config['extra_args']
    )


def test_scene_short_option_is_parsed():
    module = load_module()

    args = module.parse_args(['-s', '/isaac-sim-ucx/assets/scene.usd'])

    assert args.scene == '/isaac-sim-ucx/assets/scene.usd'


def test_sim_dt_explicit_is_parsed():
    module = load_module()

    args = module.parse_args(['--sim-dt', '0.02'])

    assert abs(args.sim_dt - 0.02) < 1e-9


def test_non_positive_sim_dt_is_rejected():
    module = load_module()

    for value in ('0', '-0.01'):
        try:
            module.parse_args(['--sim-dt', value])
        except SystemExit:
            continue
        raise AssertionError(f'--sim-dt {value} should be rejected')


def test_non_positive_substeps_is_rejected():
    module = load_module()

    for value in ('0', '-1'):
        try:
            module.parse_args(['--substeps', value])
        except SystemExit:
            continue
        raise AssertionError(f'--substeps {value} should be rejected')


def test_resolve_physics_dt_prefers_explicit_value():
    module = load_module()

    # An explicit dt is returned verbatim without touching the (absent) stage.
    assert module._resolve_physics_dt(0.02) == 0.02  # pylint: disable=protected-access


def test_idle_stage_plays_renders_and_pauses_before_accepting_a_client():
    module = load_module()
    events = []
    runtime = types.SimpleNamespace(
        app_utils=types.SimpleNamespace(
            play=lambda: events.append('play'),
            pause=lambda: events.append('pause'),
        ),
        simulation_manager=types.SimpleNamespace(step=lambda: events.append('step')),
        rendering_manager=types.SimpleNamespace(render=lambda: events.append('render')),
    )

    module._initialize_idle_stage(runtime)  # pylint: disable=protected-access

    assert events == ['play', 'step', 'render', 'pause', 'render']


def test_warmup_can_run_without_stage():
    module = load_module()
    events = []

    class FakeAppUtils:
        @staticmethod
        def play():
            events.append('play')

        @staticmethod
        def stop():
            pass

    class FakeSimulationManager:
        """Simulation manager without Isaac Sim dependencies."""

        @staticmethod
        def set_physics_dt(_sim_dt):
            pass

        @staticmethod
        def get_physics_dt():
            return 1.0 / 60.0

        @staticmethod
        def step(*, steps=1):
            _ = steps
            events.append('step')

    class FakeRenderingManager:
        @staticmethod
        def render():
            events.append('render')

    class FakeSimulationApp:
        @staticmethod
        def update():
            raise AssertionError('warmup must not request a redundant Kit update')

    def add_listener(_port):
        raise AssertionError('warmup must not wait for UCX')

    runtime = module.IsaacSimRuntime(
        simulation_app=FakeSimulationApp(),
        app_utils=FakeAppUtils(),
        rendering_manager=FakeRenderingManager,
        simulation_manager=FakeSimulationManager,
        add_listener=add_listener,
        remove_listener=lambda _port: None,
        is_stage_loading=lambda: False,
        open_stage=lambda _path: True,
    )

    module.run(runtime, None, 50008, 1.0 / 60.0, warmup=True)

    assert events == ['play', 'step', 'render']


def test_connected_listener_starts_simulation_before_stepping():
    module = load_module()
    events = []

    class FakeAppUtils:
        @staticmethod
        def play():
            events.append('play')

    class FakeSimulationManager:
        """Simulation manager without Isaac Sim dependencies."""

        @staticmethod
        def set_physics_dt(_sim_dt):
            pass

        @staticmethod
        def get_physics_dt():
            return 1.0 / 60.0

        @staticmethod
        def step(*, steps=1):
            _ = steps
            events.append('step')

    class FakeRenderingManager:
        @staticmethod
        def render():
            events.append('render')

    class FakeSimulationApp:
        """Simulation app that runs exactly one loop iteration."""

        calls = 0

        @classmethod
        def is_running(cls):
            cls.calls += 1
            return cls.calls == 1

        @staticmethod
        def update():
            events.append('update')

    class FakeListener:
        """Listener that is already connected when first polled."""

        @staticmethod
        def get_port():
            return 50008

        @staticmethod
        def is_connected():
            return True

        @staticmethod
        def wait_for_connection(timeout_ms):  # noqa: ARG004
            raise AssertionError('already-connected listener should not wait')

        @staticmethod
        def stop_progress_thread():
            pass

        @staticmethod
        def shutdown():
            pass

    runtime = module.IsaacSimRuntime(
        simulation_app=FakeSimulationApp(),
        app_utils=FakeAppUtils(),
        rendering_manager=FakeRenderingManager,
        simulation_manager=FakeSimulationManager,
        add_listener=lambda _port: FakeListener(),
        remove_listener=lambda _port: None,
        is_stage_loading=lambda: False,
        open_stage=lambda _path: True,
    )

    module.run(runtime, None, 50008, 1.0 / 60.0)

    assert events[:3] == ['play', 'step', 'render']


def _run_capturing_step_count(module, sim_dt, substeps):
    """Run one active-loop iteration and return the ``steps`` passed to step()."""
    captured = {}

    class FakeAppUtils:
        @staticmethod
        def play():
            pass

    class FakeSimulationManager:
        """Simulation manager without Isaac Sim dependencies."""

        @staticmethod
        def set_physics_dt(_sim_dt):
            pass

        @staticmethod
        def get_physics_dt():
            return 1.0 / 200.0

        @staticmethod
        def step(*, steps=1):
            captured['steps'] = steps

    class FakeRenderingManager:
        @staticmethod
        def render():
            pass

    class FakeSimulationApp:
        """Runs exactly one active-loop iteration."""

        calls = 0

        @classmethod
        def is_running(cls):
            cls.calls += 1
            return cls.calls == 1

        @staticmethod
        def update():
            pass

    class FakeListener:
        """Listener already connected when first polled."""

        @staticmethod
        def get_port():
            return 50008

        @staticmethod
        def is_connected():
            return True

        @staticmethod
        def wait_for_connection(timeout_ms):  # noqa: ARG004
            raise AssertionError('already-connected listener should not wait')

        @staticmethod
        def stop_progress_thread():
            pass

        @staticmethod
        def shutdown():
            pass

    runtime = module.IsaacSimRuntime(
        simulation_app=FakeSimulationApp(),
        app_utils=FakeAppUtils(),
        rendering_manager=FakeRenderingManager,
        simulation_manager=FakeSimulationManager,
        add_listener=lambda _port: FakeListener(),
        remove_listener=lambda _port: None,
        is_stage_loading=lambda: False,
        open_stage=lambda _path: True,
    )
    module.run(runtime, None, 50008, sim_dt, substeps=substeps)
    return captured['steps']


def test_substeps_argument_controls_physics_step_count():
    module = load_module()

    # Default 1: control runs at the physics rate.
    assert _run_capturing_step_count(module, 1.0 / 200.0, 1) == 1
    # Explicit --substeps sets the physics substeps per control step (4 -> 50 Hz at 200 Hz).
    assert _run_capturing_step_count(module, 1.0 / 200.0, 4) == 4
    assert _run_capturing_step_count(module, 1.0 / 200.0, 3) == 3


def test_physics_rate_log_uses_active_timestep():
    module = load_module()
    runtime = types.SimpleNamespace(
        simulation_manager=types.SimpleNamespace(
            set_physics_dt=lambda _dt: None,
            get_physics_dt=lambda: 0.02,
        )
    )
    with TestCase().assertLogs(module.logger) as logs:
        module._configure_physics(runtime, 0.01, 4)  # pylint: disable=protected-access
    assert 'Physics 50.0 Hz, control 12.5 Hz' in logs.output[0]


def test_physics_timestep_survives_stage_reloads():
    for physics_dt, substeps in ((0.01, 1), (0.005, 4)):
        for step_error in (False, True):
            _check_timestep_after_reloads(physics_dt, substeps, step_error)


def _check_timestep_after_reloads(physics_dt, substeps, step_error):
    """Model stage reloads restoring 60 Hz during controller sessions."""
    module = load_module()
    state = types.SimpleNamespace(dt=1.0 / 60.0, tick=0, reloads=0, failed=False, registered=False)
    playback_timesteps = []
    connections = iter((True, False, True))

    def set_physics_dt(value):
        if state.registered:
            state.dt = value

    def reload_stage(_path):
        state.registered = True
        state.dt = 1.0 / 60.0
        state.reloads += 1
        return True

    def is_running():
        state.tick += 1
        return state.tick <= 3

    def step(*, steps=1):
        _ = steps
        if state.tick == 1 and step_error and not state.failed:
            state.failed = True
            raise RuntimeError('Physics step failed')
        assert state.dt == physics_dt

    listener = types.SimpleNamespace(
        get_port=lambda: 50008,
        is_connected=lambda: next(connections),
        wait_for_connection=lambda **_kwargs: False,
        stop_progress_thread=lambda: None,
        shutdown=lambda: None,
    )
    runtime = module.IsaacSimRuntime(
        simulation_app=types.SimpleNamespace(update=lambda: None, is_running=is_running),
        app_utils=types.SimpleNamespace(
            play=lambda: playback_timesteps.append(state.dt),
            pause=lambda: None,
            stop=lambda: None,
        ),
        simulation_manager=types.SimpleNamespace(
            set_physics_dt=set_physics_dt,
            get_physics_dt=lambda: state.dt,
            step=step,
        ),
        rendering_manager=types.SimpleNamespace(render=lambda: None),
        add_listener=lambda _port: listener,
        remove_listener=lambda _port: None,
        is_stage_loading=lambda: False,
        open_stage=reload_stage,
    )
    with (
        patch.object(module, '_wait_for_stage_loaded'),
        patch.object(module, '_set_ucx_port'),
        patch.object(module, '_set_control_substeps'),
        patch.object(module, '_set_action_graphs_enabled'),
    ):
        module.run(
            runtime,
            '/scene.usda',
            50008,
            physics_dt,
            substeps=substeps,
        )

    assert state.reloads == 2
    assert playback_timesteps
    assert all(dt == physics_dt for dt in playback_timesteps)


def _get_omnigraph_node_block(usda, node_name):
    """Return one OmniGraph node definition from the textual USDA."""
    start_token = f'        def OmniGraphNode "{node_name}"'
    start = usda.index(start_token)
    end = usda.find('\n        def OmniGraphNode "', start + len(start_token))
    return usda[start:] if end == -1 else usda[start:end]


def test_g1_control_graph_is_physics_driven_and_camera_remains_playback_driven():
    usda = (
        TOOL_DIR.parent.parent / 'operators' / 'unitree_g1' / 'assets' / 'g1_ucx.usda'
    ).read_text(encoding='utf-8')

    assert 'token node:type = "isaacsim.core.nodes.OnPhysicsStep"' in usda
    assert 'token node:type = "isaacsim.core.nodes.IsaacSimulationGate"' in usda
    assert 'token pipelineStage = "pipelineStageOnDemand"' in usda
    assert (
        'prepend uint inputs:execIn.connect = '
        '</g1/ActionGraph/on_physics_step.outputs:step>' in usda
    )
    playback_connection = (
        'prepend uint inputs:execIn.connect = </g1/ActionGraph/on_playback_tick.outputs:tick>'
    )
    control_connection = (
        'prepend uint inputs:execIn.connect = '
        '</g1/ActionGraph/physics_control_gate.outputs:execOut>'
    )
    assert playback_connection in _get_omnigraph_node_block(usda, 'isaac_create_render_product')
    assert control_connection in _get_omnigraph_node_block(usda, 'isaac_read_joint_state_node')
    assert control_connection in _get_omnigraph_node_block(usda, 'isaac_read_imu_node')
    assert usda.count(playback_connection) == 1
    assert usda.count(control_connection) == 2


def test_control_substeps_are_authored_on_named_physics_gate():
    module = load_module()

    class FakeAttribute:
        """Minimal USD attribute test double."""

        # pylint: disable=invalid-name

        def __init__(self, value=None):
            self.value = value

        @staticmethod
        def IsValid():
            return True

        def Get(self):
            return self.value

        def Set(self, value):
            self.value = value

    class FakePrim:
        """Minimal USD prim test double."""

        # pylint: disable=invalid-name

        def __init__(self, name, node_type, step):
            self.name = name
            self.attributes = {
                'node:type': FakeAttribute(node_type),
                'inputs:step': FakeAttribute(step),
            }

        def GetName(self):
            return self.name

        def GetAttribute(self, name):
            return self.attributes[name]

    control_gate = FakePrim(
        'physics_control_gate',
        'isaacsim.core.nodes.IsaacSimulationGate',
        1,
    )
    unrelated_gate = FakePrim(
        'camera_gate',
        'isaacsim.core.nodes.IsaacSimulationGate',
        2,
    )
    stage = types.SimpleNamespace(Traverse=lambda: [control_gate, unrelated_gate])
    context = types.SimpleNamespace(get_stage=lambda: stage)
    fake_usd = types.ModuleType('omni.usd')
    fake_usd.get_context = lambda: context
    fake_omni = types.ModuleType('omni')
    fake_omni.__path__ = []
    fake_omni.usd = fake_usd
    original_omni = sys.modules.get('omni')
    original_omni_usd = sys.modules.get('omni.usd')

    try:
        sys.modules['omni'] = fake_omni
        sys.modules['omni.usd'] = fake_usd
        module._set_control_substeps(4)  # pylint: disable=protected-access
    finally:
        if original_omni is None:
            sys.modules.pop('omni', None)
        else:
            sys.modules['omni'] = original_omni
        if original_omni_usd is None:
            sys.modules.pop('omni.usd', None)
        else:
            sys.modules['omni.usd'] = original_omni_usd

    assert control_gate.GetAttribute('inputs:step').Get() == 4
    assert unrelated_gate.GetAttribute('inputs:step').Get() == 2


def test_main_sets_ucx_defaults_before_starting_isaac_sim():
    module = load_module()
    seen_env = []

    class FakeAppUtils:
        @staticmethod
        def stop():
            pass

    class FakeSimulationApp:
        @staticmethod
        def close():
            pass

    original_env = {key: os.environ.get(key) for key in ('UCX_TLS', 'UCX_NET_DEVICES')}

    def fake_start_isaac_sim(_headless, *, auto_consent_scriptnodes=True):
        _ = auto_consent_scriptnodes
        seen_env.append((os.environ.get('UCX_TLS'), os.environ.get('UCX_NET_DEVICES')))
        return module.IsaacSimRuntime(
            simulation_app=FakeSimulationApp(),
            app_utils=FakeAppUtils(),
            rendering_manager=object(),
            simulation_manager=object(),
            add_listener=lambda _port: object(),
            remove_listener=lambda _port: None,
            is_stage_loading=lambda: False,
            open_stage=lambda _path: True,
        )

    try:
        os.environ.pop('UCX_TLS', None)
        os.environ.pop('UCX_NET_DEVICES', None)
        setattr(module, '_start_isaac_sim', fake_start_isaac_sim)
        setattr(module, 'run', lambda *_args: None)

        module.main([])

        assert seen_env == [('tcp,self', 'all')]
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_main_passes_stage_to_runtime_after_isaac_startup():
    module = load_module()
    seen_startup = []
    seen_run = []

    class FakeAppUtils:
        @staticmethod
        def stop():
            pass

    class FakeSimulationApp:
        @staticmethod
        def close():
            pass

    def fake_start_isaac_sim(headless, *, auto_consent_scriptnodes=True):
        _ = auto_consent_scriptnodes
        seen_startup.append(headless)
        return module.IsaacSimRuntime(
            simulation_app=FakeSimulationApp(),
            app_utils=FakeAppUtils(),
            rendering_manager=object(),
            simulation_manager=object(),
            add_listener=lambda _port: object(),
            remove_listener=lambda _port: None,
            is_stage_loading=lambda: False,
            open_stage=lambda _path: True,
        )

    def fake_run(
        _runtime,
        scene_path,
        _port,
        _sim_dt,
        _warmup,
        _substeps=None,
    ):
        seen_run.append(scene_path)

    with tempfile.NamedTemporaryFile(suffix='.usda') as scene_file:
        scene_path = str(Path(scene_file.name).resolve())
        setattr(module, '_start_isaac_sim', fake_start_isaac_sim)
        setattr(module, 'run', fake_run)

        module.main(['--scene', scene_path, '--headless'])

    assert seen_startup == [True]
    assert seen_run == [scene_path]


def test_simulation_app_startup_does_not_receive_ucx_script_args():
    module = load_module()
    seen_argv = []
    original_argv = sys.argv
    original_isaacsim = sys.modules.get('isaacsim')

    class FakeSimulationApp:
        def __init__(self, _launch_config):
            seen_argv.append(list(sys.argv))

    try:
        fake_isaacsim = types.ModuleType('isaacsim')
        fake_isaacsim.SimulationApp = FakeSimulationApp
        sys.modules['isaacsim'] = fake_isaacsim
        sys.argv = ['isaac_sim_ucx.py', '--scene', '/scene.usd', '--port', '50009']

        module._create_simulation_app({'headless': True})  # pylint: disable=protected-access

        assert seen_argv == [['isaac_sim_ucx.py']]
        assert sys.argv == ['isaac_sim_ucx.py', '--scene', '/scene.usd', '--port', '50009']
    finally:
        sys.argv = original_argv
        if original_isaacsim is None:
            sys.modules.pop('isaacsim', None)
        else:
            sys.modules['isaacsim'] = original_isaacsim


if __name__ == '__main__':
    test_warmup_arguments_are_parsed()
    test_default_arguments_are_stable()
    test_require_scriptnode_consent_argument_is_parsed()
    test_scriptnode_auto_consent_settings_are_enabled_by_default()
    test_require_scriptnode_consent_omits_consent_overrides()
    test_scene_short_option_is_parsed()
    test_sim_dt_explicit_is_parsed()
    test_non_positive_sim_dt_is_rejected()
    test_non_positive_substeps_is_rejected()
    test_resolve_physics_dt_prefers_explicit_value()
    test_idle_stage_plays_renders_and_pauses_before_accepting_a_client()
    test_warmup_can_run_without_stage()
    test_connected_listener_starts_simulation_before_stepping()
    test_substeps_argument_controls_physics_step_count()
    test_physics_rate_log_uses_active_timestep()
    test_physics_timestep_survives_stage_reloads()
    test_g1_control_graph_is_physics_driven_and_camera_remains_playback_driven()
    test_control_substeps_are_authored_on_named_physics_gate()
    test_main_sets_ucx_defaults_before_starting_isaac_sim()
    test_main_passes_stage_to_runtime_after_isaac_startup()
    test_simulation_app_startup_does_not_receive_ucx_script_args()
    print('UCX entrypoint argument tests passed')
