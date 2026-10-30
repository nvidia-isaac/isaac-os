# SO-101 Isaac Sim USD assets

This directory contains the SO-101 USD assets used by the UCX Isaac Sim workflow.

## Files

- `so101_ucx_with_env.usda`: ready-to-run Isaac Sim scene. It references
  `so101_ucx_robot_only.usda`, places the robot in the Simple Room environment,
  lowers it by 3 cm, rotates it 180 degrees around Z to face the desired side of
  the room, defines the physics scene, and preserves the Kit viewport/render
  camera settings.
- `so101_reach_ucx_with_env.usda`: scene for the SO-101 reach `sim` mode. It
  sublayers `so101_ucx_with_env.usda`, repeats its timeline range because Kit reads
  that range from the root layer only, waits at most 10 ms for each joint command
  so physics advances when the controller sends none, disables clock publication
  and camera rendering, sets a 100 Hz physics step, and raises the elbow drive
  damping to the shoulder pan joint's value to stop a 2 Hz elbow oscillation.
- `so101_ucx_robot_only.usda`: reusable robot-only layer. It references the
  hosted Isaac Sim SO-101 SysID robot with its camera mount and calibrated
  WowRobo camera, sets its sensor tick rate to 60 Hz, then adds the UCX ActionGraph.
  It intentionally does not
  include an environment, scene placement, physics scene, or Kit viewport
  state.

## Upstream assets and licenses

The base SO-101 robot is not vendored here. `so101_ucx_robot_only.usda`
references the Apache-2.0 licensed Isaac Sim 6.1 hosted asset:

```text
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Robots_Multiphysics/RobotStudio/so101_new_calib_SysID/so101_new_calib.usda
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Robots_Multiphysics/RobotStudio/so101_new_calib_SysID/LICENSE
```

The scene environment is not vendored here. `so101_ucx_with_env.usda`
references the CC-BY-4.0 licensed Isaac Sim 6.1 Simple Room:

```text
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Environments/Simple_Room/simple_room.usd
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Environments/Simple_Room/LICENSE
```
