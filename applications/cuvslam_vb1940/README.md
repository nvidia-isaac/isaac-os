# cuVSLAM on a VB1940 stereo rig

Tracks the pose of a Leopard Eagle VB1940 stereo camera with cuVSLAM. Frames are captured through
SIPL in ISP mode, converted to color on the GPU, and paired by projected HSB first-data time
before they reach the tracker, which publishes each estimate into a tree of coordinate frames
the application owns for the run.

```text
SIPLCaptureOp ─┬─> Nv12ToRgbOp ─┬─> HolovizOp   (unless headless)
  (per camera) │                │
               │                └─> CuvslamOp
               │                      ▲
               ├─> CameraInfoOp ──────┘
               │
               └─> SensorTelemetrySinkOp
```

One `SIPLCaptureOp` is created per camera the description declares, all sharing a single
`SIPLCaptureService`, and the tracker takes a frame from every one of them. The rig size comes from
the description rather than from configuration: two cameras up to `CuvslamOp::kMaxCameras`. Each
camera is identified by its position in the rig, and the window shows them in a row of equal
shares.

SIPL supplies projected HSB first-data time in each Image header and scheduling metadata.
`OnAligned` pairs the existing image ports using that time; each image's timestamp also reaches
cuVSLAM. This marks data reaching the HSB FPGA and has not been calibrated to exposure start,
end or midpoint. Host PTP must remain running; see the shared
[PTP setup](../../docs/how-to/set_up_an_hsb_camera_for_sipl.mdx#ptp-clock-synchronization).

## Running

[`common/sipl/config/vb1940_stereo.json`](../../common/sipl/config) is the SIPL platform
description shared with the other SIPL applications. Before running the application,
[set up the HSB camera for SIPL](../../docs/how-to/set_up_an_hsb_camera_for_sipl.mdx).

```sh
./isaac_os run cuvslam_vb1940
```

| Option | Meaning |
| --- | --- |
| `--application-config-path` | YAML described under [Camera Calibration](#camera-calibration). |
| `--json-config` | SIPL platform description: the camera modules and their MAC and IP addresses. Must match the board. |
| `--camera-config` | Selects one configuration by name from that file. Defaults to the first that names a platform. |
| `--alignment-window-ms` | Maximum difference between projected HSB first-data timestamps in a stereo pair. Defaults to 1 ms; wider is diagnostic only. This does not establish exposure synchronization. |
| `--window-width`, `--window-height` | Size of the camera window. |
| `--headless` | Track without a window, for a machine with no display attached. |
| `--cuda-device` | Selects the GPU. Only device 0 is accepted; a nonzero value is rejected rather than half-applied. |
| `--validate` | Build and check the pipeline, then exit without capturing. |

Press Ctrl+C to stop capture and deinitialize the sensors and sensor bridge.

The tracker prints a progress report every `report_period_s`: frame sets per second, how far apart
the frames within a set were captured, the range of gaps between sets, and where the rig is.

## Camera Calibration

cuVSLAM requires a camera calibration file (`config/config.yaml`) to run.

### Generate the calibration file

Whenever you connect a new VB1940 camera, run this command from the Isaac OS checkout to download
its factory RGB calibration from EEPROM into `config/config.yaml`:

```sh
sh applications/cuvslam_vb1940/generate_vb1940_config.sh
```

The generated YAML uses the left camera as the rig origin.

### File format

The file contains intrinsics and placement for each camera, plus tracker settings.
See the [example configuration](config/config.example.yaml) for the expected format with
illustrative values.

| Section | Contents |
| --- | --- |
| `camera_info_<i>` | Intrinsics for camera `<i>`: `frame_id`, `fx`, `fy`, `cx`, `cy`, `stereo_baseline`, `distortion_model`, `distortion_parameters`. |
| `camera_placement_<i>` | Where camera `<i>` sits in the rig, as `translation` and `rotation_wxyz`. Written into the coordinate-frame tree before the run starts. |
| `cuvslam` | `odometry_frame`, `rig_frame`, `odometry_mode`, `async_setup`, `report_period_s`. |

## Stereo hardware synchronization

> [!WARNING]
> Synchronized stereo capture has a known platform startup issue: both streams can stop delivering
> frames within the first few seconds or mid-run.  If that happens restart the application.

The tracker pairs frames on HSB first-data time. Hardware synchronization remains necessary
for stereo tracking. The shared platform description asks for hardware
synchronization: both sensors share `sensorGroup`, take a `deviceIndex` apiece, and the transport
is asked for `sync_sensors`. The sensor bridge reports what it made of that at start-up, and this
is the line to read first:

```text
[HSB] NotifyAllCamerasStreaming: activeSensors=0x3, syncSensors=0x3, num_sync_sensors=2
[HSB] VSYNC ENABLED: 2 sensors configured for stereo sync at 30Hz
```

Were the sensors free-running instead, it would read:

```text
[HSB] VSYNC DISABLED: No sensors configured for sync (single camera / non-stereo mode)
```

Two free-running 30 Hz sensors drift through every relative phase, so only a varying fraction of
frames land inside a narrow window and the rest are dropped before they reach the tracker.

Widening `--alignment-window-ms` admits more frame sets but does not increase the tracker's
supported synchronization tolerance. Use a wider window only to diagnose pairing, and validate
exposure synchronization and tracking accuracy before relying on the poses.
