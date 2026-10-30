# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export VB1940 RGB factory calibration for cuVSLAM's color streams."""

import argparse
import copy
import logging
import math
import os
from pathlib import Path
import stat
import tempfile


def _vector(value, length, name):
    try:
        if len(value) != length:
            raise ValueError()
        result = [float(element) for element in value]
        if not all(math.isfinite(element) for element in result):
            raise ValueError()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must contain {length} finite numbers') from error
    return result


def _camera_info(calibration, side):
    name = f'{side}_intrinsic_parameter'
    matrix = calibration[name]
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 3:
        raise ValueError(f'{name} must be a 3 by 3 matrix')
    matrix = [_vector(row, 3, name) for row in matrix]
    fx, fy = matrix[0][0], matrix[1][1]
    if fx <= 0 or fy <= 0:
        raise ValueError(f'{side} focal lengths must be positive; check EEPROM data')

    name = f'{side}_distortion_parameters'
    distortion = calibration[name]
    if not isinstance(distortion, (list, tuple)) or len(distortion) not in (8, 14):
        raise ValueError(f'{name} must contain 8 coefficients or 14 with zero padding')
    distortion = _vector(distortion, len(distortion), name)
    if any(distortion[8:]):
        raise ValueError(f'{name} has unsupported nonzero coefficients after the first 8')
    return {
        'fx': fx,
        'fy': fy,
        'cx': matrix[0][2],
        'cy': matrix[1][2],
        'stereo_baseline': 0.0,
        'distortion_model': 'polynomial',
        'distortion_parameters': distortion[:8],
    }


def _right_placement(rotation_vector, translation, scale):
    # Interpret EEPROM R/T using OpenCV stereo: right = R * left + T.
    # The pose tree instead stores left_from_right: (R^T, -R^T * T).
    angle = math.hypot(*rotation_vector)
    if not math.isfinite(angle):
        raise ValueError('R rotation-vector magnitude must be finite')
    if angle == 0:
        quaternion = [1.0, 0.0, 0.0, 0.0]
    else:
        sine = math.sin(angle / 2)
        quaternion = [math.cos(angle / 2)] + [-(value / angle) * sine for value in rotation_vector]
    w, x, y, z = quaternion
    inverse_rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    position = [
        -sum(row[index] * translation[index] * scale for index in range(3))
        for row in inverse_rotation
    ]
    return {
        'translation': _vector(position, 3, 'camera placement translation'),
        'rotation_wxyz': quaternion,
    }


def calibration_to_config(calibration, template=None, translation_unit='m'):
    """Convert a parsed RGB block to the camera_info/camera_placement schema.

    Interpret R as an OpenCV Rodrigues rotation vector in radians, mapping left
    camera coordinates into right camera coordinates. T defaults to meters, as
    observed on Eagle; use translation_unit="mm" for millimeter calibration.
    Intrinsics retain their original pixel coordinates (no resizing/cropping).
    """
    if not isinstance(calibration, dict):
        raise ValueError('Calibration must be a parsed RGB mapping')
    if translation_unit not in ('m', 'mm'):
        raise ValueError("translation_unit must be 'm' or 'mm'")
    if template is not None and not isinstance(template, dict):
        raise ValueError('The YAML template must contain a mapping')
    config = copy.deepcopy(template) if template is not None else {}

    def section(name):
        value = config.setdefault(name, {})
        if not isinstance(value, dict):
            raise ValueError(f'Template section {name} must be a mapping')
        return value

    try:
        for index, side in enumerate(('left', 'right')):
            info = section(f'camera_info_{index}')
            info.setdefault('frame_id', f'vb1940_{index}')
            info.update(_camera_info(calibration, side))
        rotation = _vector(calibration['R'], 3, 'R')
        translation = _vector(calibration['T'], 3, 'T')
    except KeyError as error:
        raise ValueError(f'Missing EEPROM calibration field: {error.args[0]}') from error

    config.setdefault(
        'cuvslam',
        {
            'odometry_frame': 'odom',
            'rig_frame': 'vb1940',
            'odometry_mode': 'multicamera',
            'async_setup': True,
            'report_period_s': 2.0,
        },
    )
    section('camera_placement_0').update(
        {'translation': [0.0, 0.0, 0.0], 'rotation_wxyz': [1.0, 0.0, 0.0, 0.0]}
    )
    section('camera_placement_1').update(
        _right_placement(rotation, translation, 0.001 if translation_unit == 'mm' else 1.0)
    )
    return config


def read_calibration(address):
    # Keep hardware imports lazy so conversion and --help work without the SDK.
    import hololink  # pylint: disable=import-outside-toplevel
    from hololink.sensors.vb1940 import Vb1940Cam  # pylint: disable=import-outside-toplevel

    metadata = hololink.Enumerator.find_channel(channel_ip=address)
    hololink.DataChannel.use_sensor(metadata, 0)
    channel = hololink.DataChannel(metadata)
    bridge = channel.hololink()
    bridge.start()
    try:
        # Sensor 0's EEPROM contains both cameras; no stream or reset is needed.
        camera = Vb1940Cam(channel)
        return camera.get_calibration_data(0)['RGB']
    except KeyError as error:
        raise ValueError('The camera EEPROM does not contain RGB calibration') from error
    finally:
        bridge.stop()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--hololink',
        default='192.168.0.2',  # NOSONAR(S1313) Matches SIPL config; --hololink overrides it.
        help='Eagle sensor bridge IP address',
    )
    parser.add_argument(
        '--mode',
        choices=('rgb',),
        default='rgb',
        help='Only RGB calibration is supported: cuVSLAM captures color frames',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=Path(__file__).resolve().parent / 'config' / 'config.yaml',
        help='Destination YAML file (default: config/config.yaml beside this script)',
    )
    parser.add_argument(
        '--template',
        type=Path,
        help='Preserve frame IDs, cuvslam settings and other fields from this YAML',
    )
    parser.add_argument(
        '--translation-unit',
        choices=('m', 'mm'),
        default='m',
        help='EEPROM T units (default: m); output placement is always in meters',
    )
    parser.epilog = (
        "Relative template/output paths use the caller's working directory. "
        'Run in an HSB environment, or use generate_vb1940_config.sh to prepare one.'
    )
    return parser


def write_config(output, document):
    """Replace the calibration only after the complete new file is flushed."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=output.parent,
            prefix=f'.{output.name}.',
            suffix='.tmp',
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            if output.exists():
                os.fchmod(stream.fileno(), stat.S_IMODE(output.stat().st_mode))
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_calibration(args):
    # Keep --help available without the HSB environment's dependencies.
    import yaml  # pylint: disable=import-outside-toplevel

    try:
        template = None
        if args.template is not None:
            template = yaml.safe_load(args.template.read_text(encoding='utf-8'))
            if not isinstance(template, dict):
                raise ValueError('The YAML template must contain a mapping')
        calibration = read_calibration(args.hololink)
        config = calibration_to_config(calibration, template, args.translation_unit)
        document = (
            '# Eagle VB1940 RGB EEPROM calibration.\n'
            '# EEPROM R/T interpreted as OpenCV Rodrigues left-to-right extrinsics.\n'
            '# Native calibration pixels; left camera is the rig origin.\n'
            '# Placements are rig_from_camera, in meters, with wxyz quaternions.\n'
            + yaml.safe_dump(config, sort_keys=False)
        )
    except yaml.YAMLError as error:
        raise ValueError(f'Unable to read or serialize calibration YAML: {error}') from error
    write_config(args.output, document)
    print(f'Saved RGB stereo calibration to {args.output}')


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        args.output = args.output.expanduser().resolve()
        if args.template is not None:
            args.template = args.template.expanduser().resolve(strict=True)
            if not args.template.is_file():
                raise ValueError(f'The YAML template is not a regular file: {args.template}')
        export_calibration(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f'Calibration export failed: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
