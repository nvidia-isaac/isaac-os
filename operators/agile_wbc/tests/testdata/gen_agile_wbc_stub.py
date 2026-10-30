# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the WBC test stub, exposing velocity commands as its first three positions.

Requires the optional ``onnx`` Python package, which is not installed in the development image.
"""

from pathlib import Path

import onnx
from onnx import TensorProto
from onnx import helper

HISTORY_TERMS = [
    ('base_ang_vel_history', 3 * 5),
    ('projected_gravity_history', 3 * 5),
    ('velocity_commands_history', 3 * 5),
    ('controlled_joint_pos_history', 14 * 5),
    ('controlled_joint_vel_history', 14 * 5),
    ('actions_history', 14 * 5),
    ('last_actions', 14),
]


def tensor_info(name, size):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [size])


def zeros(name, size):
    return helper.make_node(
        'Constant',
        [],
        [name],
        value=helper.make_tensor(name + '_value', TensorProto.FLOAT, [size], [0.0] * size),
    )


inputs = [
    tensor_info('root_link_quat_w', 4),
    tensor_info('root_ang_vel_b', 3),
    tensor_info('velocity_commands', 3),
    tensor_info('joint_pos', 29),
    tensor_info('joint_vel', 29),
] + [tensor_info(name, size) for name, size in HISTORY_TERMS]

zero_outputs = [
    ('action_joint_pos_kp_gains', 14),
    ('action_joint_pos_kd_gains', 14),
] + [(name + '_out', size) for name, size in HISTORY_TERMS]
outputs = [tensor_info('action_joint_pos', 14)] + [
    tensor_info(name, size) for name, size in zero_outputs
]
nodes = [
    zeros('remaining_positions', 11),
    helper.make_node(
        'Concat', ['velocity_commands', 'remaining_positions'], ['action_joint_pos'], axis=0
    ),
] + [zeros(name, size) for name, size in zero_outputs]

model = helper.make_model(
    helper.make_graph(nodes, 'agile_wbc_stub', inputs, outputs),
    opset_imports=[helper.make_opsetid('', 13)],
)
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name('agile_wbc_stub.onnx'))
