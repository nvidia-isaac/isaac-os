# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a minimal identity model for TensorRTEngine tests.

Requires the optional ``onnx`` Python package, which is not installed in the development image.
"""

import os

import onnx
from onnx import TensorProto
from onnx import helper

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4])
node = helper.make_node('Identity', inputs=['input'], outputs=['output'])
graph = helper.make_graph([node], 'identity', [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
onnx.checker.check_model(model)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'identity.onnx')
onnx.save(model, out)
print(f'Saved {model.ByteSize()} bytes to {out}')
