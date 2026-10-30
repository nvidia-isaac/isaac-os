# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

include(isaac_add_resource)

isaac_add_resource(grip_reach_delta_policy_onnx
  URL "https://huggingface.co/nvidia/so101-reach/resolve/f88e85072762f36fd1bc58d93ca6aa8b4bbe2db7/policy_fingertip_delta.onnx"
  EXPECTED_HASH "SHA256=25e08afb6bc6cd62e669996303e8cd1f7a0a2aaa98e6b808b544a7d2b51e308f"
)
