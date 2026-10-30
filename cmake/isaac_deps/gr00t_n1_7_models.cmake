# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

include(isaac_add_resource)

set(_gr00t_n1_7_g1_apple_url "https://huggingface.co/nvidia/GR00T-N1.7-ApplePnP-V1/resolve/90caf594a5f1ce32e114103d6debf3ffabdb802d")
set(_gr00t_n1_7_g1_apple_target_prefix "gr00t_n1_7_g1_apple_to_plate")

isaac_add_resource_bundle(${_gr00t_n1_7_g1_apple_target_prefix}_models
  LINK_DIR ${PROJECT_SOURCE_DIR}/data/gr00t_n1_7/g1_apple_to_plate

  RESOURCE ${_gr00t_n1_7_g1_apple_target_prefix}_action_head_data
    URL ${_gr00t_n1_7_g1_apple_url}/action_head.onnx.data
    EXPECTED_HASH SHA256=e089c3f6285bdd450d8713234320d80e532e5ed543ae05db392656e92c842457

  RESOURCE ${_gr00t_n1_7_g1_apple_target_prefix}_action_head_onnx
    URL ${_gr00t_n1_7_g1_apple_url}/action_head.onnx
    EXPECTED_HASH SHA256=5673ec0559786ac4bb3aadd328afd08bcd416cf758e0b3d42099546ce144e896

  RESOURCE ${_gr00t_n1_7_g1_apple_target_prefix}_backbone_data
    URL ${_gr00t_n1_7_g1_apple_url}/backbone.onnx.data
    EXPECTED_HASH SHA256=78dbdaf48fdfea13dd1f125e7e17cae982c605fec0d41be2a9c64c812fd63832

  RESOURCE ${_gr00t_n1_7_g1_apple_target_prefix}_backbone_onnx
    URL ${_gr00t_n1_7_g1_apple_url}/backbone.onnx
    EXPECTED_HASH SHA256=ea2d384302994a26e674f0ec4908879f5e7b808965d9119207a260d56e2833a1

  RESOURCE ${_gr00t_n1_7_g1_apple_target_prefix}_decode_action
    URL ${_gr00t_n1_7_g1_apple_url}/decode_action.onnx
    EXPECTED_HASH SHA256=9f96907df55558c09e11265b02bc97424f0b40ed57cac2d2f63b8c850b0e76c1

  RESOURCE ${_gr00t_n1_7_g1_apple_target_prefix}_preprocess_state
    URL ${_gr00t_n1_7_g1_apple_url}/preprocess_state.onnx
    EXPECTED_HASH SHA256=81982f70fa8d62d36c3b47606c80dcc3c8cce08001e395348492ea7c402ffe6b

  RESOURCE ${_gr00t_n1_7_g1_apple_target_prefix}_preprocess_video
    URL ${_gr00t_n1_7_g1_apple_url}/preprocess_video.onnx
    EXPECTED_HASH SHA256=0bd9e5752cc3715fda896271e86db83ba3f8f650522f851086157bc9b0f6f39a
)

set(_gr00t_n1_7_g1_static_apple_sim_url
    "https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace/resolve/7f78bebf1a90131e7304beacfcd47eb27bad16ab/exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2")
set(_gr00t_n1_7_g1_static_apple_sim_target_prefix "gr00t_n1_7_g1_static_apple_sim")

isaac_add_resource_bundle(${_gr00t_n1_7_g1_static_apple_sim_target_prefix}_models
  LINK_DIR ${PROJECT_SOURCE_DIR}/data/gr00t_n1_7/g1_static_apple_sim

  RESOURCE ${_gr00t_n1_7_g1_static_apple_sim_target_prefix}_action_head_data
    URL ${_gr00t_n1_7_g1_static_apple_sim_url}/action_head.onnx.data
    EXPECTED_HASH SHA256=3d81ca3b672c2e7ee68170a4fc8749bfe3e10eb84bfcdbd5377b4b3fd0922ae1

  RESOURCE ${_gr00t_n1_7_g1_static_apple_sim_target_prefix}_action_head_onnx
    URL ${_gr00t_n1_7_g1_static_apple_sim_url}/action_head.onnx
    EXPECTED_HASH SHA256=50f4f8e38a8311cc581e34bd4e305590f12117c04cb5cbe0e438172359911ca8

  RESOURCE ${_gr00t_n1_7_g1_static_apple_sim_target_prefix}_backbone_data
    URL ${_gr00t_n1_7_g1_static_apple_sim_url}/backbone.onnx.data
    EXPECTED_HASH SHA256=afcefc517d3f477187f3a55f69c632c88309ba84f841fcf7e394b11733935432

  RESOURCE ${_gr00t_n1_7_g1_static_apple_sim_target_prefix}_backbone_onnx
    URL ${_gr00t_n1_7_g1_static_apple_sim_url}/backbone.onnx
    EXPECTED_HASH SHA256=ea2d384302994a26e674f0ec4908879f5e7b808965d9119207a260d56e2833a1

  RESOURCE ${_gr00t_n1_7_g1_static_apple_sim_target_prefix}_decode_action
    URL ${_gr00t_n1_7_g1_static_apple_sim_url}/decode_action.onnx
    EXPECTED_HASH SHA256=13cef938a2d07b98e79a7666c67f81ea4ea7b97091c75494b4f932c0df62cf5d

  RESOURCE ${_gr00t_n1_7_g1_static_apple_sim_target_prefix}_preprocess_state
    URL ${_gr00t_n1_7_g1_static_apple_sim_url}/preprocess_state.onnx
    EXPECTED_HASH SHA256=33b5eff99674daac13d65a92b5274efc5efd52b0ad21618ebcbf2327394555fa

  RESOURCE ${_gr00t_n1_7_g1_static_apple_sim_target_prefix}_preprocess_video
    URL ${_gr00t_n1_7_g1_static_apple_sim_url}/preprocess_video.onnx
    EXPECTED_HASH SHA256=0bd9e5752cc3715fda896271e86db83ba3f8f650522f851086157bc9b0f6f39a
)
