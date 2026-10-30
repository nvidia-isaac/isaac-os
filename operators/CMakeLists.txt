# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_isaac_operator(agile_wbc DEPENDS
  COMMON geometry_cpu_messages joint_cpu_messages tensorrt_engine
)
add_isaac_operator(g1_low_level DEPENDS
  COMMON joint_cpu_messages
)
add_isaac_operator(camera_info DEPENDS
  COMMON camera_info_cpu_messages
)
add_isaac_operator(cuvslam DEPENDS
  COMMON camera_info_cpu_messages geometry_cpu_messages pose_tree
)
add_isaac_operator(gr00t_action_handler DEPENDS
  COMMON joint_command_chunk_cpu joint_cpu_messages
)
add_isaac_operator(simple_motion_planner DEPENDS
  COMMON joint_cpu_messages so101_cpu_messages
)
add_isaac_operator(sinus_motion DEPENDS
  COMMON joint_cpu_messages so101_cpu_messages
)
add_isaac_operator(so101 DEPENDS
  COMMON joint_cpu_messages so101_cpu_messages so101_driver
)
add_isaac_operator(so101_reach DEPENDS
  COMMON so101_cpu_messages tensorrt_engine
)
add_isaac_operator(gr00t_n1_7 DEPENDS
  COMMON joint_command_chunk_cpu joint_cpu_messages tensorrt_engine
)
add_isaac_operator(image_conversion)
add_isaac_operator(joint_command_combiner DEPENDS
  COMMON joint_cpu_messages
)
add_isaac_operator(message_drop_sink)
add_isaac_operator(mock DEPENDS
  COMMON joint_cpu_messages
)
add_isaac_operator(sequential_action_handler DEPENDS
  COMMON joint_command_chunk_cpu joint_cpu_messages
)
add_isaac_operator(start_up DEPENDS
  COMMON joint_cpu_messages
)
add_isaac_operator(teleop DEPENDS
  COMMON geometry_cpu_messages
)
add_isaac_operator(ucxx_ops)
add_isaac_operator(unitree_sim_ucx DEPENDS
  COMMON joint_cpu_messages
  OPERATORS ucxx_ops
)
add_isaac_operator(v4l2_camera)
