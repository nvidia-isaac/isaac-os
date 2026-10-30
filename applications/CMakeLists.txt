# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_isaac_application(cuvslam_vb1940 DEPENDS
  COMMON camera_info_cpu_messages geometry_cpu_messages pose_tree
  OPERATORS camera_info cuvslam image_conversion
)
add_isaac_application(mock_build)
add_isaac_application(robot_control_benchmark)
add_isaac_application(so101_reach DEPENDS
  COMMON joint_cpu_messages so101_cpu_messages so101_driver
  OPERATORS so101 so101_reach ucxx_ops
)
add_isaac_application(so101_sinus_motion DEPENDS
  COMMON joint_cpu_messages so101_cpu_messages so101_driver
  OPERATORS simple_motion_planner sinus_motion so101 ucxx_ops
)
add_isaac_application(wbc_g1 DEPENDS
  COMMON joint_cpu_messages
  OPERATORS agile_wbc g1_low_level message_drop_sink mock start_up teleop ucxx_ops
            unitree_sim_ucx
)
add_isaac_application(gr00t_n1_7_wbc_g1 DEPENDS
  COMMON joint_command_chunk_cpu joint_cpu_messages
  OPERATORS agile_wbc g1_low_level gr00t_action_handler gr00t_n1_7 joint_command_combiner
            mock sequential_action_handler start_up teleop ucxx_ops unitree_sim_ucx v4l2_camera
)
add_isaac_application(mcap_logger_reference DEPENDS
  COMMON camera_info_cpu_messages geometry_cpu_messages image_schema joint_cpu_messages mcap_logger
)
add_isaac_application(mcap_replay_reference DEPENDS
  COMMON camera_info_cpu_messages geometry_cpu_messages image_schema joint_cpu_messages
         mcap_replayer
)
add_isaac_application(ros2_bridge_reference DEPENDS
  COMMON geometry_cpu_messages ros2_bridge
)
add_isaac_application(sipl_camera)
