# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Carries no library, only this graph's tests for the SDK-independent workload sources. The
# workload itself is compiled by each runtime that consumes it.
add_subdirectory(robot_control_benchmark)

# Must be listed above every library in its DEPENDS:
# add_isaac_common() forces LIB_<dep> ON at the point the *dependent* is processed, so a library
# with DEPENDS has to appear above its dependencies or the first configure of a clean build
# directory fails (a warm build directory hides this, since by then every LIB_<dep> is already
# cached ON).
add_isaac_common(mcap_replayer DEPENDS geometry_cpu_messages image_schema mcap_logger)
add_isaac_common(joint_command_chunk_cpu)
add_isaac_common(ros2_bridge DEPENDS camera_info_cpu_messages geometry_cpu_messages image_schema joint_cpu_messages math_cpu_messages)
add_isaac_common(mcap_logger DEPENDS image_schema joint_cpu_messages)
# Only mcap uses the video codec libraries. LIB_mcap_replayer forces LIB_mcap_logger ON above.
if(ISAAC_ENABLE_NV_VIDEO_SDK AND LIB_mcap_logger)
  add_subdirectory(nv_video_decoder)
  add_subdirectory(nv_video_encoder)
endif()
add_isaac_common(image_schema)
add_isaac_common(joint_cpu_messages)
add_isaac_common(geometry_cpu_messages DEPENDS math_cpu_messages)
add_isaac_common(camera_info_cpu_messages DEPENDS math_cpu_messages)
add_isaac_common(math_cpu_messages)
add_isaac_common(so101_cpu_messages)
add_isaac_common(so101_driver)
add_isaac_common(pose_tree)
add_isaac_common(tensorrt_engine)
