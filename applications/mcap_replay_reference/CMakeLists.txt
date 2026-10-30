# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)
include(isaac_deps/mcap)

add_executable(mcap_replay_reference main.cc)
target_include_directories(mcap_replay_reference PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(mcap_replay_reference
  PRIVATE
    holoscan::core
    holoscan::visualization::op_holoviz
    isaac::common::camera_info_cpu_messages
    isaac::common::geometry_cpu_messages
    isaac::common::image_schema
    isaac::common::joint_cpu_messages
    isaac::common::mcap_replayer
    # main.cc opens an mcap reader of its own to read how many messages the recording holds, so
    # it can stop after one pass. mcap_replayer propagates mcap's headers to its consumers but
    # deliberately not its library, so this executable links it for itself.
    mcap
)

if(ISAAC_BUILD_TESTS)
  # Replaying needs a recording, which this project does not produce, so the smoke test gates
  # what can be checked without one: that the executable links, starts, and reports a usage
  # problem rather than crashing or hanging on it.
  add_test(NAME test_mcap_replay_reference_usage COMMAND mcap_replay_reference --help)
  set_tests_properties(test_mcap_replay_reference_usage PROPERTIES
    LABELS "applications;cpp;integration"
    TIMEOUT 30
  )
  add_test(NAME test_mcap_replay_reference_rejects_missing_input
    COMMAND mcap_replay_reference --input-path=${CMAKE_CURRENT_BINARY_DIR}/no_such_recording.mcap
  )
  set_tests_properties(test_mcap_replay_reference_rejects_missing_input PROPERTIES
    LABELS "applications;cpp;integration"
    TIMEOUT 30
    WILL_FAIL TRUE
  )
endif()
