# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/ftservo)
include(isaac_deps/holoscan)
include(isaac_deps/nlohmann_json)

add_library(so101_driver SHARED
  feetech_serial_driver.cc
  so101_driver_config.cc
  so101_unit_conversions.cc
)
add_library(isaac::common::so101_driver ALIAS so101_driver)

target_include_directories(so101_driver
  PUBLIC
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(so101_driver
  PUBLIC
    holoscan::config_io
  PRIVATE
    isaac::ftservo
    nlohmann_json::nlohmann_json
)

isaac_install(
  COMPONENT LIB_so101_driver
  TARGETS so101_driver
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
