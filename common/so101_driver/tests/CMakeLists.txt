# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/gtest)

add_executable(test_feetech_serial_driver test_feetech_serial_driver.cc)
add_executable(test_so101_unit_conversions test_so101_unit_conversions.cc)
target_link_libraries(test_feetech_serial_driver
  PRIVATE
    GTest::gtest_main
    isaac::common::so101_driver
)
target_link_libraries(test_so101_unit_conversions
  PRIVATE
    GTest::gtest_main
    isaac::common::so101_driver
)
add_test(NAME test_feetech_serial_driver COMMAND test_feetech_serial_driver)
add_test(NAME test_so101_unit_conversions COMMAND test_so101_unit_conversions)
set_tests_properties(test_feetech_serial_driver PROPERTIES
  LABELS "common;cpp"
  TIMEOUT 30
)
set_tests_properties(test_so101_unit_conversions PROPERTIES
  LABELS "common;cpp"
  TIMEOUT 30
)

add_executable(test_ftservo_serial test_ftservo_serial.cc)
target_link_libraries(test_ftservo_serial PRIVATE GTest::gtest_main isaac::ftservo)
add_test(NAME test_ftservo_serial COMMAND test_ftservo_serial)
set_tests_properties(test_ftservo_serial PROPERTIES LABELS "common;cpp" TIMEOUT 10)
