/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <string>

#include "gtest/gtest.h"
#include <cuda_runtime.h>

#include "common/ros2_bridge/cuda_device_utils.h"

namespace isaac::os::ros2_bridge {
namespace {

// "" when the bind succeeded, otherwise the error message.
std::string BindError(int device) {
  try {
    BindPoolDevice(device);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return "";
}

// How many CUDA devices this machine has; 0 when there is no usable driver.
int CudaDeviceCount() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}

class CudaPoolDevice : public ::testing::Test {
 protected:
  void SetUp() override {
    if (CudaDeviceCount() == 0) {
      GTEST_SKIP() << "no CUDA device";
    }
  }
};

// Tests run in declaration order in this binary and share one process-wide pool.
TEST_F(CudaPoolDevice, ADeviceThatDoesNotExistIsAnErrorNamingIt) {
  EXPECT_NE(BindError(1000).find("1000"), std::string::npos);
  EXPECT_FALSE(PoolDeviceBound());
}

TEST_F(CudaPoolDevice, BindsThePoolToTheRequestedDeviceAndRecordsIt) {
  EXPECT_EQ(BindError(0), "");
  EXPECT_TRUE(PoolDeviceBound());
  EXPECT_EQ(PoolDevice(), 0);
}

TEST_F(CudaPoolDevice, BindingTheSameDeviceAgainSucceeds) { EXPECT_EQ(BindError(0), ""); }

TEST_F(CudaPoolDevice, BindingADifferentDeviceAfterwardsIsAnErrorNamingBoth) {
  if (CudaDeviceCount() < 2) GTEST_SKIP() << "needs two CUDA devices";
  const std::string error = BindError(1);
  EXPECT_NE(error.find("GPU 0"), std::string::npos);
  EXPECT_NE(error.find("GPU 1"), std::string::npos);
  EXPECT_EQ(PoolDevice(), 0);  // the pool stays where it was
}

}  // namespace
}  // namespace isaac::os::ros2_bridge
