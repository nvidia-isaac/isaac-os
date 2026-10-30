/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <thread>

#include "gtest/gtest.h"
#include <cuda_runtime.h>

#include "common/ros2_bridge/cuda_device_utils.h"

namespace isaac::os::ros2_bridge {
namespace {

int Current() {
  int device = -1;
  (void)cudaGetDevice(&device);
  return device;
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

class CudaDeviceUtils : public ::testing::Test {
 protected:
  void SetUp() override {
    if (CudaDeviceCount() == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
  }
};

// A stream outlives the thread that made it: a tensor's backing can be released from a transport
// thread after the creating thread has exited.
TEST_F(CudaDeviceUtils, AStreamStaysUsableAfterTheThreadThatCreatedItExits) {
  detail::PoolDeviceStorage().store(0);
  cudaStream_t stream = nullptr;
  std::jthread worker([&] { stream = ThisThreadStream(); });
  worker.join();
  ASSERT_NE(stream, nullptr);
  int device = -1;
  ASSERT_EQ(cudaStreamGetDevice(stream, &device),
            cudaSuccess);  // also fails for a destroyed stream
  EXPECT_EQ(device, PoolDevice());
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  detail::PoolDeviceStorage().store(-1);
}

TEST_F(CudaDeviceUtils, AnUnboundPoolDeviceIsAnErrorNotGpuZero) {
  detail::PoolDeviceStorage().store(-1);
  EXPECT_THROW((void)PoolDevice(), std::runtime_error);
  EXPECT_THROW(UsePoolDeviceOnThisThread(), std::runtime_error);
}

// Two-GPU tests: skipped on a single-GPU host, so a multi-GPU runner is the tester.
TEST_F(CudaDeviceUtils, ScopedDeviceSwitchesAndRestoresAcrossDevices) {
  if (CudaDeviceCount() < 2) GTEST_SKIP() << "needs two CUDA devices";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  {
    ScopedDevice scope(1);
    EXPECT_EQ(Current(), 1);
  }
  EXPECT_EQ(Current(), 0);
}

TEST_F(CudaDeviceUtils, AThreadStartedAfterTheBindUsesThePoolDevice) {
  if (CudaDeviceCount() < 2) GTEST_SKIP() << "needs two CUDA devices";
  detail::PoolDeviceStorage().store(1);
  int current = -1;
  int stream_device = -1;
  std::jthread worker([&] {
    UsePoolDeviceOnThisThread();
    current = Current();
    (void)cudaStreamGetDevice(ThisThreadStream(), &stream_device);
  });
  worker.join();
  detail::PoolDeviceStorage().store(-1);
  EXPECT_EQ(current, 1);
  EXPECT_EQ(stream_device, 1);
}

}  // namespace
}  // namespace isaac::os::ros2_bridge
