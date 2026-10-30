/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <string>

#include "gtest/gtest.h"
#include "holoscan/core/errors.hpp"

#include "common/ros2_bridge/cuda_buffer_config.h"
#include "common/ros2_bridge/memory_transfer.h"

namespace isaac::os::ros2_bridge {
namespace {

constexpr Ros2MemoryKind kHost = Ros2MemoryKind::kHost;
constexpr Ros2MemoryKind kDevice = Ros2MemoryKind::kCudaDevice;
constexpr Placement kOnHost{kHost, 0};
constexpr Placement kOnGpu0{kDevice, 0};
constexpr Placement kOnGpu1{kDevice, 1};
constexpr Placement kOnGpu2{kDevice, 2};

TEST(MemoryTransfer, MemoryKindDefaultsToHostAndExistingDesignatedInitializersStillCompile) {
  const Ros2TopicSpec spec{.port_name = "p", .ros2_topic = "/t", .max_tensor_bytes = 16U};
  EXPECT_EQ(spec.memory_kind, Ros2MemoryKind::kHost);
}

TEST(MemoryTransfer, IngressRules) {
  using enum TransferAction;
  struct Case {
    Placement source;
    Ros2MemoryKind target;
    int pool_device;
    TransferAction expected;
  };
  constexpr std::array<Case, 8> kCases{{
      {kOnGpu0, kDevice, 0, kWrapDevice},
      {kOnGpu1, kDevice, 0, kReject},
      {kOnHost, kDevice, 0, kHostToDevice},
      {kOnGpu0, kHost, 0, kDeviceToHost},
      {kOnGpu1, kHost, 0, kDeviceToHost},
      {kOnHost, kHost, 0, kWrapHost},
      // The pool device need not be 0.
      {kOnGpu2, kDevice, 2, kWrapDevice},
      {kOnGpu0, kDevice, 2, kReject},
  }};
  for (const Case& c : kCases) {
    EXPECT_EQ(PlanIngress(c.source, c.target, c.pool_device), c.expected)
        << "source kind " << static_cast<int>(c.source.kind) << " on GPU " << c.source.device
        << ", target kind " << static_cast<int>(c.target) << ", pool GPU " << c.pool_device;
  }
}

TEST(MemoryTransfer, EgressRules) {
  using enum TransferAction;
  EXPECT_EQ(PlanEgress(kOnHost, kDevice), kHostToDevice);
  EXPECT_EQ(PlanEgress(kOnGpu0, kDevice), kDeviceToDevice);
  EXPECT_EQ(PlanEgress(kOnGpu1, kDevice), kDeviceToDevice);
  EXPECT_EQ(PlanEgress(kOnGpu0, kHost), kDeviceToHost);
  EXPECT_EQ(PlanEgress(kOnHost, kHost), kHostCopy);
}

TEST(MemoryTransfer, IngressSubscriptionAsksForCudaBuffersOnlyFromAGpuPublisherOnADeviceTopic) {
  const auto backends = [](Ros2MemoryKind topic, Ros2MemoryKind publisher) {
    return std::string(AcceptableBufferBackends(
        Ros2TopicSpec{.memory_kind = topic, .publisher_memory = publisher}));
  };
  EXPECT_EQ(backends(Ros2MemoryKind::kCudaDevice, Ros2MemoryKind::kCudaDevice), "cuda");
  EXPECT_EQ(backends(Ros2MemoryKind::kCudaDevice, Ros2MemoryKind::kHost), "cpu");
  EXPECT_EQ(backends(Ros2MemoryKind::kHost, Ros2MemoryKind::kCudaDevice), "cpu");
  EXPECT_EQ(backends(Ros2MemoryKind::kHost, Ros2MemoryKind::kHost), "cpu");
  // The default publisher memory keeps today's behaviour for a device topic.
  EXPECT_EQ(std::string(AcceptableBufferBackends(
                Ros2TopicSpec{.memory_kind = Ros2MemoryKind::kCudaDevice})),
            "cuda");
}

TEST(CudaBufferConfig, DeviceTopicWithoutTheBackendThrowsAndNamesTheTopic) {
  Ros2TopicSpec spec{.port_name = "camera_in"};
  spec.memory_kind = Ros2MemoryKind::kCudaDevice;
  try {
    RequireCudaBufferBackend(spec, /*built_in=*/false);
    FAIL() << "expected holoscan::RuntimeError";
  } catch (const holoscan::RuntimeError& error) {
    const std::string what = error.what();
    EXPECT_NE(what.find("camera_in"), std::string::npos);
    EXPECT_NE(what.find("kCudaDevice"), std::string::npos);
    EXPECT_NE(what.find("cuda_buffer"), std::string::npos);
  }
}

TEST(CudaBufferConfig, HostTopicNeverNeedsTheBackend) {
  EXPECT_NO_THROW(RequireCudaBufferBackend(Ros2TopicSpec{.port_name = "camera_in"},
                                           /*built_in=*/false));
}

}  // namespace
}  // namespace isaac::os::ros2_bridge
