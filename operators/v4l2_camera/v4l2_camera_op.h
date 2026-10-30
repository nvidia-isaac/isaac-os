/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_V4L2_CAMERA_V4L2_CAMERA_OP_H
#define ISAAC_OS_OPERATORS_V4L2_CAMERA_V4L2_CAMERA_OP_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace isaac::os::ops {

struct V4l2CameraParams {
  std::string device{"/dev/video0"};
  std::int64_t width{640};      ///< Pixels; must be even for YUYV.
  std::int64_t height{480};     ///< Pixels.
  std::int64_t frame_rate{30};  ///< Frames per second.
  std::int64_t buffer_count{4};
};

HOLOSCAN_PARAM_SCHEMA(V4l2CameraParams,
                      holoscan::param_schema("com.nvidia.isaac.V4l2CameraParams", 1),
                      holoscan::field<&V4l2CameraParams::device>("device"),
                      holoscan::field<&V4l2CameraParams::width>("width"),
                      holoscan::field<&V4l2CameraParams::height>("height"),
                      holoscan::field<&V4l2CameraParams::frame_rate>("frame_rate"),
                      holoscan::field<&V4l2CameraParams::buffer_count>("buffer_count"));
/// Captures the latest V4L2 YUYV frame and publishes a GPU RGB tensor.
class V4l2CameraOp final : public holoscan::Operator<V4l2CameraParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;
  holoscan::LifecycleStatus arm(holoscan::LifecycleContext& context) noexcept;

  holoscan::Output<holoscan::Tensor> image;

 private:
  struct MappedBuffer {
    void* data{};
    std::size_t size{};
  };

  [[nodiscard]] std::size_t captured_frame_bytes() const;
  [[nodiscard]] std::size_t published_frame_bytes() const;
  void open_device();
  void close_device();
  void capture_frames();
  bool capture_frame();  ///< Returns false on a fatal capture error; otherwise keep capturing.
  void fail_capture(std::string message);

  holoscan::NotificationSource frame_available_;
  holoscan::NotificationSender frame_sender_;

  int descriptor_{-1};
  std::vector<MappedBuffer> mapped_buffers_;
  std::thread capture_thread_;
  std::atomic<bool> stopping_{};
  bool warned_incomplete_{};
  // Capture-thread handoff to compute().
  std::mutex frame_mutex_;
  std::string capture_error_;
  std::vector<std::uint8_t> staged_frame_;
  std::int64_t staged_capture_ns_{};
  bool staged_frame_valid_{};
  std::uint8_t* upload_buffer_{};  ///< Pinned host memory.

  cudaEvent_t upload_complete_{};
  void* device_capture_{};
  struct DeviceProperties {
    int device_id{};
    int multiprocessor_count{};
    int max_threads_per_multiprocessor{};
    int max_threads_per_block{};
    std::size_t shared_memory_per_block{};
    int compute_capability_major{};
    int compute_capability_minor{};
  };
  DeviceProperties device_properties_;
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_V4L2_CAMERA_V4L2_CAMERA_OP_H
