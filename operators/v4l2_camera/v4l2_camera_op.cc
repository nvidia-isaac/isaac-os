/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/v4l2_camera/v4l2_camera_op.h"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <nppi_color_conversion.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/logger/logger.hpp>

namespace isaac::os::ops {
namespace {
constexpr std::size_t kCapturedBytesPerPixel = 2U;
constexpr std::size_t kPublishedBytesPerPixel = 3U;
constexpr DLDataType kImageDtype{kDLUInt, 8U, 1U};
template <typename T>
int retry_on_interrupt(int descriptor, unsigned long request, T* argument) {
  int result = 0;
  do {
    result = ioctl(descriptor, request, argument);
  } while (result == -1 && errno == EINTR);
  return result;
}

std::string describe_errno() { return std::string{std::strerror(errno)}; }

std::int64_t monotonic_now_ns() {
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<std::int64_t>(now.tv_sec) * 1000000000LL + now.tv_nsec;
}

}  // namespace

std::size_t V4l2CameraOp::captured_frame_bytes() const {
  return static_cast<std::size_t>(params().width) * static_cast<std::size_t>(params().height) *
         kCapturedBytesPerPixel;
}

std::size_t V4l2CameraOp::published_frame_bytes() const {
  return static_cast<std::size_t>(params().width) * static_cast<std::size_t>(params().height) *
         kPublishedBytesPerPixel;
}

void V4l2CameraOp::setup(holoscan::OperatorSpec& spec) {
  const holoscan::TensorOutputSpec published_image{
      .representation =
          {
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = kImageDtype,
              .rank = 3U,
          },
      .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = published_frame_bytes()},
      .storage = holoscan::TensorOutputStorage::kRuntimePool,
  };
  spec.output(image, "out_image").produces_tensor(published_image);
  spec.notification_source(frame_available_, "frame_available").capacity(2U);
  spec.lifecycle().stage(holoscan::LifecycleStage::kArm, &V4l2CameraOp::arm);
}

holoscan::Contract V4l2CameraOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnNotified{.event = frame_available_});
  return contract;
}

holoscan::LifecycleStatus V4l2CameraOp::arm(holoscan::LifecycleContext& context) noexcept {
  auto sender = context.notification_sender(frame_available_);
  if (!sender) {
    HOLOSCAN_LOG_ERROR("Unable to prepare for camera frame delivery: {}.", sender.error().message);
    return holoscan::LifecycleStatus::kFatalFailure;
  }
  frame_sender_ = std::move(*sender);
  return holoscan::LifecycleStatus::kOk;
}

void V4l2CameraOp::open_device() {
  const V4l2CameraParams& config = params();
  descriptor_ = open(config.device.c_str(), O_RDWR | O_NONBLOCK, 0);
  if (descriptor_ < 0) {
    throw std::runtime_error("Unable to open the camera at " + config.device + ": " +
                             describe_errno() + ".");
  }

  v4l2_format format{};
  format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  format.fmt.pix.width = static_cast<unsigned int>(config.width);
  format.fmt.pix.height = static_cast<unsigned int>(config.height);
  format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  format.fmt.pix.field = V4L2_FIELD_NONE;
  if (retry_on_interrupt(descriptor_, VIDIOC_S_FMT, &format) < 0) {
    throw std::runtime_error("The camera at " + config.device +
                             " did not accept the requested frame format: " + describe_errno() +
                             ".");
  }
  if (format.fmt.pix.width != static_cast<unsigned int>(config.width) ||
      format.fmt.pix.height != static_cast<unsigned int>(config.height) ||
      format.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kIncompatible,
        std::format("The camera at {} cannot deliver {}x{} frames in the expected colour format; "
                    "it offered {}x{}.",
                    config.device, config.width, config.height, format.fmt.pix.width,
                    format.fmt.pix.height));
  }
  if (format.fmt.pix.bytesperline != config.width * kCapturedBytesPerPixel) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kNotSupported,
        "The camera at " + config.device + " requires padded image rows, which are not supported.");
  }

  v4l2_streamparm stream_parameters{};
  stream_parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  auto& frame_interval = stream_parameters.parm.capture.timeperframe;  // codespell:ignore parm
  frame_interval.numerator = 1;
  frame_interval.denominator = static_cast<unsigned int>(config.frame_rate);
  if (retry_on_interrupt(descriptor_, VIDIOC_S_PARM, &stream_parameters) < 0) {
    HOLOSCAN_LOG_WARN("The camera at {} kept its own frame rate.", config.device);
  }

  v4l2_requestbuffers request{};
  request.count = static_cast<unsigned int>(config.buffer_count);
  request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  request.memory = V4L2_MEMORY_MMAP;
  if (retry_on_interrupt(descriptor_, VIDIOC_REQBUFS, &request) < 0 || request.count == 0U) {
    throw std::runtime_error("The camera at " + config.device +
                             " did not provide capture buffers: " + describe_errno() + ".");
  }

  mapped_buffers_.resize(request.count);
  for (unsigned int index = 0; index < request.count; ++index) {
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = index;
    if (retry_on_interrupt(descriptor_, VIDIOC_QUERYBUF, &buffer) < 0) {
      throw std::runtime_error("Unable to prepare a capture buffer for the camera at " +
                               config.device + ": " + describe_errno() + ".");
    }
    if (buffer.length < captured_frame_bytes()) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible,
                                   "A capture buffer from the camera at " + config.device +
                                       " is too small for a complete frame.");
    }
    void* mapped = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor_,
                        static_cast<off_t>(buffer.m.offset));
    if (mapped == MAP_FAILED) {
      throw std::runtime_error("Unable to map a capture buffer for the camera at " + config.device +
                               ": " + describe_errno() + ".");
    }
    mapped_buffers_[index] = MappedBuffer{.data = mapped, .size = buffer.length};
    if (retry_on_interrupt(descriptor_, VIDIOC_QBUF, &buffer) < 0) {
      throw std::runtime_error("Unable to hand a capture buffer to the camera at " + config.device +
                               ": " + describe_errno() + ".");
    }
  }

  auto buffer_type = static_cast<int>(V4L2_BUF_TYPE_VIDEO_CAPTURE);
  if (retry_on_interrupt(descriptor_, VIDIOC_STREAMON, &buffer_type) < 0) {
    throw std::runtime_error("The camera at " + config.device +
                             " did not start streaming: " + describe_errno() + ".");
  }
}

void V4l2CameraOp::close_device() {
  if (descriptor_ >= 0) {
    auto buffer_type = static_cast<int>(V4L2_BUF_TYPE_VIDEO_CAPTURE);
    retry_on_interrupt(descriptor_, VIDIOC_STREAMOFF, &buffer_type);
  }
  for (MappedBuffer& buffer : mapped_buffers_) {
    if (buffer.data != nullptr) {
      munmap(buffer.data, buffer.size);
    }
  }
  mapped_buffers_.clear();
  if (descriptor_ >= 0) {
    close(descriptor_);
    descriptor_ = -1;
  }
}

void V4l2CameraOp::start() {
  const V4l2CameraParams& config = params();
  if (config.width <= 0 || config.height <= 0 || (config.width % 2) != 0) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kInvalidArgument,
        "The camera frame size must be positive, and its width must be even because the camera "
        "pairs neighbouring pixels.");
  }
  if (config.frame_rate <= 0) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "The camera frame rate must be at least one frame per second.");
  }
  if (config.buffer_count <= 0) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "The camera needs at least one capture buffer.");
  }

  cudaDeviceProp properties{};
  if (cudaGetDevice(&device_properties_.device_id) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device_properties_.device_id) != cudaSuccess) {
    throw std::runtime_error("Unable to read the GPU properties the camera conversion needs.");
  }
  device_properties_.multiprocessor_count = properties.multiProcessorCount;
  device_properties_.max_threads_per_multiprocessor = properties.maxThreadsPerMultiProcessor;
  device_properties_.max_threads_per_block = properties.maxThreadsPerBlock;
  device_properties_.shared_memory_per_block = properties.sharedMemPerBlock;
  device_properties_.compute_capability_major = properties.major;
  device_properties_.compute_capability_minor = properties.minor;

  staged_frame_.assign(captured_frame_bytes(), 0U);
  capture_error_.clear();
  staged_frame_valid_ = false;
  staged_capture_ns_ = 0;
  warned_incomplete_ = false;

  if (cudaMallocHost(&upload_buffer_, captured_frame_bytes()) != cudaSuccess) {
    throw std::runtime_error("Unable to reserve host memory for camera frames.");
  }
  if (cudaMalloc(&device_capture_, captured_frame_bytes()) != cudaSuccess) {
    cudaFreeHost(upload_buffer_);
    upload_buffer_ = nullptr;
    throw std::runtime_error("Unable to reserve GPU memory for camera frames.");
  }

  try {
    if (cudaEventCreateWithFlags(&upload_complete_, cudaEventDisableTiming) != cudaSuccess) {
      throw std::runtime_error("Unable to track camera frame uploads to the GPU.");
    }
    open_device();
    stopping_.store(false);
    capture_thread_ = std::thread{[this] { capture_frames(); }};
  } catch (...) {
    stop();
    throw;
  }
  HOLOSCAN_LOG_INFO("Capturing {}x{} frames at {} per second from {}.", config.width, config.height,
                    config.frame_rate, config.device);
}

void V4l2CameraOp::stop() {
  stopping_.store(true);
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
  close_device();
  if (upload_complete_ != nullptr) {
    cudaEventSynchronize(upload_complete_);
    cudaEventDestroy(upload_complete_);
    upload_complete_ = nullptr;
  }
  if (device_capture_ != nullptr) {
    cudaFree(device_capture_);
    device_capture_ = nullptr;
  }
  if (upload_buffer_ != nullptr) {
    cudaFreeHost(upload_buffer_);
    upload_buffer_ = nullptr;
  }
}

void V4l2CameraOp::fail_capture(std::string message) {
  {
    const std::lock_guard lock(frame_mutex_);
    capture_error_ = "Capture stopped for the camera at " + params().device + ": " + message;
  }
  while (!stopping_.load()) {
    const auto posted = frame_sender_.post();
    if (posted || (posted.error().code != holoscan::ErrorCode::kNotReady &&
                   posted.error().code != holoscan::ErrorCode::kBackpressured)) {
      break;
    }
    // Wait for startup or notification capacity so the failure reaches compute().
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

void V4l2CameraOp::capture_frames() {
  while (!stopping_.load()) {
    if (!capture_frame()) {
      return;
    }
  }
}

bool V4l2CameraOp::capture_frame() {
  const std::size_t frame_bytes = captured_frame_bytes();
  pollfd waiter{.fd = descriptor_, .events = POLLIN, .revents = 0};
  const int ready = poll(&waiter, 1, 200);
  if (ready == 0 || (ready < 0 && errno == EINTR)) {
    return true;
  }
  if (ready < 0) {
    fail_capture("Unable to wait for a frame: " + describe_errno() + ".");
    return false;
  }
  if ((waiter.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
    fail_capture(std::format("The device reported a polling error or disconnection (events {}).",
                             waiter.revents));
    return false;
  }
  if ((waiter.revents & POLLIN) == 0) {
    return true;
  }

  v4l2_buffer buffer{};
  buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buffer.memory = V4L2_MEMORY_MMAP;
  if (retry_on_interrupt(descriptor_, VIDIOC_DQBUF, &buffer) < 0) {
    if (errno == EAGAIN) {
      return true;
    }
    fail_capture("Unable to read a frame: " + describe_errno() + ".");
    return false;
  }
  if (buffer.index >= mapped_buffers_.size()) {
    fail_capture("The device returned an unknown capture buffer.");
    return false;
  }
  const bool complete =
      (buffer.flags & V4L2_BUF_FLAG_ERROR) == 0U && buffer.bytesused >= frame_bytes;
  bool staged = false;
  if (!complete) {
    if (!warned_incomplete_) {
      warned_incomplete_ = true;
      HOLOSCAN_LOG_WARN(
          "The camera returned a frame of {} of {} bytes{}; discarding it and waiting for a "
          "complete one.",
          buffer.bytesused, frame_bytes,
          (buffer.flags & V4L2_BUF_FLAG_ERROR) != 0U ? " marked faulty" : "");
    }
  } else {
    const auto captured_ns = static_cast<std::int64_t>(buffer.timestamp.tv_sec) * 1000000000LL +
                             static_cast<std::int64_t>(buffer.timestamp.tv_usec) * 1000LL;
    {
      const std::lock_guard lock(frame_mutex_);
      std::memcpy(staged_frame_.data(), mapped_buffers_[buffer.index].data, frame_bytes);
      staged_capture_ns_ = captured_ns;
      staged_frame_valid_ = true;
      staged = true;
    }
  }

  if (retry_on_interrupt(descriptor_, VIDIOC_QBUF, &buffer) < 0) {
    fail_capture("Unable to return a frame buffer: " + describe_errno() + ".");
    return false;
  }
  if (staged) {
    static_cast<void>(frame_sender_.post());
  }
  return true;
}

holoscan::expected<void, holoscan::Error> V4l2CameraOp::compute(
    holoscan::ExecutionContext& context) {
  // Wait for the previous upload before reusing its host buffer.
  if (cudaEventSynchronize(upload_complete_) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "Unable to finish uploading a camera frame to the GPU."});
  }
  std::int64_t captured_ns = 0;
  {
    const std::lock_guard lock(frame_mutex_);
    if (!capture_error_.empty()) {
      HOLOSCAN_LOG_ERROR("{}", capture_error_);
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, capture_error_});
    }
    if (!staged_frame_valid_) {
      return {};
    }
    std::memcpy(upload_buffer_, staged_frame_.data(), staged_frame_.size());
    captured_ns = staged_capture_ns_;
    staged_frame_valid_ = false;
  }

  auto now = context.clock().now();
  if (!now) {
    return holoscan::make_unexpected(std::move(now).error());
  }
  // Translate the monotonic camera timestamp onto the graph clock.
  const std::int64_t age_ns = std::max<std::int64_t>(0, monotonic_now_ns() - captured_ns);
  holoscan::TimePoint capture_time = *now;
  capture_time.timestamp_ns = std::max<std::int64_t>(0, now->timestamp_ns - age_ns);

  const auto width = static_cast<int>(params().width);
  const auto height = static_cast<int>(params().height);
  const std::array<std::int64_t, 3U> shape{params().height, params().width,
                                           static_cast<std::int64_t>(kPublishedBytesPerPixel)};

  auto loan =
      image.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kImageDtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (writer->byte_size() != published_frame_bytes()) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        "The reserved image does not match the camera's frame size."});
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (cudaMemcpyAsync(device_capture_, upload_buffer_, captured_frame_bytes(),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure,
                                                     "Unable to move a camera frame to the GPU."});
  }
  if (cudaEventRecord(upload_complete_, stream) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "Unable to track a camera frame upload to the GPU."});
  }
  unsigned int stream_flags = 0U;
  if (cudaStreamGetFlags(stream, &stream_flags) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "Unable to prepare the camera colour conversion."});
  }
  const NppStreamContext npp_context{
      .hStream = stream,
      .nCudaDeviceId = device_properties_.device_id,
      .nMultiProcessorCount = device_properties_.multiprocessor_count,
      .nMaxThreadsPerMultiProcessor = device_properties_.max_threads_per_multiprocessor,
      .nMaxThreadsPerBlock = device_properties_.max_threads_per_block,
      .nSharedMemPerBlock = device_properties_.shared_memory_per_block,
      .nCudaDevAttrComputeCapabilityMajor = device_properties_.compute_capability_major,
      .nCudaDevAttrComputeCapabilityMinor = device_properties_.compute_capability_minor,
      .nStreamFlags = stream_flags,
      .nReserved0 = 0,
  };
  const NppiSize region{.width = width, .height = height};
  if (nppiYUV422ToRGB_8u_C2C3R_Ctx(
          static_cast<const Npp8u*>(device_capture_),
          width * static_cast<int>(kCapturedBytesPerPixel), static_cast<Npp8u*>(writer->data()),
          width * static_cast<int>(kPublishedBytesPerPixel), region, npp_context) != NPP_SUCCESS) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure, "Unable to convert a camera frame to colour."});
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return image.emit(std::move(*loan), holoscan::EmitOptions{.capture_time = capture_time});
}

}  // namespace isaac::os::ops
