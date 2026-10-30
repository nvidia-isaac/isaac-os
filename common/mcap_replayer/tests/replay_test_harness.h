/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/temporal_contract.hpp"
#include "holoscan/sensor_io/image_metadata.hpp"
#include "holoscan/time/realtime_clock.hpp"

#include "common/mcap_replayer/mcap_replayer.h"

namespace isaac::os::mcap_replayer::test_support {

// Copies a tensor's bytes out. Replay reconstructs tensors in host memory, so this only ever reads
// ordinary host memory.
inline std::vector<std::uint8_t> tensor_bytes_of(const holoscan::Tensor& tensor) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes()));
  std::memcpy(bytes.data(), tensor.data(), bytes.size());
  return bytes;
}

// What a sink saw of one replayed frame.
struct ReceivedImage {
  std::int32_t width = 0;
  std::int32_t height = 0;
  holoscan::schema::ImageEncoding encoding = holoscan::schema::ImageEncoding_CUSTOM;
  bool has_tensor = false;
  std::vector<std::int64_t> shape;
  DLDeviceType device_type = kDLCPU;
  std::vector<std::uint8_t> bytes;
};

// Static, not an instance member: an operator built through Graph::op<>() only ever receives
// copy-constructed arguments, so it cannot be handed a reference to live test state.
class ReceivedLog {
 public:
  static void reset() {
    const std::scoped_lock lock(mutex_);
    images_.clear();
  }

  static void record(ReceivedImage image) {
    const std::scoped_lock lock(mutex_);
    images_.push_back(std::move(image));
  }

  static std::vector<ReceivedImage> wait_for(std::size_t wanted,
                                             std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      {
        const std::scoped_lock lock(mutex_);
        if (images_.size() >= wanted) {
          return images_;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const std::scoped_lock lock(mutex_);
    return images_;
  }

 private:
  static inline std::mutex mutex_;
  static inline std::vector<ReceivedImage> images_;
};

class ImageSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(256); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{in});
    return contract;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = in.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    ReceivedImage image;
    image.width = received->data.width;
    image.height = received->data.height;
    image.encoding = received->data.encoding;
    image.has_tensor = received->data.data != nullptr;
    if (image.has_tensor) {
      const holoscan::Tensor& tensor = *received->data.data;
      // Taken once: two calls each return their own view, and iterators from different calls
      // are not guaranteed to bound the same range.
      const auto shape = tensor.shape();
      image.shape.assign(shape.begin(), shape.end());
      image.device_type = tensor.device().device_type;
      image.bytes = tensor_bytes_of(tensor);
    }
    ReceivedLog::record(std::move(image));
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> in;
};

// A graph that owns the replayer its compose() builds, so a test can wire any topics to any sinks.
class ReplayerTestGraph final : public holoscan::Graph {
 public:
  explicit ReplayerTestGraph(
      std::function<std::unique_ptr<McapReplayer>(holoscan::Graph&)> build_replayer)
      : Graph("replay_test"), build_replayer_(std::move(build_replayer)) {}

  [[nodiscard]] McapReplayer& replayer() { return *replayer_; }

 private:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    replayer_ = build_replayer_(*this);
  }

  std::function<std::unique_ptr<McapReplayer>(holoscan::Graph&)> build_replayer_;
  std::unique_ptr<McapReplayer> replayer_;
};

}  // namespace isaac::os::mcap_replayer::test_support
