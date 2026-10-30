/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <exception>
#include <thread>
#include <utility>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "operators/ucxx_ops/ucxx_endpoint.h"

namespace isaac::os::ops {

/// Wait for the initial connection after the global startup barrier, without a running clock.
class UcxxConnectOp final : public holoscan::Operator<> {
 public:
  explicit UcxxConnectOp(UcxxEndpoint endpoint, UcxxConnectOptions options = {})
      : endpoint_(std::move(endpoint)), options_(options) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.notification_source(ready_, "connect").capacity(1U).sender_reference_capacity(1U);
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kArm, &UcxxConnectOp::arm)
        .stage(holoscan::LifecycleStage::kStart, &UcxxConnectOp::start_connection)
        .stage(holoscan::LifecycleStage::kStop, &UcxxConnectOp::stop_connection);
  }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnNotified{.event = ready_});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    try {
      endpoint_.connect(options_, context.cancellation());
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, error.what()});
    }
    return {};
  }

 private:
  holoscan::LifecycleStatus arm(holoscan::LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(ready_);
    if (!sender) return holoscan::LifecycleStatus::kFatalFailure;
    sender_ = std::move(*sender);
    return holoscan::LifecycleStatus::kOk;
  }
  holoscan::LifecycleStatus start_connection(holoscan::LifecycleContext&) noexcept {
    try {
      // Notification admission opens only after every operator has completed Start.
      startup_ = std::jthread([this](std::stop_token stop) {
        while (!stop.stop_requested()) {
          const auto posted = sender_.post();
          if (posted) return;
          if (posted.error().code != holoscan::ErrorCode::kNotReady &&
              posted.error().code != holoscan::ErrorCode::kBackpressured)
            return;
          std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
      });
    } catch (...) {
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }
  holoscan::LifecycleStatus stop_connection(holoscan::LifecycleContext&) noexcept {
    startup_.request_stop();
    try {
      if (startup_.joinable()) startup_.join();
    } catch (...) {
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }
  UcxxEndpoint endpoint_;
  UcxxConnectOptions options_;
  holoscan::NotificationSource ready_;
  holoscan::NotificationSender sender_;
  std::jthread startup_;
};

}  // namespace isaac::os::ops
