/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_UCXX_OPS_UCXX_ENDPOINT_H
#define ISAAC_OS_OPERATORS_UCXX_OPS_UCXX_ENDPOINT_H

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "ucxx/api.h"
#include <holoscan/core/lifecycle.hpp>

namespace isaac::os::ops {

struct UcxxConnectOptions {
  std::chrono::milliseconds timeout{0};  ///< Zero waits indefinitely for the initial connection.
  std::chrono::milliseconds retry_interval{1000};
};

/// Value handle whose copies share one UCX client endpoint.
class UcxxEndpoint {
 public:
  using ConnectCallback = std::function<void(const std::shared_ptr<ucxx::Endpoint>&)>;
  /// Construct an unbound handle. `valid()` returns false.
  UcxxEndpoint() = default;

  /// Construct a handle to a freshly-created endpoint. Ops that copy this
  /// handle share the same underlying connection. The port must be between 1
  /// and 65535.
  UcxxEndpoint(std::string hostname, int port, bool connect_on_construction = true);

  // Copy and move operations share the internal state.
  UcxxEndpoint(const UcxxEndpoint&) = default;
  UcxxEndpoint& operator=(const UcxxEndpoint&) = default;
  UcxxEndpoint(UcxxEndpoint&&) noexcept = default;
  UcxxEndpoint& operator=(UcxxEndpoint&&) noexcept = default;
  ~UcxxEndpoint() = default;

  /// True once bound to internal state (i.e., not default-constructed).
  [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }

  /// Wait for the initial connection, until canceled or the timeout expires.
  /// Repeated calls on a live connection are no-ops. After an established peer
  /// disconnects, reconnection requires a new endpoint and a restarted graph.
  void connect(UcxxConnectOptions options = {}, holoscan::CancellationView cancellation = {});

  /// Register a receiver callback for the initial connection (or call it now if live).
  /// The endpoint retains only a weak reference; the receiver owns its callback.
  void notify_when_connected(const std::shared_ptr<ConnectCallback>& callback);

  /// The active UCX endpoint, or `nullptr` before the deferred initial
  /// connection, if the peer has disconnected, or if the endpoint is otherwise
  /// no longer alive. There is no auto-reconnect after an established peer drops.
  /// Consumers -- see `UcxxSenderOp::compute` and `UcxxReceiverOp::compute` --
  /// surface a null return as a fatal error rather than silently no-oping.
  /// Deferred receivers must first be activated by the connection callback;
  /// endpoint() returning nullptr alone does not distinguish pending startup
  /// from failure and does not wake a receiver.
  std::shared_ptr<ucxx::Endpoint> endpoint() const;

  /// Underlying UCX worker; used by receivers to post tag receives and by
  /// tests to progress the worker manually.
  std::shared_ptr<ucxx::Worker> worker() const;

  const std::string& hostname() const;
  int port() const;

 private:
  struct State;
  std::shared_ptr<State> state_;
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_UCXX_OPS_UCXX_ENDPOINT_H
