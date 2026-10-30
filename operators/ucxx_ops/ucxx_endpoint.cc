/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/ucxx_ops/ucxx_endpoint.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <format>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <ucxx/exception.h>
#include <ucxx/utils/ucx.h>

namespace isaac::os::ops {
namespace {

int validate_port(int port) {
  if (port < 1 || port > 65535) {
    throw std::invalid_argument(std::format("UCX port must be between 1 and 65535; got {}.", port));
  }
  return port;
}

}  // namespace

struct UcxxEndpoint::State : public std::enable_shared_from_this<State> {
  std::string hostname;
  int port{};
  std::shared_ptr<ucxx::Context> context;
  std::shared_ptr<ucxx::Worker> worker;
  std::shared_ptr<ucxx::Endpoint> endpoint;
  bool endpoint_connected{false};
  bool connection_established{false};
  std::vector<std::weak_ptr<ConnectCallback>> connect_callbacks;

  std::mutex connect_mutex;
  mutable std::mutex endpoint_mutex;

  State(std::string h, int p) : hostname(std::move(h)), port(p) {}

  ~State() {
    std::shared_ptr<ucxx::Endpoint> ep;
    bool close_endpoint = false;
    {
      std::lock_guard<std::mutex> lock(endpoint_mutex);
      ep = endpoint;
      close_endpoint = endpoint_connected;
      endpoint_connected = false;
      endpoint.reset();
    }
    if (close_endpoint && ep) {
      ep->closeBlocking(/*period_ms=*/1000, /*maxAttempts=*/5);
    }
    if (worker) worker->stopProgressThread();
  }

  void on_endpoint_closed(ucs_status_t status) {
    {
      std::lock_guard<std::mutex> lock(endpoint_mutex);
      endpoint_connected = false;
    }
    HOLOSCAN_LOG_INFO("UCX connection closed.");
    if (status != UCS_OK) {
      HOLOSCAN_LOG_ERROR("UCX connection closed with status: {}", ucs_status_string(status));
    }
  }

  void set_endpoint_close_callback(const std::shared_ptr<ucxx::Endpoint>& ep) {
    // UCXX 0.46+ throws when the callback arg is non-null. Pass an unused
    // sentinel so callers can catch and avoid publishing an already-closing
    // endpoint.
    const auto callback_registration_sentinel = std::make_shared<int>(0);
    ep->setCloseCallback(
        [weak_this = weak_from_this()](ucs_status_t status, std::shared_ptr<void>) {
          if (auto shared = weak_this.lock()) {
            shared->on_endpoint_closed(status);
          }
        },
        callback_registration_sentinel);
  }

  std::shared_ptr<ucxx::Endpoint> wait_for_endpoint(UcxxConnectOptions options,
                                                    holoscan::CancellationView cancellation) {
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::milliseconds;
    const auto started = Clock::now();
    const auto timed_out = [options, started] {
      return options.timeout > Milliseconds::zero() && Clock::now() - started >= options.timeout;
    };
    HOLOSCAN_LOG_INFO("Waiting for the UCX peer at {}:{}.", hostname, port);
    std::shared_ptr<ucxx::Endpoint> ep;
    std::string last_error;
    while (!ep && !cancellation.stop_requested() && !timed_out()) {
      try {
        ep = worker->createEndpointFromHostname(hostname, port, true);
        // Endpoint creation is asynchronous. Flush waits for UCX wireup without a payload.
        auto wireup = ep->flush();
        while (!wireup->isCompleted() && !cancellation.stop_requested() && !timed_out()) {
          std::this_thread::sleep_for(Milliseconds{1});
        }
        if (cancellation.stop_requested() || timed_out()) {
          ep->closeBlocking();
          break;
        }
        ucxx::utils::ucsErrorThrow(wireup->getStatus());
        ep->raiseOnError();
      } catch (const ucxx::Error& error) {
        const bool retryable = dynamic_cast<const ucxx::UnreachableError*>(&error) ||
                               dynamic_cast<const ucxx::RejectedError*>(&error) ||
                               dynamic_cast<const ucxx::NotConnectedError*>(&error) ||
                               dynamic_cast<const ucxx::TimedOutError*>(&error) ||
                               dynamic_cast<const ucxx::ConnectionResetError*>(&error) ||
                               dynamic_cast<const ucxx::EndpointTimeoutError*>(&error);
        if (!retryable) throw;
        last_error = error.what();
        ep.reset();
        const auto waiting = Clock::now();
        while (!cancellation.stop_requested() && !timed_out()) {
          const auto remaining = options.retry_interval - (Clock::now() - waiting);
          if (remaining <= Clock::duration::zero()) break;
          const auto sleep_duration = std::chrono::duration_cast<Milliseconds>(remaining);
          std::this_thread::sleep_for(
              std::max(std::min(sleep_duration, Milliseconds{10}), Milliseconds{1}));
        }
      }
    }
    if (cancellation.stop_requested()) return {};
    if (timed_out()) {
      throw ucxx::TimedOutError(std::format("Timed out waiting for the UCX peer at {}:{}. {}",
                                            hostname, port, last_error));
    }
    return ep;
  }

  // Called from the constructor when connect_on_construction is true, or
  // explicitly after graph startup for a deferred initial connection.
  // There is no auto-reconnect: if the peer drops mid-run,
  // on_endpoint_closed() clears endpoint_connected and subsequent endpoint()
  // calls return nullptr. UcxxReceiverOp::compute and UcxxSenderOp::compute
  // surface that as a fatal error rather than silently no-oping.
  void connect_impl(UcxxConnectOptions options = {}, holoscan::CancellationView cancellation = {}) {
    using Milliseconds = std::chrono::milliseconds;
    if (options.timeout < Milliseconds::zero() || options.retry_interval <= Milliseconds::zero()) {
      throw std::invalid_argument(
          "Connection timeout must be nonnegative and retry interval positive.");
    }
    if (!worker) {
      throw std::runtime_error("Unable to connect to the UCX peer: worker is not initialized.");
    }
    // Serialize initial connection creation. Receivers post requests once;
    // reconnecting would require resetting every sender and receiver together.
    std::unique_lock connect_lock(connect_mutex);
    {
      std::lock_guard lock(endpoint_mutex);
      if (endpoint_connected && endpoint && endpoint->isAlive()) {
        return;
      }
      if (connection_established) {
        throw std::runtime_error(
            "The UCX connection failed or closed. Restart the application to reconnect.");
      }
    }
    auto ep = wait_for_endpoint(options, cancellation);
    if (!ep) return;
    set_endpoint_close_callback(ep);
    std::unique_lock lock(endpoint_mutex);
    if (!ep->isAlive()) {
      // Fail-fast: the ctor promises a live handle on return so callers do
      // not have to double-check endpoint() before wiring ops. A dead-on-
      // arrival endpoint here means the peer refused wireup, and there is
      // no reconnect path -- surface it now instead of letting the first
      // UcxxReceiverOp::on_start / UcxxSenderOp::compute call terminate
      // the session with a null-endpoint error.
      throw std::runtime_error("Unable to connect to the UCX peer at " + hostname + ":" +
                               std::to_string(port) + ": endpoint is not alive.");
    }
    endpoint = ep;
    endpoint_connected = true;
    connection_established = true;
    HOLOSCAN_LOG_INFO("Connected to the UCX peer at {}:{}.", hostname, port);
    std::vector<std::shared_ptr<ConnectCallback>> callbacks;
    auto output = connect_callbacks.begin();
    for (auto input = connect_callbacks.begin(); input != connect_callbacks.end(); ++input) {
      if (auto callback = input->lock()) {
        callbacks.push_back(std::move(callback));
        *output++ = *input;
      }
    }
    connect_callbacks.erase(output, connect_callbacks.end());
    lock.unlock();
    connect_lock.unlock();
    for (const auto& callback : callbacks) {
      (*callback)(ep);
    }
  }

  void notify_when_connected(const std::shared_ptr<ConnectCallback>& callback) {
    std::shared_ptr<ucxx::Endpoint> live_endpoint;
    {
      std::lock_guard<std::mutex> lock(endpoint_mutex);
      connect_callbacks.emplace_back(callback);
      if (endpoint_connected && endpoint && endpoint->isAlive()) {
        live_endpoint = endpoint;
      }
    }
    if (live_endpoint) {
      (*callback)(live_endpoint);
    }
  }
};

UcxxEndpoint::UcxxEndpoint(std::string hostname, int port, bool connect_on_construction)
    : state_(std::make_shared<State>(std::move(hostname), validate_port(port))) {
  // UCX 1.19.0 default UCX_CM_USE_ALL_DEVICES=y makes the CM path pack an
  // extra rsc_index byte after each iface_attr on the wire, but the peer
  // unpacker only skips that byte when its own unpack_flags carry
  // UCP_ADDRESS_PACK_FLAG_TL_RSC_IDX. Sim's isaacsim.ucx.core does not, so
  // its unpack offsets drift by one byte per iface and the wireup handshake
  // is rejected with "invalid bandwidth 0.00". Pin the client to =n so the
  // extra byte is never packed. setenv overwrite=0 preserves any caller
  // override.
  ::setenv("UCX_CM_USE_ALL_DEVICES", "n", /*overwrite=*/0);
  state_->context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
  state_->worker = state_->context->createWorker();
  state_->worker->startProgressThread(/*pollingMode=*/false);
  if (connect_on_construction) {
    state_->connect_impl();
  }
}

void UcxxEndpoint::connect(UcxxConnectOptions options, holoscan::CancellationView cancellation) {
  if (!state_) {
    throw std::runtime_error("Unable to connect an unbound UCX endpoint.");
  }
  state_->connect_impl(options, cancellation);
}

void UcxxEndpoint::notify_when_connected(const std::shared_ptr<ConnectCallback>& callback) {
  if (state_ && callback) state_->notify_when_connected(callback);
}

std::shared_ptr<ucxx::Endpoint> UcxxEndpoint::endpoint() const {
  if (!state_) return nullptr;
  std::lock_guard<std::mutex> lock(state_->endpoint_mutex);
  if (!state_->endpoint_connected || !state_->endpoint || !state_->endpoint->isAlive()) {
    return nullptr;
  }
  return state_->endpoint;
}

std::shared_ptr<ucxx::Worker> UcxxEndpoint::worker() const {
  return state_ ? state_->worker : nullptr;
}

const std::string& UcxxEndpoint::hostname() const {
  static const std::string kEmpty;
  return state_ ? state_->hostname : kEmpty;
}

int UcxxEndpoint::port() const { return state_ ? state_->port : 0; }

}  // namespace isaac::os::ops
