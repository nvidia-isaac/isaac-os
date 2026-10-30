/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "holoscan/channel/runtime.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/logger/logger.hpp"
#include "holoscan/schema/binary_schema.hpp"
#include "holoscan/schema/schema_package.hpp"

#include "common/mcap_logger/failure_log_schedule.h"
#include "common/mcap_logger/mcap_sink.h"

namespace isaac::os::mcap_logger {

// The deadline for a poller's next wait for samples. This thread has nothing else to do, so the
// wait has none once a publisher is connected. Until then it wakes every millisecond: a producer's
// first emit can be refused while this subscriber is still connecting, and waking refreshes that
// connection, as polling did.
template <typename Subscriber>
[[nodiscard]] std::chrono::steady_clock::time_point NextWaitDeadline(const Subscriber& subscriber) {
  return subscriber.has_publishers()
             ? std::chrono::steady_clock::time_point::max()
             : std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
}

// Type-erased base so HsdkMcapLogger can hold every TopicPoller<T> (whatever T is) in one vector
// and destroy-and-join each polymorphically in shutdown(). It also owns what every poller does the
// same way: the topic's name, reporting a sample the topic's queue dropped, and keeping one failed
// write from ending the topic.
class PollerBase {
 public:
  virtual ~PollerBase() = default;

 protected:
  PollerBase(std::string unique_id, std::uint32_t queue_depth)
      : unique_id_(std::move(unique_id)), queue_depth_(queue_depth) {}

  [[nodiscard]] const std::string& unique_id() const noexcept { return unique_id_; }
  [[nodiscard]] std::uint32_t queue_depth() const noexcept { return queue_depth_; }

  // Call with each received sample's publication sequence. The topic's queue drops its oldest
  // sample when it is full, which leaves a gap here; the first gap is reported, then every tenth.
  // Zero means the publisher did not stamp a sequence, and one that does not advance is a restart
  // or reorder, not a drop.
  //
  // A publisher's sequence starts at 1, so a first sample above 1 means earlier ones were dropped
  // before it was received. That holds when the subscriber is attached before the producer first
  // publishes, which is the order start() then run_async() gives; attaching to a producer that is
  // already running would report the samples published before it as dropped, once. It also holds
  // when the topic carries no clock-health lane: a health report takes a number from the same
  // sequence and would read as a drop.
  void note_sequence(std::uint64_t sequence) {
    if (sequence == 0 || sequence <= last_sequence_) {
      return;
    }
    const std::uint64_t missed = sequence - last_sequence_ - 1;
    last_sequence_ = sequence;
    if (missed == 0) {
      return;
    }
    dropped_ += missed;
    if (IsFailureReportDue(++drop_events_)) {
      HOLOSCAN_LOG_WARN(
          "The MCAP logger's topic '{}' dropped samples because writing fell behind: {} so far, "
          "oldest first, with {} samples held. If this is a burst rather than a sustained "
          "overload, increase the topic's queue depth.",
          unique_id_, dropped_, queue_depth_);
    }
  }

  // Runs one sample's write, so an exception from it costs that sample and not the topic.
  template <typename Write>
  void run_guarded(Write&& write) {
    try {
      std::forward<Write>(write)();
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_WARN("A logging task for topic '{}' failed: {}", unique_id_, e.what());
    } catch (...) {
      HOLOSCAN_LOG_WARN("A logging task for topic '{}' failed with an unknown exception.",
                        unique_id_);
    }
  }

  // Reports why a poller's wait for samples ended. A stop request, or the producer having closed
  // its end because the graph is shutting down (kClosed), is how a poller normally ends and is
  // silent. Anything else means the topic stopped being recorded for a reason nobody asked for.
  void report_receive_ended(const holoscan::Error& error) const {
    if (error.code == holoscan::ErrorCode::kInterrupted ||
        error.code == holoscan::ErrorCode::kClosed) {
      return;
    }
    HOLOSCAN_LOG_ERROR(
        "The MCAP logger has stopped recording topic '{}' and will write no further samples "
        "for it: {}.",
        unique_id_, holoscan::runtime_error_message(error));
  }

 private:
  std::string unique_id_;
  std::uint32_t queue_depth_;
  std::uint64_t last_sequence_ = 0;
  std::uint64_t dropped_ = 0;
  std::uint64_t drop_events_ = 0;
};

// Owns one topic's ChannelSubscriber<T> and blocks on it via wait_and_receive_sample() on a
// dedicated thread -- catalog/export_topic endpoints carry ChannelOptions::enable_event_wake ==
// true unconditionally (see endpoint_catalog.cpp's materialize_normalized_ownership()), so the
// SDK's kernel-wake wait set is always available here; there is no poll-interval fallback to
// configure. The same thread serializes each received sample via T's SchemaPackageDescriptor and
// writes it to the shared McapSink before receiving the next, so a topic's samples are written in
// the order they arrived. stop() joins it, and it first writes what the queue still holds.
template <typename T>
class TopicPoller : public PollerBase {
 public:
  TopicPoller(std::string unique_id, holoscan::ChannelSubscriber<T> subscriber,
              std::shared_ptr<McapSink> sink, std::uint32_t queue_depth)
      : PollerBase(std::move(unique_id), queue_depth),
        subscriber_(std::move(subscriber)),
        sink_(std::move(sink)),
        thread_(&TopicPoller::poll_loop, this) {}

  ~TopicPoller() { stop(); }

  TopicPoller(const TopicPoller&) = delete;
  TopicPoller& operator=(const TopicPoller&) = delete;

  void stop() {
    stop_source_.request_stop();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  void poll_loop() {
    const holoscan::SchemaPackageDescriptor<T>& descriptor = holoscan::schema_package<T>();
    const std::stop_token token = stop_source_.get_token();

    while (true) {
      // wait_and_receive_sample() already retries internally against a kNotReady race, so every
      // error it returns here is terminal for this subscriber, other than the timeout below: a stop
      // request, or e.g. kClosed (producer end of stream, the normal way a graph shutdown
      // arrives), which latches and would otherwise return immediately forever.
      auto sample =
          subscriber_.template wait_and_receive_sample<T>(NextWaitDeadline(subscriber_), token);
      if (!sample && sample.error().code == holoscan::ErrorCode::kTimeout) {
        continue;  // No publisher connected yet.
      }
      if (!sample) {
        report_receive_ended(sample.error());
        break;  // Stop requested, or the subscriber is permanently unusable.
      }
      write(descriptor, *sample);
    }

    // Nothing more will be waited for: write what the queue still holds, without blocking. The
    // queue holds at most queue_depth samples, so that many receives drain it, and a producer
    // still publishing cannot keep this thread from finishing.
    for (std::uint32_t drained = 0; drained < queue_depth(); ++drained) {
      auto sample = subscriber_.template receive_sample<T>();
      if (!sample) {
        break;
      }
      write(descriptor, *sample);
    }
  }

  // Serializes one received sample and writes it. log_time drives replay pacing;
  // publish_timestamp_ns is the runtime's own graph-emit time, stable across every subscriber. A
  // message's own capture/acquisition time, if it has one, lives in its payload, not here.
  template <typename Sample>
  void write(const holoscan::SchemaPackageDescriptor<T>& descriptor, Sample& sample) {
    note_sequence(sample.metadata.sequence);
    const std::uint64_t log_time_ns =
        static_cast<std::uint64_t>(sample.metadata.publish_timestamp_ns);
    run_guarded([&] {
      std::vector<std::byte> buffer(descriptor.max_serialized_size());
      auto encoded = descriptor.encode(sample.data, buffer);
      if (!encoded) {
        const std::uint64_t count = ++encode_failure_log_count_;
        if (IsFailureReportDue(count)) {
          HOLOSCAN_LOG_WARN(
              "Unable to serialize a sample from the MCAP logger's topic '{}': {} ({} such "
              "failures so far.)",
              unique_id(), holoscan::runtime_error_message(encoded.error()), count);
        }
        return;
      }
      const auto& identity = descriptor.identity();
      sink_->write_flatbuffer_schema(unique_id(), identity.canonical_name,
                                     holoscan::binary_schema<T>(), log_time_ns,
                                     std::span<const std::byte>(buffer.data(), *encoded));
    });
  }

  holoscan::ChannelSubscriber<T> subscriber_;
  std::shared_ptr<McapSink> sink_;
  std::stop_source stop_source_;
  std::uint64_t encode_failure_log_count_ = 0;  // Only this topic's poll thread touches it.
  std::thread thread_;
};

}  // namespace isaac::os::mcap_logger
