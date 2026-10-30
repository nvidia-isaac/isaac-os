/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "holoscan/channel/endpoint_catalog.hpp"
#include "holoscan/channel/runtime.hpp"
#include "holoscan/core/connection_options.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/port.hpp"
#include "holoscan/core/schema_identity.hpp"
#include "holoscan/schema/binary_schema.hpp"
#include "holoscan/schema/schema_package.hpp"
#include "holoscan/sensor_io/image_metadata.hpp"

#include "common/mcap_replayer/compressed_video_builder.h"
#include "common/mcap_replayer/decoder.h"
#include "common/mcap_replayer/mcap_file.h"
#include "common/mcap_replayer/replay_context.h"
#include "common/mcap_replayer/replay_stats.h"
#include "common/mcap_replayer/typed_port.h"

namespace isaac::os::mcap_replayer {

// One topic to replay, named together with the type its recorded messages decode to. Naming the
// type lets the library replay any message type without knowing any of them in advance.
template <typename T>
struct Topic {
  std::string_view name;
};

// A tensor-schema topic (holoscan::schema::ImageT, for example) also carries the contract its
// imported edge needs: the bounds, memory kind and transfer policy of each tensor field. The caller
// supplies it, as for mcap_logger's attach(), because what is right for a type depends on the
// frames it will carry; see image_schema.h's ImageTensorFieldContract() for ImageT. There is no
// constructor without one, so omitting it does not compile.
template <typename T>
  requires holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>
struct Topic<T> {
  Topic(std::string_view topic_name, holoscan::DescriptorTensorTopicContract<T> topic_contract)
      : name(topic_name), contract(std::move(topic_contract)) {}

  std::string_view name;
  holoscan::DescriptorTensorTopicContract<T> contract;
};

// How a recording is played back.
struct ReplayOptions {
  // Recorded seconds per wall-clock second: 2.0 replays twice as fast, 0.5 half as fast. Must be
  // greater than zero.
  double playback_speed = 1.0;

  // Whether the recording starts over at its end. When false it plays once and then stops
  // producing.
  bool loop = false;

  // How many decoded messages may wait ahead of the playback cursor, absorbing decode stalls.
  // The default is a few seconds of cushion for a 100 Hz stream. Must be at least one.
  std::size_t buffer_messages = 256;

  // How long reading the recording may go without reading a single new message while the run is
  // starting before the run gives up. Replay cannot keep its recorded timing behind a reader that
  // is not making progress, so it logs an error and sends nothing. Slow reading that keeps
  // delivering messages is never a stall. Must be greater than zero.
  std::chrono::milliseconds start_stall_timeout = std::chrono::seconds(5);

  // Prefix of every replayed topic's external Graph endpoint, "<operator_name>/<topic>". A graph
  // has one flat endpoint namespace for imports and exports alike, so a bare topic name would
  // collide with e.g. an mcap_logger recording the same topic in the same graph. Two replays in
  // one graph need distinct names for the same reason.
  std::string operator_name = "mcap_replayer";

  // The CUDA device that decodes any topic the logger recorded as encoded video. Ignored for a
  // recording with none. Construction fails if this device cannot decode.
  std::uint32_t video_decoder_cuda_device = 0;
};

// Replays a recording into a live graph at the pacing it was recorded with.
//
// Construct it from inside the consumer graph's compose(), naming every topic to replay with the
// type it is wanted as, then wire each one to its consumer:
//
//   McapReplayer replayer("run.mcap", Topic<isaac::PoseCpuT>{"pose"});
//   // later, in the graph's compose():
//   replayer.attach<isaac::PoseCpuT>(*this, "pose", my_op->in_pose);
//
// After holoscan::compile() and holoscan::run_async(), start() begins the run; stop(), or
// destroying this object, ends it.
//
// Reading and decoding run on a prefetch thread and timing on a pacing thread, so a brief decode
// stall is absorbed by the buffer between them instead of pushing later messages late.
class McapReplayer {
 public:
  // Opens path and prepares every named topic: checks the recording holds it as the type named
  // and builds the message builder and port its messages are published on. Needs no Graph; a topic
  // joins the graph when attach() wires it to a consumer.
  //
  // Throws std::invalid_argument if the options cannot describe a replay, if a named topic is
  // absent from the recording, or if a topic's recorded schema does not match the type named for
  // it, if a topic recorded as encoded video has no metadata channel or one of another type, or if
  // video_decoder_cuda_device cannot decode a topic recorded as encoded video;
  // std::runtime_error if the recording cannot be opened.
  template <typename... Ts>
  explicit McapReplayer(const std::filesystem::path& path, Topic<Ts>... topics)
      : McapReplayer(path, ReplayOptions{}, std::move(topics)...) {}

  template <typename... Ts>
  McapReplayer(const std::filesystem::path& path, ReplayOptions options, Topic<Ts>... topics)
      : McapReplayer(path, options) {
    (add_topic<Ts>(std::move(topics)), ...);
  }

  // Ends any run in progress; see stop().
  ~McapReplayer();

  McapReplayer(const McapReplayer&) = delete;
  McapReplayer& operator=(const McapReplayer&) = delete;
  McapReplayer(McapReplayer&&) = delete;
  McapReplayer& operator=(McapReplayer&&) = delete;

  // Wires topic into graph, delivered to target: imports the topic on its first attach(), so a
  // topic nobody attach()es never joins the graph and is not read from the recording. Call this
  // from graph's own compose(), as Graph::import_topic() requires. One replayed topic feeds one
  // consumer: holoscan::compile() rejects a second attach() of the same topic, since an imported
  // topic cannot fan out to two edges.
  //
  // The edge always uses kDropOldest, and there is no option to change it. An imported topic's
  // publisher follows the consuming edge's queue policy, and a blocking edge would hold the pacing
  // thread until the consumer took each sample, letting a slow consumer drag the schedule: the
  // replay would no longer keep the recording's timing. kDropOldest overwrites the oldest unread
  // sample instead, so pacing never waits. The cost is that the overwrite is silent: a message
  // counted as emitted can be replaced before the consumer reads it, so stats() reports what was
  // published, not what was consumed. A consumer that cannot keep up has to be made faster or the
  // playback speed lowered.
  //
  // Throws std::invalid_argument if topic is not being replayed, or if T is not the type it was
  // named with.
  template <typename T>
  void attach(holoscan::Graph& graph, std::string_view topic, const holoscan::Input<T>& target) {
    constexpr holoscan::ConnectionOptions kOptions{.queue_policy =
                                                       holoscan::QueuePolicy::kDropOldest};
    const std::optional<mcap::ChannelId> channel = file_->channel_of(topic);
    const auto found = channel ? context_->topics.find(*channel) : context_->topics.end();
    if (found == context_->topics.end()) {
      throw std::invalid_argument(
          "Topic '" + std::string(topic) +
          "' is not being replayed. Topics being replayed: " + replayed_topics() + ".");
    }
    ReplayTopic& replay_topic = found->second;
    // The port kind is chosen at compile time (see add_topic<T>()): naming TypedPort<T> for a
    // tensor-schema T would instantiate a type that cannot exist, even in a branch never taken.
    if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>) {
      if (auto* const tensor_typed = dynamic_cast<TypedTensorPort<T>*>(replay_topic.port.get())) {
        tensor_typed->import_into(graph, endpoint_name(topic));
        graph.add_flow(tensor_typed->output(), target, kOptions);
        replay_topic.stats_id = context_->stats.register_topic(replay_topic.name);
        replay_topic.attached = true;
        return;
      }
    } else {
      if (auto* const typed = dynamic_cast<TypedPort<T>*>(replay_topic.port.get())) {
        typed->import_into(graph, endpoint_name(topic));
        graph.add_flow(typed->output(), target, kOptions);
        replay_topic.stats_id = context_->stats.register_topic(replay_topic.name);
        replay_topic.attached = true;
        return;
      }
    }
    throw std::invalid_argument(
        "Topic '" + std::string(topic) +
        "' is being replayed as a different type than the one asked for here.");
  }

  // Begins the run: reading and decoding start immediately, and messages reach their consumers
  // once every attach()ed topic's consumer has subscribed. catalog must be the compiled plan's
  // endpoint catalog (holoscan::compile(graph).endpoint_catalog()). Every topic's publisher is
  // attached synchronously before this returns. Call it after holoscan::run_async(plan).
  // Throws std::runtime_error if any publisher fails to attach, and std::logic_error if called
  // twice; replay a recording again with a new McapReplayer.
  //
  // Subscribing happens on the graph's side after this returns, so a consumer that never
  // subscribes cannot be reported from here. If an attach()ed topic still has no subscriber
  // kSubscriberWaitDeadline (5 s) into the run, the run stops itself without sending anything,
  // logs an error naming each such topic, and stats().never_subscribed() is true for each.
  // stop() is still needed afterwards to join the threads and log the summary.
  void start(const holoscan::ChannelEndpointCatalog& catalog);

  // Ends the run, joins its threads and logs the summary once. Safe to call if the run was never
  // started and safe to call repeatedly. The destructor calls it.
  //
  // start() and stop() must be called from the thread that owns this object (the one that
  // constructs and destroys it); calling them from different threads can race the destructor
  // to the run's threads.
  void stop();

  [[nodiscard]] const ReplayStats& stats() const { return context_->stats; }

  // Whether the run has ended: the reader reached the end of the recording and every message it
  // read has gone out, or the run gave up (stats().start_stalled(), stats().never_subscribed()),
  // failed to read (stats().read_failed()), or was stopped. False before start() and for as long as
  // a looping replay keeps going. The consumers may still be handling the last messages.
  [[nodiscard]] bool finished() const { return finished_.load(); }

  // How many messages the recording holds on topic's replayed channel, from its statistics
  // section; nullopt if the topic is not in the recording or the recording has none. A topic
  // recorded on several channels counts only the replayed one.
  [[nodiscard]] std::optional<std::uint64_t> message_count(std::string_view topic) const {
    const std::optional<mcap::ChannelId> channel = file_->channel_of(topic);
    return channel ? file_->message_count(*channel) : std::nullopt;
  }

  // Whether the recording at path has topic, checked before a replayer exists so a caller can
  // leave out a topic that only some recordings hold. Throws std::runtime_error if the file cannot
  // be opened, as the constructor does.
  [[nodiscard]] static bool recording_has_topic(const std::filesystem::path& path,
                                                std::string_view topic) {
    return McapFile::open(path).channel_of(topic).has_value();
  }

  // The codec ("h264" or "h265") the recording at path holds topic in, when topic is recorded as
  // encoded video whose codec can be read; empty otherwise, with the reason logged. Lets a caller
  // check the codec can be decoded, as the constructor does. Throws std::runtime_error if the file
  // cannot be opened. A template so that only a caller that includes the image schema pays for
  // verifying the topic as it.
  template <typename Image = holoscan::schema::ImageT>
  [[nodiscard]] static std::optional<std::string> recorded_video_codec(
      const std::filesystem::path& path, std::string_view topic) {
    McapFile file = McapFile::open(path);
    if (!file.verify_topic<Image>(topic)) {
      return std::nullopt;
    }
    return file.video_codec_of(topic);
  }

 private:
  // Validates the options and opens the recording; out of line so neither is repeated per topic
  // type in the constructor template.
  McapReplayer(const std::filesystem::path& path, ReplayOptions options);

  // Prepares one topic: verifies the recording holds T under that name, then builds the port its
  // messages are published on and the builder for them while T is still known; everything
  // downstream is type-erased.
  //
  // A tensor-schema T always takes the TypedTensorPort<T> path: ChannelPublisher<T> only exists
  // for types the SDK accepts as a single catalog payload, and a tensor-schema type's tensors are
  // published separately from its metadata (ChannelTensorSchemaPublisher). Consequently a
  // recording holding such a T flat, not wrapped in the tensor envelope, is readable but cannot
  // be published live, and is rejected here.
  template <typename T>
  void add_topic(Topic<T> declared) {
    const std::string_view topic = declared.name;
    if (!file_->verify_topic<T>(topic)) {
      throw std::invalid_argument("Topic '" + std::string(topic) +
                                  "' is not recorded as the type it was declared with; see the "
                                  "warning above for why.");
    }

    if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>) {
      if (!file_->is_tensor_channel(topic)) {
        throw std::invalid_argument(
            "Topic '" + std::string(topic) +
            "' is recorded as a tensor-schema type's flat bytes rather than wrapped in the "
            "tensor envelope, which this library cannot publish live; re-record it through the "
            "tensor envelope to replay it.");
      }
      add_replay_topic(topic, std::make_unique<TypedTensorPort<T>>(std::move(declared.contract)),
                       tensor_builder<T>(topic));
    } else {
      add_replay_topic(topic, std::make_unique<TypedPort<T>>(), make_message_builder<T>());
    }
  }

  // The builder for a tensor-schema topic. A holoscan::schema::ImageT topic the logger recorded as
  // encoded video is decoded on video_decoder_cuda_device, which is checked here for the codec the
  // topic was recorded in rather than left to fail frame by frame; any other tensor topic is
  // rebuilt from its tensor envelope.
  template <typename T>
  MessageBuilder tensor_builder(std::string_view topic) {
    if constexpr (std::is_same_v<T, holoscan::schema::ImageT>) {
      if (file_->is_paired_video(topic)) {
        // verify_topic() read the codec in add_topic(), and refuses a topic whose codec it cannot.
        const std::optional<std::string> codec = file_->video_codec_of(topic);
        if (!codec.has_value()) {
          throw std::invalid_argument("Topic '" + std::string(topic) +
                                      "' is recorded as encoded video, but the codec it was "
                                      "recorded in could not be read.");
        }
        if (const auto reason =
                video_decode_unavailable_reason(options_.video_decoder_cuda_device, *codec)) {
          throw std::invalid_argument("Topic '" + std::string(topic) +
                                      "' is recorded as encoded video, but it cannot be "
                                      "decoded: " +
                                      *reason +
                                      ". Choose a GPU with a hardware video decoder for replay.");
        }
        return make_compressed_video_builder(options_.video_decoder_cuda_device);
      }
    }
    return make_tensor_message_builder<T>();
  }

  // Registers one prepared topic. Not a template, to keep per-type code in add_topic<T>() minimal.
  void add_replay_topic(std::string_view topic, std::unique_ptr<PortOwner> port,
                        MessageBuilder build);

  [[nodiscard]] std::string replayed_topics() const;

  // The external endpoint name topic is imported and attached under: "<operator_name>/<topic>"
  // (see ReplayOptions::operator_name for why it is not the bare topic name).
  [[nodiscard]] std::string endpoint_name(std::string_view topic) const;

  // Reads, decodes and buffers as fast as the buffer allows, blocking while it is full.
  void prefetch_loop();

  // Releases each buffered message as it falls due and publishes it. Sole owner of the
  // PacingClock, which has no lock of its own; it is lent to ReplayBuffer::release_if_due() so
  // that deciding a message is due and popping it happen under the buffer lock.
  void pacing_loop();

  // The steps of pacing_loop(), all on the pacing thread. publish() sends one released message
  // through its topic's port and records how that went; the two record_ functions are its outcomes.
  void publish(BufferedEntry& released);
  void record_delivery(const ReplayTopic& topic, const BufferedEntry& released);
  void record_refusal(const ReplayTopic& topic);

  // Warns once if every message released so far carries the same recorded time, which means the
  // recording has no cadence to replay. Tracks that from the log times it is shown.
  void warn_if_recording_has_no_timeline(std::uint64_t log_time);

  // The run's start-up gate: pacing does not begin until this returns true. Polls two conditions
  // against deadlines counted from the same start: a little decoded ahead, so the first messages
  // are not charged for the reader's head start, and a subscriber on every attach()ed topic, so
  // the opening messages are not published into nothing.
  //
  // Both are fatal if they never happen. Buffering has a target (the most it normally waits for,
  // or every message the attached topics hold in a recording played once, if that is fewer); it is
  // done when the target is reached or the reader finishes, and a reader that makes no progress
  // for ReplayOptions::start_stall_timeout is reported by report_start_stalled(). A topic nobody
  // attach()ed is never waited on. An attach()ed topic still unsubscribed at
  // kSubscriberWaitDeadline is reported by report_missing_consumers(). Either failure returns
  // false with nothing sent, as does a stop request.
  //
  // It deliberately does not start the timeline: buffering can end with nothing buffered, and an
  // origin taken then would be in the past by the time the first message appeared, flushing the
  // opening stretch of the recording at full rate.
  [[nodiscard]] bool prime();

  // How many messages prime() waits to have buffered: the usual amount, or fewer if the attached
  // topics hold fewer in a recording that plays once, so that having them all counts as buffered
  // however long the reader still has to scan past data nobody attached. Looping has no such
  // bound, and neither has a recording without a statistics section.
  [[nodiscard]] std::size_t buffering_target() const;

  // Records and logs an error naming every attach()ed topic that never got a subscriber during
  // prime().
  void report_missing_consumers();

  // Records and logs an error that reading the recording made no progress during prime(), with
  // how many messages had been read by then. Does nothing if a stop was requested meanwhile, which
  // is the caller's choice and not a stalled reader.
  void report_start_stalled(std::uint64_t messages_read);

  [[nodiscard]] bool stopping() const { return stop_requested_.load(); }

  // Tells both run threads to finish without joining them. Called by stop() and by the pacing
  // thread when the run cannot begin; safe to call repeatedly from either.
  void request_stop();

  // Sleeps until the given instant or a stop request, whichever is first.
  void wait_for_stop_until(std::chrono::steady_clock::time_point until);

  ReplayOptions options_;
  std::optional<McapFile> file_;
  // The publishers in context_ borrow this runtime and must not outlive it, so it is declared
  // before context_ and therefore destroyed after it.
  std::optional<holoscan::ChannelRuntime> runtime_;
  std::shared_ptr<ReplayContext> context_;
  std::atomic<bool> started_{false};
  std::atomic<bool> stopped_{false};
  // Set by request_stop(), from stop() or from the pacing thread when the run cannot begin. Unlike
  // stopped_ it does not mean stop() has run, so stop() still joins the threads afterwards.
  std::atomic<bool> stop_requested_{false};
  // The warning describes the whole run, so it is logged once. Pacing thread only.
  bool warned_about_a_recording_without_a_timeline_ = false;
  // What warn_if_recording_has_no_timeline() tracks: the first recorded time released, whether any
  // other has been seen (log times are non-decreasing, so seeing only the first, even across a
  // wrap, means no cadence), and how many messages have been released. Pacing thread only.
  std::optional<std::uint64_t> first_log_time_;
  bool log_time_has_advanced_ = false;
  std::size_t released_total_ = 0;
  // Set when the pacing thread returns, however the run ended. See finished().
  std::atomic<bool> finished_{false};
  std::mutex stop_mutex_;
  std::condition_variable stop_signal_;
  std::thread prefetch_thread_;
  std::thread pacing_thread_;
};

}  // namespace isaac::os::mcap_replayer
