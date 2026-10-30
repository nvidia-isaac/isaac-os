/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "holoscan/channel/endpoint_catalog.hpp"
#include "holoscan/channel/runtime.hpp"
#include "holoscan/channel/tensor_schema_endpoints.hpp"
#include "holoscan/core/external_topic.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/logger/logger.hpp"
#include "holoscan/schema/flatbuffer_schema_package.hpp"
#include <fmt/format.h>

#include "common/mcap_logger/frame_encoder.h"
#include "common/mcap_logger/mcap_sink.h"
#include "common/mcap_logger/tensor_topic_poller.h"
#include "common/mcap_logger/topic_log_options.h"
#include "common/mcap_logger/topic_poller.h"
#include "common/mcap_logger/transparent_string_hash.h"
#include "common/mcap_logger/video_encode_options.h"

namespace isaac::os::mcap_logger {

// Aggregate options for the logger as a whole; what a single topic asks for -- how deep its queues
// are, whether it records as video -- is TopicLogOptions, passed to attach(). Add new fields at the
// END, after the last existing one: a caller that
// initializes positionally, or with fewer initializers than there are fields, keeps meaning what it
// meant, where a field inserted in the middle shifts every one after it -- and can quietly hand a
// value to a field of a different type that happens to accept it. Callers should use designated
// initializers, which survive appending and cannot misassign a field.
struct HsdkMcapLoggerOptions {
  // When timestamp_output_path is true (the default), this is the directory the logger writes
  // into, and the actual file is named mcap_log_<local-timestamp>.mcap inside it -- so forgetting
  // to change this between runs still produces a distinct file each time instead of silently
  // overwriting the previous run's recording. Set timestamp_output_path to false to use this as
  // the exact literal output file path instead (e.g. a test that wants full control over the
  // resulting name). Ignored entirely when enable_write_file is false.
  std::filesystem::path output_path;
  bool timestamp_output_path = true;
  // Whether this logger writes an .mcap file at all. Defaults to true, matching every prior
  // release's only behavior. Set to false for a stream-only run that persists nothing to disk --
  // in that mode, output_path()/output_path above are unused, and output_path() returns nullopt.
  bool enable_write_file = true;
  // Whether this logger also (or instead, if enable_write_file is false) streams every attached
  // topic live to a connected Foxglove client. Independent of enable_write_file -- at least one
  // of the two must be true, or the constructor throws.
  bool enable_stream = false;
  // Matches the Foxglove SDK's own default. A loopback bind is unreachable from anywhere but the
  // application's own network namespace -- forward the port (e.g. over SSH) or override this to
  // "0.0.0.0" or a specific interface to reach it from elsewhere. See the "Foxglove live
  // streaming" section of the repository's SECURITY.md before widening the bind. Ignored when
  // enable_stream is false.
  std::string websocket_host = "127.0.0.1";
  // Matches Foxglove's own client default (8765). Pass 0 to let the OS assign a free ephemeral
  // port instead. Ignored when enable_stream is false.
  std::uint16_t websocket_port = 8765;
  // How each chunk of the MCAP file is compressed: None, Lz4 or Zstd.
  foxglove::McapCompression compression = kDefaultMcapCompression;
};

// MCAP logger. Call attach<T>() for each graph output to log during compose(),
// then start(catalog) after holoscan::compile() to attach channel subscribers and spawn one
// poll thread per attached topic. No detach -- an attached topic logs for this object's life.
class HsdkMcapLogger {
 public:
  // Throws std::runtime_error if neither options.enable_write_file nor options.enable_stream is
  // true, if the output file can't be opened, or if streaming is enabled but the websocket
  // server can't bind (see McapSink's constructor) -- a run that can't actually record or stream
  // anything fails here rather than completing normally with every sample silently dropped.
  explicit HsdkMcapLogger(HsdkMcapLoggerOptions options)
      : options_(std::move(options)),
        output_path_(options_.enable_write_file ? std::make_optional(resolve_output_path(options_))
                                                : std::nullopt),
        sink_(std::make_shared<McapSink>(McapSinkOptions{.output_path = output_path_,
                                                         .enable_stream = options_.enable_stream,
                                                         .websocket_host = options_.websocket_host,
                                                         .websocket_port = options_.websocket_port,
                                                         .compression = options_.compression})) {}

  ~HsdkMcapLogger() { shutdown(); }

  HsdkMcapLogger(const HsdkMcapLogger&) = delete;
  HsdkMcapLogger& operator=(const HsdkMcapLogger&) = delete;

  // The exact file this logger writes to -- HsdkMcapLoggerOptions::output_path verbatim when
  // timestamp_output_path is false, or that path (treated as a directory) plus a generated
  // mcap_log_<timestamp>.mcap name when true (the default). nullopt when enable_write_file is
  // false (stream-only mode) -- there is no file. Valid for the life of this object.
  [[nodiscard]] const std::optional<std::filesystem::path>& output_path() const noexcept {
    return output_path_;
  }

  // Wires one graph output onto this logger, creating its exported HoloMQ topic on first use.
  // Call during your own Graph::compose(), before holoscan::compile(). Records a starter that
  // start() replays later, spawning a TopicPoller<T> for the attached topic.
  //
  // `topic` sizes this topic's queues; see TopicLogOptions. Its `encode` has to stay unset here:
  // video encoding is for tensor-schema image topics, so asking for it on a plain topic throws.
  template <typename T>
  void attach(holoscan::Graph& graph, std::string_view port_name, const holoscan::Output<T>& source,
              TopicLogOptions topic = {}, holoscan::ConnectionOptions options = {}) {
    static_assert(!holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>,
                  "T is a tensor-schema type; use the attach<T>(graph, port_name, source, "
                  "contract, ...) overload instead.");
    const std::string port_name_str(port_name);
    check_port_name_not_attached(port_name_str);
    ValidateTopicLogOptions(topic, port_name_str);
    if (topic.encode) {
      throw std::invalid_argument(fmt::format(
          "Unable to encode video for the MCAP logger's topic '{}': video encoding is supported "
          "for camera images only.",
          port_name_str));
    }
    auto exported_topic = graph.template export_topic<T>(
        "holomq://" + port_name_str, holoscan::TopicOptions{.endpoint_name = port_name_str});
    graph.add_flow(source, exported_topic.input(), EdgeOptionsFor(topic, options));

    starters_[port_name_str] = [this, port_name_str, queue_depth = topic.queue_depth](
                                   holoscan::ChannelRuntime& runtime,
                                   const holoscan::ChannelEndpointCatalog& catalog) {
      auto subscriber = runtime.template create_subscriber<T>(catalog, port_name_str);
      if (!subscriber) {
        throw std::runtime_error(fmt::format("Unable to attach the MCAP logger's topic '{}': {}",
                                             port_name_str,
                                             holoscan::runtime_error_message(subscriber.error())));
      }
      pollers_.push_back(std::make_unique<TopicPoller<T>>(port_name_str, std::move(*subscriber),
                                                          sink_, queue_depth));
    };
    claimed_channel_names_.try_emplace(port_name_str, ClaimedBy::kAttachedTopic);
  }

  // Tensor-schema counterpart to attach<T>() above, for T's like holoscan::schema::ImageT
  // whose tensor data is transported out-of-band rather than embedded in one flat buffer.
  // `contract` describes the bounds/memory-kind/transfer policy for T's tensor field(s) -- see
  // common/image_schema/image_schema.h's ImageTensorFieldContract() for a ready-made
  // helper for holoscan::schema::ImageT; a custom type with no helper builds its own
  // DescriptorTensorTopicContract<T> directly. The SDK's own export_topic<T> constraints
  // already make calling the wrong overload for a given T a compile error.
  //
  // Set `topic.encode` to record this topic as hardware-encoded video (foxglove.CompressedVideo)
  // instead of raw tensors, for T = holoscan::schema::ImageT only. Leave it unset -- the default
  // -- to record raw tensors. start() throws if this host has no usable hardware encoder on the
  // device `encode` names, rather than recording raw tensors in its place;
  // FrameEncoder::hardware_available() lets a caller that wants that fallback choose it openly.
  // The rest of `topic` sizes this topic's queues; see TopicLogOptions.
  template <typename T>
  void attach(holoscan::Graph& graph, std::string_view port_name, const holoscan::Output<T>& source,
              holoscan::DescriptorTensorTopicContract<T> contract, TopicLogOptions topic = {},
              holoscan::ConnectionOptions options = {}) {
    using RootType = typename holoscan::DescriptorTensorSchemaNativeTraits<T>::root_type;
    using Payload = holoscan::schema::DescriptorFlatBufferPayloadFor<RootType>;

    const std::string port_name_str(port_name);
    check_port_name_not_attached(port_name_str);
    // Rejected here, before anything is wired, so a bad request leaves no half-wired exported
    // topic behind -- same reason the duplicate-port check runs first.
    ValidateTopicLogOptions(topic, port_name_str);
    std::string metadata_channel_name;
    if (topic.encode) {
      // Only holoscan.schema.Image carries the width, height, and encoding the encoder needs:
      // the transported tensor is a flat byte buffer with no shape of its own, so no other
      // tensor-schema payload could describe a frame to encode.
      if constexpr (!std::is_same_v<RootType, holoscan::schema::Image>) {
        throw std::invalid_argument(fmt::format(
            "Unable to encode video for the MCAP logger's topic '{}': video encoding is "
            "supported for camera images only.",
            port_name_str));
      }
      // An encoded topic records its image metadata on a second channel named after it, so that
      // name has to be as exclusive as the port name itself. Another topic attached under it
      // would write a different schema to the same channel, and a channel carries one schema for
      // its whole life: whichever of the two wrote first would decide the schema, and the other's
      // messages would then be recorded under it and read back as garbage, with nothing logged.
      metadata_channel_name = VideoMetadataChannelName(port_name_str);
      if (claimed_channel_names_.contains(metadata_channel_name)) {
        throw std::invalid_argument(fmt::format(
            "Unable to encode video for the MCAP logger's topic '{}': it records its image "
            "metadata under '{}', and a topic is already attached under that name.",
            port_name_str, metadata_channel_name));
      }
    }
    // No subscriber bounds: the edge's depth sizes the queue, see EdgeOptionsFor().
    auto exported_topic = graph.template export_topic<T>(
        "holomq://" + port_name_str, holoscan::TopicOptions{.endpoint_name = port_name_str},
        std::move(contract));
    graph.add_flow(source, exported_topic.input(), EdgeOptionsFor(topic, options));

    starters_[port_name_str] = [this, port_name_str, topic = std::move(topic)](
                                   holoscan::ChannelRuntime& runtime,
                                   const holoscan::ChannelEndpointCatalog& catalog) {
      start_tensor_topic<Payload>(runtime, catalog, port_name_str, topic);
    };
    // Claimed only now that this attach() has succeeded. The sidecar name is claimed whether or
    // not this host turns out to have a usable video encoder, so which names a caller may attach
    // does not depend on the machine the graph happens to run on.
    claimed_channel_names_.try_emplace(port_name_str, ClaimedBy::kAttachedTopic);
    if (!metadata_channel_name.empty()) {
      claimed_channel_names_.try_emplace(std::move(metadata_channel_name),
                                         ClaimedBy::kVideoMetadataSidecar);
    }
  }

  // Call once after holoscan::compile() succeeds. Creates this logger's ChannelRuntime, then
  // replays every starter recorded by attach() during compose().
  void start(const holoscan::ChannelEndpointCatalog& catalog) {
    if (runtime_ || terminated_) {
      throw std::runtime_error(
          "This MCAP logger has already been started or shut down, and cannot be started "
          "again.");
    }
    auto runtime_result = holoscan::ChannelRuntime::create();
    if (!runtime_result) {
      throw std::runtime_error(
          fmt::format("Unable to initialize the MCAP logger's messaging runtime: {}",
                      holoscan::runtime_error_message(runtime_result.error())));
    }
    runtime_ = std::make_unique<holoscan::ChannelRuntime>(std::move(*runtime_result));

    for (auto& [port_name, starter] : starters_) {
      starter(*runtime_, catalog);
    }
  }

  // Safe to call unconditionally. Terminal: once called, this logger can never start again --
  // a restart would receive on pollers that have already stopped.
  void shutdown() {
    terminated_ = true;
    // Each poller's destructor joins its poll thread, which first writes whatever its queue still
    // holds, so when this returns every sample that was not dropped has been written and every read
    // claim has been released -- all before the runtime they came from is destroyed.
    pollers_.clear();
    runtime_.reset();
    // No poller is left running, so nothing still holds a shared_ptr<McapSink>. Dropping our own
    // reference here finalizes the output file -- this is what makes shutdown() (not just
    // ~HsdkMcapLogger()) finalize it.
    sink_.reset();
  }

 private:
  // Attaches the subscriber for one tensor-schema topic and starts its poller. Run by start(), for
  // every topic attach() recorded, once the graph is compiled and there are endpoints to attach to.
  template <typename Payload>
  void start_tensor_topic(holoscan::ChannelRuntime& runtime,
                          const holoscan::ChannelEndpointCatalog& catalog,
                          const std::string& port_name, const TopicLogOptions& topic) {
    auto subscriber = runtime.template create_tensor_schema_subscriber<Payload>(catalog, port_name);
    if (!subscriber) {
      throw std::runtime_error(fmt::format("Unable to attach the MCAP logger's topic '{}': {}",
                                           port_name,
                                           holoscan::runtime_error_message(subscriber.error())));
    }
    pollers_.push_back(std::make_unique<TensorTopicPoller<Payload>>(
        port_name, std::move(*subscriber), sink_, topic.queue_depth,
        ChooseFrameEncoder(topic.encode, port_name)));
  }

  // The encoder a topic records with, or null to record raw tensors.
  //
  // Whether a topic records video or raw tensors is settled here, once, before its first sample
  // creates the topic's channel -- a channel carries one schema for its whole life, so there is no
  // falling back to raw tensors later. Deferred to start() rather than decided while composing the
  // graph because only now is the device known to be usable.
  //
  // A topic that asked for video on a device that cannot encode throws out of start() rather than
  // recording raw tensors: the run would otherwise look fine while producing a file without the
  // video the caller asked for.
  [[nodiscard]] static std::shared_ptr<FrameEncoder> ChooseFrameEncoder(
      const std::optional<VideoEncodeOptions>& encode, const std::string& port_name) {
    if (!encode) {
      return nullptr;
    }
    return std::make_shared<FrameEncoder>(*encode, port_name);
  }

  using Starter =
      std::function<void(holoscan::ChannelRuntime&, const holoscan::ChannelEndpointCatalog&)>;

  // What claimed a name in claimed_channel_names_. Recorded only so a rejected attach() can say
  // which kind of channel it collided with: the two collisions have different fixes.
  enum class ClaimedBy { kAttachedTopic, kVideoMetadataSidecar };

  // Shared by both attach<T>() overloads, and the only place an attached port name is checked --
  // an encoded topic's sidecar name is checked at its own attach site, against this same
  // container. Two topics on one name would otherwise go undetected: starters_[port_name_str] = ...
  // silently overwrites an earlier attach() under the same port_name, dropping its starter (and
  // therefore its poller) from start() entirely -- with no error anywhere, and the earlier topic
  // simply never getting logged. Called before either overload does any other work, so a rejected
  // attach() leaves no half-wired exported topic behind.
  void check_port_name_not_attached(const std::string& port_name_str) const {
    const auto claim = claimed_channel_names_.find(port_name_str);
    if (claim == claimed_channel_names_.end()) {
      return;
    }
    if (claim->second == ClaimedBy::kVideoMetadataSidecar) {
      // A video-encoded topic already records its image metadata under this name, and one channel
      // carries one schema for its whole life -- attaching here would silently record two
      // different schemas on that one channel, leaving one of them unreadable.
      throw std::invalid_argument(
          fmt::format("The MCAP logger already records the image metadata of a video-encoded "
                      "topic under port_name '{}'.",
                      port_name_str));
    }
    throw std::invalid_argument(fmt::format(
        "The MCAP logger already has a topic attached under port_name '{}'.", port_name_str));
  }

  // Uses local time with second precision (YYYYMMDD_HHMMSS).
  [[nodiscard]] static std::filesystem::path resolve_output_path(
      const HsdkMcapLoggerOptions& options) {
    if (!options.timestamp_output_path) {
      return options.output_path;
    }
    // output_path is a directory in this mode; a file-like extension here is almost always a
    // caller habit (e.g. still writing "run.mcap") rather than an intentionally weird directory
    // name -- warn rather than silently creating a directory literally named that.
    if (options.output_path.has_extension()) {
      HOLOSCAN_LOG_WARN(
          "The MCAP logger's output_path '{}' looks like a file (extension '{}'), but "
          "timestamp_output_path is true, so it is being treated as a directory -- a "
          "mcap_log_<timestamp>.mcap file will be created inside it. Set timestamp_output_path "
          "= false to use output_path as the exact file to write instead.",
          options.output_path.string(), options.output_path.extension().string());
    }
    const auto now_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local_tm{};
    localtime_r(&now_time_t, &local_tm);
    std::ostringstream timestamp;
    timestamp << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
    return options.output_path / fmt::format("mcap_log_{}.mcap", timestamp.str());
  }

  HsdkMcapLoggerOptions options_;
  std::optional<std::filesystem::path> output_path_;
  std::shared_ptr<McapSink> sink_;
  // What start() replays, keyed by port name. Not the authority on which names are taken: a
  // video-encoded topic occupies a second name nothing polls.
  std::unordered_map<std::string, Starter, TransparentStringHash, std::equal_to<>> starters_;
  // The single authority on which channel names are already taken -- every attach() claims its
  // port name here, and an encoded attach() also claims its metadata sidecar name. Kept as one
  // container so a name can only be free or taken, never free by one test and taken by another.
  std::unordered_map<std::string, ClaimedBy, TransparentStringHash, std::equal_to<>>
      claimed_channel_names_;
  // Stored through PollerBase since one logger attaches many different T's -- no single
  // std::vector<TopicPoller<T>> could hold them all.
  std::vector<std::unique_ptr<PollerBase>> pollers_;
  std::unique_ptr<holoscan::ChannelRuntime> runtime_;
  // Set by shutdown(); once true, start() refuses to run again.
  bool terminated_ = false;
};

}  // namespace isaac::os::mcap_logger
