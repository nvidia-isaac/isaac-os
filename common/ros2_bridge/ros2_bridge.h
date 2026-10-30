/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "holoscan/channel/endpoint_catalog.hpp"
#include "holoscan/channel/runtime.hpp"
#include "holoscan/core/external_topic.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/logger/logger.hpp"
#include <fmt/format.h>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include "common/ros2_bridge/converter_concepts.h"
#include "common/ros2_bridge/cuda_buffer_config.h"
#include "common/ros2_bridge/cuda_device_utils.h"
#include "common/ros2_bridge/default_ros2_converter.h"
#include "common/ros2_bridge/memory_transfer.h"
#include "common/ros2_bridge/ros2_timestamp.h"
#include "common/ros2_bridge/ros2_topic_spec.h"
#include "common/ros2_bridge/tensor_converter.h"

namespace isaac::os::ros2_bridge {

/// Ensures rclcpp::init() has been called exactly once for this process. Idempotent and
/// thread-safe (mutex-guarded, not std::call_once, so it stays correct if rclcpp::shutdown()
/// is ever called); multiple rclcpp::Node instances can share one init. The Ros2Bridge shuts ROS
/// down again when it is destroyed.
///
/// rclcpp's own SIGINT/SIGTERM handler is not installed: it shuts ROS down from the signal while
/// the graph is still running, which unloads the CUDA buffer backend under it and crashes the
/// process. The application owns signal handling and stops the graph, then the bridge, in order.
inline void EnsureRclcppInit() {
  static std::mutex init_mutex;
  const std::lock_guard<std::mutex> guard(init_mutex);
  if (!rclcpp::ok()) {
    rclcpp::InitOptions init_options;
    init_options.auto_initialize_logging(false);
    rclcpp::init(0, nullptr, init_options, rclcpp::SignalHandlerOptions::None);
  }
}

// The bridge's ROS2-facing runtime: a plain field bundle, not a class with its own methods.
// Held via std::optional<RuntimeState> inside Ros2Bridge, engaged by start() and disengaged
// by stop() -- Ros2Bridge's own start()/stop() do all the actual work on these fields
// directly. Grouping them into one struct behind one optional makes "started or not" a
// single invariant enforced by the type system, rather than several independently-nullable
// members that would have to be trusted to stay in sync by convention.
struct RuntimeState {
  rclcpp::Node::SharedPtr node;
  holoscan::ChannelRuntime runtime;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  std::thread spin_thread;
  // Set by the spin thread just after executor->spin() returns. Ros2Bridge::stop() polls
  // this (retrying cancel() in between) instead of trusting a single cancel() call -- see
  // stop()'s comment.
  std::shared_ptr<std::atomic<bool>> spin_finished;
  std::vector<std::stop_source> poll_stop_sources;
  std::vector<std::thread> poll_threads;
  std::vector<std::shared_ptr<void>> subscriptions;
};

namespace detail {

// Appends T to Acc unless it's already present, preserving first-seen order.
template <typename Acc, typename T>
struct AppendUnique;

template <typename... Ts, typename T>
struct AppendUnique<std::tuple<Ts...>, T> {
  using Type =
      std::conditional_t<(std::is_same_v<T, Ts> || ...), std::tuple<Ts...>, std::tuple<Ts..., T>>;
};

// Folds Pairs' ::HsdkType members into a tuple of unique types, in first-seen order. Multiple
// Ros2Types can share one HsdkType (e.g. Pose and PoseStamped both -> isaac.PoseCpu), and this
// collapses those down to a single entry -- see TopicVariantFor for why that matters.
template <typename Acc, typename Pairs>
struct UniqueHsdkTypes;

template <typename Acc>
struct UniqueHsdkTypes<Acc, std::tuple<>> {
  using Type = Acc;
};

template <typename Acc, typename Pair, typename... Rest>
struct UniqueHsdkTypes<Acc, std::tuple<Pair, Rest...>>
    : UniqueHsdkTypes<typename AppendUnique<Acc, typename Pair::HsdkType>::Type,
                      std::tuple<Rest...>> {};

// Builds std::variant<ImportedTopic<HsdkType>...> (or ExportedTopic<...>) from a tuple of
// unique HsdkTypes, so Ros2Bridge can store "any of the registered HsdkTypes" per port_name
// without std::any. Must be built from unique HsdkTypes, not one alternative per SupportedPairs
// entry: two Ros2Types sharing one HsdkType would otherwise give the variant a duplicate
// alternative, making every converting construction and std::get<> against it an ambiguous,
// hard compile error.
template <template <typename> class TopicTemplate, typename UniqueTypes>
struct TopicVariantFor;

template <template <typename> class TopicTemplate, typename... UniqueTypes>
struct TopicVariantFor<TopicTemplate, std::tuple<UniqueTypes...>> {
  using Type = std::variant<TopicTemplate<UniqueTypes>...>;
};

}  // namespace detail

// port_name -> variant<ImportedTopic<HsdkT>...>, one entry per unique HsdkType in
// ConverterT::SupportedPairs.
template <typename ConverterT>
using ImportedTopicVariant = typename detail::TopicVariantFor<
    holoscan::ImportedTopic, typename detail::UniqueHsdkTypes<
                                 std::tuple<>, typename ConverterT::SupportedPairs>::Type>::Type;
template <typename ConverterT>
using ExportedTopicVariant = typename detail::TopicVariantFor<
    holoscan::ExportedTopic, typename detail::UniqueHsdkTypes<
                                 std::tuple<>, typename ConverterT::SupportedPairs>::Type>::Type;

template <typename ConverterT = DefaultRos2Converter>
class Ros2Bridge {
 public:
  /// Constructs a bridge that will run its ROS2 node under node_name. Does not wire or start
  /// anything -- call from_ros2()/to_ros2() for each port during Graph::compose(), then start()
  /// after compile().
  /// @param node_name Name of the rclcpp::Node this bridge creates in start().
  /// @param pool_device The one CUDA device every kCudaDevice topic uses: the ROS CUDA buffer pool
  /// is a process-wide singleton bound to a single device, so everything the bridge allocates from
  /// it, and every CUDA buffer it can import, lives there. Ignored unless a tensor topic has
  /// memory_kind kCudaDevice.
  /// @param node_options Options for the rclcpp::Node. Set use_intra_process_comms(true) when other
  /// nodes in this process, such as a test's or a demo's, publish to or subscribe from the bridge's
  /// topics, and give those nodes the same option: their messages then skip DDS, and a CUDA buffer
  /// is handed over in place. Without it, a CUDA buffer between two nodes of one process takes the
  /// ROS CUDA buffer backend's same-process IPC path, which blocks every publish on the GPU and
  /// logs a warning. Peers in other processes are unaffected by this option.
  explicit Ros2Bridge(std::string node_name = "isaac_os_bridge", int pool_device = 0,
                      rclcpp::NodeOptions node_options = rclcpp::NodeOptions())
      : node_name_(std::move(node_name)),
        pool_device_(pool_device),
        node_options_(std::move(node_options)) {}

  /// Calls stop() to ensure a started bridge's runtime is torn down before destruction, then
  /// shuts rclcpp down. rclcpp's static default context would otherwise shut ROS down during
  /// process exit, which unloads the rosidl::Buffer backend plugins after class_loader's own
  /// static state is already gone and segfaults. Assumes one bridge per process, which owns ROS.
  ~Ros2Bridge() {
    stop();

    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }

  // Cannot be copied or moved: once wired, graphs hold a reference to the bridge, and once
  // started, runtime_state_ owns joinable threads and endpoints tied to the runtime's address.
  Ros2Bridge(Ros2Bridge&&) = delete;
  Ros2Bridge& operator=(Ros2Bridge&&) = delete;
  Ros2Bridge(const Ros2Bridge&) = delete;
  Ros2Bridge& operator=(const Ros2Bridge&) = delete;

  /// Wires one ROS2 topic into the graph, creating its imported HoloMQ topic on first use. Call
  /// from inside your own Graph::compose(), before holoscan::compile(). Ros2T/HsdkT are resolved
  /// against ConverterT::SupportedPairs at compile time -- an unregistered or ambiguous pair is a
  /// static_assert failure, not a runtime throw. Memoized on spec.port_name, so a second call for
  /// an already-wired port reuses the same imported topic instead of re-importing it -- but
  /// fanning that reused topic out to two consumers still won't compile
  /// (EXTERNAL_CHANNEL_PLAN_INVALID; no manual-partition workaround exists for this).
  ///
  /// @param graph The graph being composed; receives the imported topic and the new edge.
  /// @param target The graph input this topic's imported output is connected to.
  /// @param spec Port name, ROS2 topic name, qos, and (for a tensor-backed HsdkT) max_tensor_bytes
  /// for this topic.
  /// @param options Holoscan connection options forwarded to Graph::add_flow.
  /// @throws std::runtime_error If spec.port_name was already wired with a different HsdkT or
  /// a different ros2_topic.
  template <typename Ros2T, typename HsdkT>
  void from_ros2(holoscan::Graph& graph, const holoscan::Input<HsdkT>& target, Ros2TopicSpec spec,
                 holoscan::ConnectionOptions options = {}) {
    note_topic<HsdkT>(spec);
    create_graph_topic<Ros2T, HsdkT>(graph, spec, /*is_ingress=*/true);
    auto& imported_topic = std::get<holoscan::ImportedTopic<HsdkT>>(imported_.at(spec.port_name));
    graph.add_flow(imported_topic.output(), target, options);

    ingress_ros2_topics_[spec.port_name] = spec.ros2_topic;
    dispatch_by_type_pair<typename ConverterT::SupportedPairs, Ros2T, HsdkT>([&](auto traits) {
      using Traits = decltype(traits);
      ingress_starters_[spec.port_name] = [spec](holoscan::ChannelRuntime& runtime,
                                                 const holoscan::ChannelEndpointCatalog& catalog,
                                                 rclcpp::Node& node) -> std::shared_ptr<void> {
        auto shared_publisher = attach_ingress_publisher<Traits>(runtime, catalog, spec.port_name);
        return create_ingress_subscription<Traits>(node, spec, std::move(shared_publisher));
      };
    });
  }

  /// Wires one graph output onto a ROS2 topic, creating its exported HoloMQ topic on first use.
  /// Same laziness, memoization, and compile-time converter resolution as from_ros2() above.
  /// Unlike ingress, a second to_ros2 for an already-wired egress port_name fails loudly
  /// (GRAPH_INPUT_FANIN_UNSUPPORTED) rather than silently rewiring: Holoscan rejects a second
  /// edge into an already-targeted Input regardless of source.
  ///
  /// @param graph The graph being composed; receives the exported topic and the new edge.
  /// @param source The graph output to connect onto this topic's exported input.
  /// @param spec Port name, ROS2 topic name, qos, and (for a tensor-backed HsdkT) max_tensor_bytes
  /// for this topic.
  /// @param options Holoscan connection options forwarded to Graph::add_flow.
  /// @throws std::runtime_error If spec.port_name was already wired with a different HsdkT or
  /// a different ros2_topic.
  template <typename Ros2T, typename HsdkT>
  void to_ros2(holoscan::Graph& graph, const holoscan::Output<HsdkT>& source, Ros2TopicSpec spec,
               holoscan::ConnectionOptions options = {}) {
    note_topic<HsdkT>(spec);
    create_graph_topic<Ros2T, HsdkT>(graph, spec, /*is_ingress=*/false);
    auto& exported_topic = std::get<holoscan::ExportedTopic<HsdkT>>(exported_.at(spec.port_name));
    graph.add_flow(source, exported_topic.input(), options);

    egress_ros2_topics_[spec.port_name] = spec.ros2_topic;
    dispatch_by_type_pair<typename ConverterT::SupportedPairs, Ros2T, HsdkT>([&](auto traits) {
      using Traits = decltype(traits);
      egress_starters_[spec.port_name] =
          [spec](holoscan::ChannelRuntime& runtime, const holoscan::ChannelEndpointCatalog& catalog,
                 rclcpp::Node& node, std::stop_token stop_token) -> std::thread {
        auto channel_subscriber =
            attach_egress_subscriber<Traits>(runtime, catalog, spec.port_name);
        auto ros2_publisher = node.template create_publisher<Ros2T>(spec.ros2_topic, spec.qos);
        return spawn_egress_poll_thread<Traits>(std::move(channel_subscriber), ros2_publisher,
                                                std::move(stop_token), spec);
      };
    });
  }

  /// Call once after holoscan::compile() succeeds. Creates the bridge's ChannelRuntime and
  /// ROS2 node, then replays every ingress/egress starter recorded by from_ros2()/to_ros2()
  /// during compose() to attach ROS2 subscriptions/publishers.
  ///
  /// Teardown ordering: stop the Holoscan RunSession (request_stop()/wait()) before, or
  /// concurrently with, stopping this bridge -- never after. This bridge's poll threads and ROS2
  /// callbacks keep calling into the graph's channels while started, so stopping it first tears
  /// those channels out from under a live graph. See test_ros2_bridge_roundtrip.cc for the
  /// correct ordering.
  ///
  /// @param catalog Channel endpoint catalog used to attach every wired topic's publisher(s)/
  /// subscriber(s); typically holoscan::compile()'s resulting plan's endpoint catalog.
  /// @throws std::runtime_error If the ChannelRuntime or any publisher/subscriber attachment
  /// fails, if the CUDA buffer pool cannot be bound to pool_device (a kCudaDevice topic exists),
  /// or if start() was already called on this bridge.
  void start(const holoscan::ChannelEndpointCatalog& catalog) {
    if (runtime_state_) {
      throw std::runtime_error("Ros2Bridge::start() was already called on this bridge.");
    }

    EnsureRclcppInit();

#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
    // The CUDA buffer pool is a process-wide singleton bound to one device at first use, so choose
    // that device deliberately before any buffer is allocated. Throws if it cannot be bound.
    if (uses_cuda_pool_) {
      BindPoolDevice(pool_device_);
    }
#endif

    auto runtime_result = holoscan::ChannelRuntime::create();
    if (!runtime_result) {
      throw std::runtime_error(
          fmt::format("Unable to create the ROS2 bridge's ChannelRuntime: {}",
                      holoscan::runtime_error_message(runtime_result.error())));
    }
    // Engaged for the rest of this bridge's started lifetime -- every publisher/subscriber
    // it creates below must not outlive it (confirmed in channel/runtime.hpp's own doc
    // comment), which holds here since it's a member, not a local. Only node/runtime are set
    // here; executor/spin_thread need runtime_state_ to already exist (their lambda captures
    // a pointer into it), so they're set just below instead of in this same aggregate-init.
    runtime_state_.emplace(RuntimeState{
        .node = std::make_shared<rclcpp::Node>(node_name_, node_options_),
        .runtime = std::move(*runtime_result),
    });

    // From here on, a thrown exception (a channel attach failure partway through either loop
    // below, a std::thread constructor failure, etc.) must not leave this bridge half-started:
    // runtime_state_ is already engaged, so without this guard the caller would be left holding
    // a Ros2Bridge that looks "started" (runtime_state_ engaged) with some threads/subscriptions
    // already live, but that never finished start() and was never stop()'d -- relying on this
    // object's own destructor to clean that up would only work if the bridge itself happens to
    // unwind off the stack together with the exception, which callers should not have to arrange.
    // stop() is safe to call on a RuntimeState in any partially-populated state (every field it
    // touches is null/empty/non-joinable by default), so it's a safe universal rollback here.
    try {
      runtime_state_->executor = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
      runtime_state_->executor->add_node(runtime_state_->node);
      runtime_state_->spin_finished = std::make_shared<std::atomic<bool>>(false);
      // Capture a raw pointer to the heap-allocated executor, not `this` -- defensive against a
      // future move of this bridge while started, even though move-assignment is deleted and
      // nothing today moves a started bridge. The executor object itself does not relocate when
      // the owning unique_ptr moves, so this pointer stays valid regardless; capturing `this`
      // would not.
      auto* executor = runtime_state_->executor.get();
      auto finished = runtime_state_->spin_finished;
      runtime_state_->spin_thread =
          std::thread([executor, finished, uses_pool = uses_cuda_pool_]() {
            // The ROS buffer backend checks a CUDA buffer's GPU against this thread's current GPU
            // while it deserializes a message, so a bridge with a CUDA topic must put this thread
            // on the pool device before it spins.
            if (uses_pool && !use_pool_device_or_log("ingress spin thread")) {
              finished->store(true);
              return;
            }
            executor->spin();
            finished->store(true);
          });

      // Every ingress_starters_/egress_starters_ entry was recorded by a from_ros2()/to_ros2()
      // call during compose(); there is no "configured but unwired" case to skip anymore -- a
      // starter only exists if the caller actually wired that port_name.
      for (auto& [port_name, starter] : ingress_starters_) {
        // rclcpp does not retain subscriptions on the caller's behalf -- a node/executor only
        // stores a weak reference -- so this shared_ptr must be kept somewhere or it (and
        // everything its callback captures) is destroyed immediately. Type-erased to
        // shared_ptr<void> because the subscription's message type varies per topic.
        auto subscription = starter(runtime_state_->runtime, catalog, *runtime_state_->node);
        runtime_state_->subscriptions.push_back(std::move(subscription));
      }

      for (auto& [port_name, starter] : egress_starters_) {
        std::stop_source stop_source;
        auto poll_thread = starter(runtime_state_->runtime, catalog, *runtime_state_->node,
                                   stop_source.get_token());
        runtime_state_->poll_stop_sources.push_back(std::move(stop_source));
        runtime_state_->poll_threads.push_back(std::move(poll_thread));
      }
    } catch (...) {
      stop();
      throw;
    }
  }

  /// Stops the bridge's ROS2-facing runtime at an explicit point, rather than only implicitly
  /// via the destructor. See start()'s ordering requirement. Safe to call unconditionally
  /// (no-op if never started, or already stopped); a fresh start() afterward is valid.
  void stop() {
    if (!runtime_state_) return;

    for (auto& source : runtime_state_->poll_stop_sources) {
      source.request_stop();
    }
    for (auto& thread : runtime_state_->poll_threads) {
      if (thread.joinable()) thread.join();
    }

    if (runtime_state_->executor) {
      // A single executor->cancel() call races with the spin thread: if cancel()'s
      // interrupt-guard-condition trigger fires before that thread has actually entered its
      // blocking DDS wait (very likely here, since start() returns almost immediately and
      // stop() typically runs shortly after), the trigger is not observed and spin() blocks
      // forever, hanging the join below. Retry cancel() until the spin thread itself reports
      // (via spin_finished) that executor->spin() has returned, which is unaffected by that
      // race and guarantees termination.
      int attempts = 0;
      while (runtime_state_->spin_thread.joinable() && runtime_state_->spin_finished &&
             !runtime_state_->spin_finished->load()) {
        runtime_state_->executor->cancel();
        if (++attempts >= kMaxCancelAttempts) {
          HOLOSCAN_LOG_ERROR(
              "The ROS2 bridge's executor did not stop after {} attempts; shutdown may hang.",
              attempts);
          break;
        }
        std::this_thread::sleep_for(kCancelRetryInterval);
      }
    }
    if (runtime_state_->spin_thread.joinable()) {
      runtime_state_->spin_thread.join();
    }

    // Members are destroyed in reverse declaration order, so this destroys subscriptions
    // (declared last in RuntimeState) before runtime (declared second) -- which matters:
    // each subscription's callback captures a ChannelPublisher, which must not outlive the
    // ChannelRuntime that created it.
    runtime_state_.reset();
  }

  /// The one CUDA device every kCudaDevice topic uses (the constructor's pool_device). Use it as
  /// the device in the bind_imported_topic_device binding for each device ingress topic.
  [[nodiscard]] int pool_device() const noexcept { return pool_device_; }

 private:
  // Which way a topic flows -- ROS2 into Holoscan (from_ros2, imported_) or
  // Holoscan into ROS2 (to_ros2, exported_) -- used as a plain runtime argument by
  // warn_relay_dropped for its log wording.
  enum class RelayDirection { kRos2ToHolo, kHoloToRos2 };

  /// Creates the HoloMQ topic for spec.port_name (imported_ if is_ingress, exported_ otherwise)
  /// via Graph::import_topic/export_topic, or reuses it if already created. Shared skeleton
  /// behind from_ros2/to_ros2. Rejects a port_name re-wired with a different HsdkT or a different
  /// ros2_topic; otherwise leaves fan-out/fan-in validation to Holoscan's own compile() checks.
  /// @param graph The graph being composed; receives the imported/exported topic on first use.
  /// @param spec Port/topic spec for the port being wired; spec.port_name selects (or creates)
  /// the topic, spec.max_tensor_bytes is forwarded to Traits::contract() for a tensor-backed
  /// HsdkT.
  /// @param is_ingress True to create/reuse an imported (ROS2 -> Holoscan) topic, false for an
  /// exported (Holoscan -> ROS2) topic.
  /// @throws std::runtime_error If spec.port_name was already wired with a different HsdkT or a
  /// different ros2_topic.
  template <typename Ros2T, typename HsdkT>
  void create_graph_topic(holoscan::Graph& graph, const Ros2TopicSpec& spec, bool is_ingress) {
    // Checked before the existing-port lookups so a bad type pair on a re-wire fails here, not
    // as a misleading "different HsdkT" error or a silent no-op below.
    static_assert(
        detail::MatchCount<typename ConverterT::SupportedPairs, Ros2T, HsdkT>::value == 1,
        "No unambiguous ROS2 converter registered for this (Ros2Type, HsdkType) pair -- see "
        "dispatch_by_type_pair's own static_asserts in default_ros2_converter.h.");

    const bool already_wired = is_ingress ? port_already_wired<holoscan::ImportedTopic<HsdkT>>(
                                                imported_, ingress_ros2_topics_, spec, "ingress")
                                          : port_already_wired<holoscan::ExportedTopic<HsdkT>>(
                                                exported_, egress_ros2_topics_, spec, "egress");
    if (already_wired) {
      return;
    }

    dispatch_by_type_pair<typename ConverterT::SupportedPairs, Ros2T, HsdkT>([&](auto traits) {
      using Traits = decltype(traits);
      static_assert_converter_kind<Traits, HsdkT>();
      if (is_ingress) {
        add_imported_topic<Traits, HsdkT>(graph, spec);
      } else {
        add_exported_topic<Traits, HsdkT>(graph, spec);
      }
    });
  }

  /// Whether spec.port_name is already wired in `topics` for this direction.
  /// @tparam TopicT The topic type a wiring of this port with the same HsdkT would have created.
  /// @param topics The bridge's imported_ or exported_ topics.
  /// @param ros2_topics The ROS2 topic each port was wired to, for the same direction.
  /// @param direction "ingress" or "egress", for the error text.
  /// @return True if the port is already wired the same way, so there is nothing to create.
  /// @throws std::runtime_error If the port was already wired with a different HsdkT or to a
  /// different ros2_topic.
  template <typename TopicT, typename TopicsT>
  static bool port_already_wired(const TopicsT& topics,
                                 const std::unordered_map<std::string, std::string>& ros2_topics,
                                 const Ros2TopicSpec& spec, const char* direction) {
    const auto existing = topics.find(spec.port_name);
    if (existing == topics.end()) {
      return false;
    }
    if (!std::holds_alternative<TopicT>(existing->second)) {
      throw std::runtime_error(
          fmt::format("The ROS2 bridge port '{}' was already wired as {} with a different HsdkT.",
                      spec.port_name, direction));
    }
    if (spec.ros2_topic != ros2_topics.at(spec.port_name)) {
      throw std::runtime_error(
          fmt::format("The ROS2 bridge port '{}' was already wired to ROS2 topic '{}'; "
                      "cannot re-wire it to '{}'.",
                      spec.port_name, ros2_topics.at(spec.port_name), spec.ros2_topic));
    }
    return true;
  }

  /// Fails to compile, with a message naming the right converter kind, if the converter for this
  /// pair does not match whether HsdkT is a tensor-schema type.
  template <typename Traits, typename HsdkT>
  static constexpr void static_assert_converter_kind() {
    if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<HsdkT>) {
      static_assert(
          TensorConverterFor<Traits>,
          "ROS2 bridge: this (Ros2Type, HsdkType) pair is a tensor-schema type, so its "
          "converter must be a tensor converter providing a Payload alias, "
          "contract(const Ros2TopicSpec&), to_tensors(const std::shared_ptr<Ros2Type>&, "
          "const Ros2TopicSpec&) and from_tensors(const ChannelResolvedTensorBundle<Payload>&, "
          "const Ros2TopicSpec&). Specialize TensorConverter, not Converter.");
    } else {
      static_assert(
          PlainConverter<Traits>,
          "ROS2 bridge: this (Ros2Type, HsdkType) pair is a plain type, so its converter must "
          "provide Ros2Type and HsdkType aliases, to_hsdk(const Ros2Type&) and "
          "to_ros2(const HsdkType&) (returning Ros2Type, or holoscan::expected<Ros2Type, "
          "holoscan::Error> for a conversion that can fail). If HsdkType carries tensors, "
          "specialize TensorConverter instead.");
    }
  }

  /// Imports spec.port_name into the graph as a HoloMQ topic (ROS2 -> Holoscan). A tensor-schema
  /// topic also gets the converter's tensor contract for this spec.
  template <typename Traits, typename HsdkT>
  void add_imported_topic(holoscan::Graph& graph, const Ros2TopicSpec& spec) {
    const std::string uri = "holomq://" + spec.port_name;
    holoscan::TopicOptions options{.endpoint_name = spec.port_name};
    if (spec.capture_time_clock) {
      if constexpr (!kHasHeaderStamp<typename Traits::Ros2Type>) {
        HOLOSCAN_LOG_WARN(
            "ROS2 topic '{}' has no header.stamp to take a capture time from, so each message is "
            "stamped with the time the bridge received it.",
            spec.ros2_topic);
      }
      options.capture_time = holoscan::RequireCaptureTime{.clock = *spec.capture_time_clock};
    }
    if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<HsdkT>) {
      imported_.insert_or_assign(
          spec.port_name, ImportedTopicVariant<ConverterT>(graph.template import_topic<HsdkT>(
                              uri, options, Traits::contract(spec))));
    } else {
      imported_.insert_or_assign(
          spec.port_name,
          ImportedTopicVariant<ConverterT>(graph.template import_topic<HsdkT>(uri, options)));
    }
  }

  /// Exports spec.port_name from the graph as a HoloMQ topic (Holoscan -> ROS2). A tensor-schema
  /// topic also gets the converter's tensor contract for this spec.
  template <typename Traits, typename HsdkT>
  void add_exported_topic(holoscan::Graph& graph, const Ros2TopicSpec& spec) {
    const std::string uri = "holomq://" + spec.port_name;
    const holoscan::TopicOptions options{.endpoint_name = spec.port_name};
    if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<HsdkT>) {
      exported_.insert_or_assign(
          spec.port_name, ExportedTopicVariant<ConverterT>(graph.template export_topic<HsdkT>(
                              uri, options, Traits::contract(spec))));
    } else {
      exported_.insert_or_assign(
          spec.port_name,
          ExportedTopicVariant<ConverterT>(graph.template export_topic<HsdkT>(uri, options)));
    }
  }

  // Wiring-time checks for a topic. A kCudaDevice tensor topic needs the CUDA buffer backend and is
  // recorded so start() binds the pool.
  template <typename HsdkT>
  void note_topic(const Ros2TopicSpec& spec) {
    if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<HsdkT>) {
      RequireCudaBufferBackend(spec);
      if (spec.memory_kind == Ros2MemoryKind::kCudaDevice) {
        uses_cuda_pool_ = true;
      }
    }
  }

  // Bounds stop()'s cancel()-retry loop so a spin thread that never returns for a reason
  // unrelated to the trigger-loss race that loop exists to work around fails legibly (a
  // logged error, still followed by a join that may then block) instead of hanging silently
  // forever.
  static constexpr int kMaxCancelAttempts = 400;  // 400 * 5ms = 2s.
  static constexpr auto kCancelRetryInterval = std::chrono::milliseconds(5);

  // Rate-limits a relay-path warning to at most this many occurrences per topic, matching the
  // "limit log counts to avoid spamming" convention used elsewhere in this repo (see e.g.
  // operators/ucxx/ucxx_receiver_op.cc).
  static constexpr int kMaxRelayErrorLogs = 3;

  /// Logs a rate-limited warning for one dropped relay message, in the direction given by
  /// `direction`.
  /// @param direction Which way the message was being relayed, selecting the log wording.
  /// @param ros2_topic The ROS2 topic involved.
  /// @param port_name The Holoscan port/HoloMQ topic involved.
  /// @param error_count Running count for this topic; incremented here, compared against
  /// kMaxRelayErrorLogs to decide whether to log.
  /// @param error The failure being reported.
  static void warn_relay_dropped(RelayDirection direction, const std::string& ros2_topic,
                                 const std::string& port_name, int& error_count,
                                 const holoscan::Error& error) {
    if (error_count >= kMaxRelayErrorLogs) return;
    ++error_count;
    const char* suppressed_suffix =
        error_count == kMaxRelayErrorLogs
            ? " Further such messages for this topic will be suppressed."
            : "";
    if (direction == RelayDirection::kRos2ToHolo) {
      HOLOSCAN_LOG_WARN(
          "Dropped a message relayed from ROS2 topic '{}' to Holoscan port '{}': {}.{}", ros2_topic,
          port_name, holoscan::runtime_error_message(error), suppressed_suffix);
    } else {
      HOLOSCAN_LOG_WARN(
          "Dropped a message relayed from Holoscan port '{}' to ROS2 topic '{}': {}.{}", port_name,
          ros2_topic, holoscan::runtime_error_message(error), suppressed_suffix);
    }
  }

  /// Whether MsgT carries a std_msgs/Header-style stamp to take a capture time from.
  template <typename MsgT>
  static constexpr bool kHasHeaderStamp = requires(const MsgT& msg) {
    msg.header.stamp.sec;
    msg.header.stamp.nanosec;
  };

  /// The capture time to emit for `msg`: empty (no capture time) when the spec declared no capture
  /// clock; otherwise its header.stamp in nanoseconds, or the ROS2 time it was received at when
  /// the type has no header. An error for a stamp that is not a valid time, so the caller drops the
  /// message instead of emitting a wrong capture time.
  template <typename MsgT>
  [[nodiscard]] static holoscan::expected<std::optional<std::int64_t>, holoscan::Error>
  capture_ns_for(const Ros2TopicSpec& spec, const MsgT& msg, const rclcpp::Node& node) {
    if (!spec.capture_time_clock) return std::optional<std::int64_t>{};
    if constexpr (kHasHeaderStamp<MsgT>) {
      auto capture_ns = from_ros2_timestamp(msg.header.stamp.sec, msg.header.stamp.nanosec);
      if (!capture_ns) return holoscan::make_unexpected(std::move(capture_ns).error());
      return std::optional<std::int64_t>{*capture_ns};
    } else {
      return std::optional<std::int64_t>{node.now().nanoseconds()};
    }
  }

  /// Retries emitting one already-converted value onto the bridge's HoloMQ publisher briefly,
  /// while the channel is not ready or backpressured. Logs (rate-limited) and gives up on any
  /// other failure, or after kMaxAttempts retries. Only owns the retry/drop/log loop -- the
  /// caller has already done any conversion (Traits::to_hsdk/to_tensors).
  /// @param shared_publisher The HoloMQ publisher to emit onto.
  /// @param ros2_topic The originating ROS2 topic, used only for logging.
  /// @param port_name The destination Holoscan port, used only for logging.
  /// @param error_count Running drop-log count for this topic; forwarded to warn_relay_dropped.
  /// @param emit_args Forwarded verbatim to shared_publisher->emit(...) on each attempt.
  /// @param capture_ns When set, emitted as the sample's capture time in the topic's declared
  /// capture clock (emit_captured); when empty, the sample carries no capture time.
  template <typename PublisherT, typename... EmitArgs>
  static void relay_ros2_to_holoscan(const std::shared_ptr<PublisherT>& shared_publisher,
                                     const std::string& ros2_topic, const std::string& port_name,
                                     const std::shared_ptr<int>& error_count,
                                     std::optional<std::int64_t> capture_ns,
                                     const EmitArgs&... emit_args) {
    constexpr int kMaxAttempts = 5;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
      auto updated = shared_publisher->update_connections();
      if (!updated) {
        warn_relay_dropped(RelayDirection::kRos2ToHolo, ros2_topic, port_name, *error_count,
                           updated.error());
        return;
      }
      auto emitted = capture_ns ? shared_publisher->emit_captured(emit_args..., *capture_ns)
                                : shared_publisher->emit(emit_args...);
      if (emitted) return;
      if (emitted.error().code != holoscan::ErrorCode::kNotReady &&
          emitted.error().code != holoscan::ErrorCode::kBackpressured) {
        warn_relay_dropped(RelayDirection::kRos2ToHolo, ros2_topic, port_name, *error_count,
                           emitted.error());
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    warn_relay_dropped(RelayDirection::kRos2ToHolo, ros2_topic, port_name, *error_count,
                       holoscan::Error{holoscan::ErrorCode::kTimeout,
                                       fmt::format("gave up after {} attempts", kMaxAttempts)});
  }

  /// Attempts one receive-and-convert from the bridge's HoloMQ subscriber: tensor-schema uses
  /// receive_tensors()+from_tensors(), plain uses receive_sample()+to_ros2().
  /// @param channel_subscriber The HoloMQ subscriber to receive from.
  /// @param spec The topic being relayed; its names are used for logging and the converter reads
  /// its memory settings.
  /// @param error_count Running drop-log count for this topic; forwarded to warn_relay_dropped.
  /// @return The converted message, or nullopt on "not ready" (expected, not logged) or on any
  /// other receive/convert failure (logged via warn_relay_dropped).
  template <typename Traits, typename SubscriberT>
  static std::optional<typename Traits::Ros2Type> try_receive_ros2_message(
      SubscriberT& channel_subscriber, const Ros2TopicSpec& spec, int& error_count) {
    using HsdkT = typename Traits::HsdkType;
    using Ros2T = typename Traits::Ros2Type;
    holoscan::expected<Ros2T, holoscan::Error> ros2_msg =
        [&]() -> holoscan::expected<Ros2T, holoscan::Error> {
      if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<HsdkT>) {
        auto bundle = channel_subscriber.receive_tensors();
        if (!bundle) return holoscan::make_unexpected(bundle.error());
        return Traits::from_tensors(*bundle, spec);
      } else {
        auto sample = channel_subscriber.receive_sample();
        if (!sample) return holoscan::make_unexpected(sample.error());
        return Traits::to_ros2(sample->data);
      }
    }();
    if (!ros2_msg) {
      if (ros2_msg.error().code != holoscan::ErrorCode::kNotReady) {
        warn_relay_dropped(RelayDirection::kHoloToRos2, spec.ros2_topic, spec.port_name,
                           error_count, ros2_msg.error());
      }
      return std::nullopt;
    }
    return std::move(*ros2_msg);
  }

  /// Poll-loop body for the Holoscan-to-ROS2 direction: blocks on the bridge's HoloMQ
  /// subscriber's readiness and republishes each sample onto ROS2, until `stop_token` requests
  /// a stop. Runs on its own thread, started from start() and joined in stop().
  /// @param channel_subscriber The HoloMQ subscriber to wait on and receive from.
  /// @param ros2_publisher The ROS2 publisher each received message is republished onto.
  /// @param stop_token Observed each loop iteration; a requested stop ends this thread.
  /// @param spec The topic being relayed; its names are used for logging and the converter reads
  /// its memory settings.
  template <typename Traits, typename SubscriberT>
  static void poll_and_relay_holoscan_to_ros2(
      std::shared_ptr<SubscriberT> channel_subscriber,
      typename rclcpp::Publisher<typename Traits::Ros2Type>::SharedPtr ros2_publisher,
      std::stop_token stop_token, Ros2TopicSpec spec) {
    // A device topic copies into pooled buffers, which live on the pool device. Only a
    // tensor-schema topic can be a device topic; a plain pair ignores memory_kind.
    if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<
                      typename Traits::HsdkType>) {
      if (spec.memory_kind == Ros2MemoryKind::kCudaDevice &&
          !use_pool_device_or_log(spec.port_name)) {
        return;
      }
    }
    int error_count = 0;
    while (true) {
      // This thread has nothing else to do but wait for the next sample, so the wait has no
      // deadline once a publisher is connected. Until then, wake every millisecond: a producer's
      // first emit can be refused while this subscriber is still connecting, and waking refreshes
      // that connection, as the old poll loop did.
      const auto deadline = channel_subscriber->has_publishers()
                                ? std::chrono::steady_clock::time_point::max()
                                : std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
      auto wait_result = channel_subscriber->wait_for_sample(deadline, stop_token);
      if (!wait_result && wait_result.error().code == holoscan::ErrorCode::kTimeout) {
        continue;  // No publisher connected yet.
      }
      if (!wait_result) {
        // kInterrupted is stop(); kClosed is the producer ending, which is how an orderly graph
        // shutdown arrives (see start()). Neither is a relay failure.
        const auto code = wait_result.error().code;
        if (code != holoscan::ErrorCode::kInterrupted && code != holoscan::ErrorCode::kClosed) {
          // Not rate-limited: this happens once, and the relay does not come back.
          HOLOSCAN_LOG_ERROR(
              "The relay from Holoscan port '{}' to ROS2 topic '{}' has stopped and will forward "
              "no further messages: {}.",
              spec.port_name, spec.ros2_topic,
              holoscan::runtime_error_message(wait_result.error()));
        }
        break;
      }
      // wait_for_sample() is only a hint: the receive can still lose its race (kNotReady) or fail
      // with a logged error. Either way, go back to waiting.
      auto ros2_msg = try_receive_ros2_message<Traits>(*channel_subscriber, spec, error_count);
      if (!ros2_msg) {
        continue;
      }
      // Moved, not copied: a copy of a message holding a CUDA buffer clones the GPU memory, which
      // intra-process delivery would do on every frame.
      ros2_publisher->publish(std::move(*ros2_msg));
    }
  }

  /// Attaches the HoloMQ publisher appropriate for Traits::HsdkType (tensor-schema or plain)
  /// and wraps it in the shared_ptr the ROS2 subscription callback needs.
  /// @param runtime The bridge's ChannelRuntime, used to attach the publisher.
  /// @param catalog Channel endpoint catalog identifying this topic's endpoint.
  /// @param port_name Port name to attach the publisher for.
  /// @return A shared_ptr to the attached ChannelTensorSchemaPublisher<Payload> or
  /// ChannelPublisher<HsdkT>, depending on Traits::HsdkType.
  /// @throws std::runtime_error If the attach call fails.
  template <typename Traits>
  [[nodiscard]] static auto attach_ingress_publisher(
      holoscan::ChannelRuntime& runtime, const holoscan::ChannelEndpointCatalog& catalog,
      const std::string& port_name) {
    using HsdkT = typename Traits::HsdkType;
    if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<HsdkT>) {
      auto publisher = runtime.template create_tensor_schema_publisher<typename Traits::Payload>(
          catalog, port_name);
      if (!publisher) {
        throw std::runtime_error(
            fmt::format("Unable to attach the '{}' HoloMQ tensor-schema publisher: {}", port_name,
                        holoscan::runtime_error_message(publisher.error())));
      }
      return std::make_shared<holoscan::ChannelTensorSchemaPublisher<typename Traits::Payload>>(
          std::move(*publisher));
    } else {
      auto publisher = runtime.template create_publisher<HsdkT>(catalog, port_name);
      if (!publisher) {
        throw std::runtime_error(fmt::format("Unable to attach the '{}' HoloMQ publisher: {}",
                                             port_name,
                                             holoscan::runtime_error_message(publisher.error())));
      }
      return std::make_shared<holoscan::ChannelPublisher<HsdkT>>(std::move(*publisher));
    }
  }

  /// Makes the pool device current on the calling thread. A thread entry must not let an exception
  /// escape (that would terminate the process), so a failure is logged and reported to the caller,
  /// which stops its thread.
  /// @param what The thread or topic, for the log.
  [[nodiscard]] static bool use_pool_device_or_log(const std::string& what) {
    try {
      UsePoolDeviceOnThisThread();
      return true;
    } catch (const std::exception& error) {
      HOLOSCAN_LOG_ERROR("ROS2 bridge {} could not use the CUDA pool device and is stopping: {}",
                         what, error.what());
      return false;
    }
  }

  /// Warns once per topic when a topic declared with a GPU publisher (the default) receives a host
  /// buffer: that publisher sends host memory, and every frame is slower than it needs to be until
  /// the spec says so. The reverse mistake, a GPU publisher declared as host, cannot be seen from
  /// here because the subscription then asks ROS for host buffers.
  template <typename MsgT>
  static void warn_if_host_buffer_on_cuda_publisher_topic(const Ros2TopicSpec& spec,
                                                          [[maybe_unused]] const MsgT& msg,
                                                          std::atomic<bool>& warned) {
#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
    if constexpr (requires { msg.data.get_backend_type(); }) {
      if (AcceptableBufferBackends(spec) == std::string_view("cuda") &&
          msg.data.get_backend_type() == "cpu" && !warned.exchange(true)) {
        HOLOSCAN_LOG_WARN(
            "ROS2 topic '{}' feeds the device port '{}' and expects CUDA buffers from its "
            "publisher, but a host buffer arrived. Every frame is slower than it needs to be. If "
            "the publisher sends host memory, set publisher_memory = Ros2MemoryKind::kHost in the "
            "topic spec. If it should send GPU memory, check that the CUDA buffer backend is "
            "installed in the publisher's process.",
            spec.ros2_topic, spec.port_name);
      }
    }
#else
    (void)spec;
    (void)warned;
#endif
  }

  /// Creates the ROS2 subscription appropriate for Traits::HsdkType, whose callback converts
  /// each received message (Traits::to_hsdk/to_tensors) and relays it via relay_ros2_to_holoscan.
  /// error_count is a shared_ptr<int>, not a plain int, so the callback lambda can stay
  /// non-mutable (rclcpp can't deduce a mutable lambda's callback signature) while still letting
  /// warn_relay_dropped increment it across calls.
  /// @param node The ROS2 node the subscription is created on.
  /// @param spec The topic being relayed.
  /// @param shared_publisher The HoloMQ publisher each received message is relayed onto.
  /// @return A type-erased shared_ptr owning the subscription, to be kept alive by the caller.
  template <typename Traits, typename SharedPublisherPtr>
  [[nodiscard]] static std::shared_ptr<void> create_ingress_subscription(
      rclcpp::Node& node, const Ros2TopicSpec& spec, SharedPublisherPtr shared_publisher) {
    using Ros2T = typename Traits::Ros2Type;
    using HsdkT = typename Traits::HsdkType;
    auto error_count = std::make_shared<int>(0);
    if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<HsdkT>) {
      rclcpp::SubscriptionOptions subscription_options;
#ifdef ISAAC_ROS2_BRIDGE_CUDA_BUFFER
      // The default ("cpu") makes the transport convert a CUDA message to host memory before the
      // callback runs, so a device topic whose publisher sends GPU memory must opt in. Asking for
      // CUDA buffers also slows the receive of a host publisher's frames, so a publisher declared
      // as host memory keeps the default.
      subscription_options.acceptable_buffer_backends = AcceptableBufferBackends(spec);
#endif
      auto warned_about_host_buffer = std::make_shared<std::atomic<bool>>(false);
      return node.template create_subscription<Ros2T>(
          spec.ros2_topic, spec.qos,
          [shared_publisher, spec, error_count, warned_about_host_buffer,
           node_ptr = &node](const std::shared_ptr<Ros2T> msg) {
            warn_if_host_buffer_on_cuda_publisher_topic(spec, *msg, *warned_about_host_buffer);
            auto built = Traits::to_tensors(msg, spec);
            if (!built) {
              warn_relay_dropped(RelayDirection::kRos2ToHolo, spec.ros2_topic, spec.port_name,
                                 *error_count, built.error());
              return;
            }
            const auto capture_ns = capture_ns_for(spec, *msg, *node_ptr);
            if (!capture_ns) {
              warn_relay_dropped(RelayDirection::kRos2ToHolo, spec.ros2_topic, spec.port_name,
                                 *error_count, capture_ns.error());
              return;
            }
            auto& [metadata_bytes, tensors] = *built;
            relay_ros2_to_holoscan(shared_publisher, spec.ros2_topic, spec.port_name, error_count,
                                   *capture_ns, metadata_bytes, tensors);
          },
          subscription_options);
    } else {
      return node.template create_subscription<Ros2T>(
          spec.ros2_topic, spec.qos,
          [shared_publisher, spec, error_count,
           node_ptr = &node](const std::shared_ptr<Ros2T> msg) {
            const auto capture_ns = capture_ns_for(spec, *msg, *node_ptr);
            if (!capture_ns) {
              warn_relay_dropped(RelayDirection::kRos2ToHolo, spec.ros2_topic, spec.port_name,
                                 *error_count, capture_ns.error());
              return;
            }
            relay_ros2_to_holoscan(shared_publisher, spec.ros2_topic, spec.port_name, error_count,
                                   *capture_ns, Traits::to_hsdk(*msg));
          });
    }
  }

  /// Attaches the HoloMQ subscriber appropriate for Traits::HsdkType (tensor-schema or plain)
  /// and wraps it in the shared_ptr the poll thread needs.
  /// @param runtime The bridge's ChannelRuntime, used to attach the subscriber.
  /// @param catalog Channel endpoint catalog identifying this topic's endpoint.
  /// @param port_name Port name to attach the subscriber for.
  /// @return A shared_ptr to the attached ChannelTensorSchemaSubscriber<Payload> or
  /// ChannelSubscriber<HsdkT>, depending on Traits::HsdkType.
  /// @throws std::runtime_error If the attach call fails.
  template <typename Traits>
  [[nodiscard]] static auto attach_egress_subscriber(
      holoscan::ChannelRuntime& runtime, const holoscan::ChannelEndpointCatalog& catalog,
      const std::string& port_name) {
    using HsdkT = typename Traits::HsdkType;
    if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<HsdkT>) {
      auto subscriber = runtime.template create_tensor_schema_subscriber<typename Traits::Payload>(
          catalog, port_name);
      if (!subscriber) {
        throw std::runtime_error(
            fmt::format("Unable to attach the '{}' HoloMQ tensor-schema subscriber: {}", port_name,
                        holoscan::runtime_error_message(subscriber.error())));
      }
      return std::make_shared<holoscan::ChannelTensorSchemaSubscriber<typename Traits::Payload>>(
          std::move(*subscriber));
    } else {
      auto subscriber = runtime.template create_subscriber<HsdkT>(catalog, port_name);
      if (!subscriber) {
        throw std::runtime_error(fmt::format("Unable to attach the '{}' HoloMQ subscriber: {}",
                                             port_name,
                                             holoscan::runtime_error_message(subscriber.error())));
      }
      return std::make_shared<holoscan::ChannelSubscriber<HsdkT>>(std::move(*subscriber));
    }
  }

  /// Spawns the poll thread that runs poll_and_relay_holoscan_to_ros2 on a dedicated thread.
  /// @param channel_subscriber The HoloMQ subscriber the spawned thread will wait on and receive
  /// from.
  /// @param ros2_publisher The ROS2 publisher the spawned thread republishes onto.
  /// @param stop_token Passed through to poll_and_relay_holoscan_to_ros2 to stop the thread.
  /// @param spec The topic being relayed; its names are used for logging and the converter reads
  /// its memory settings.
  /// @return The spawned, joinable std::thread.
  template <typename Traits, typename ChannelSubscriberPtr>
  [[nodiscard]] static std::thread spawn_egress_poll_thread(
      ChannelSubscriberPtr channel_subscriber,
      typename rclcpp::Publisher<typename Traits::Ros2Type>::SharedPtr ros2_publisher,
      std::stop_token stop_token, Ros2TopicSpec spec) {
    using SubscriberT = typename ChannelSubscriberPtr::element_type;
    return std::thread(&poll_and_relay_holoscan_to_ros2<Traits, SubscriberT>,
                       std::move(channel_subscriber), std::move(ros2_publisher),
                       std::move(stop_token), std::move(spec));
  }

  std::string node_name_;
  // The one GPU every kCudaDevice topic uses (see the constructor).
  int pool_device_ = 0;
  rclcpp::NodeOptions node_options_;
  // True once any tensor topic asks for kCudaDevice: start() then binds the CUDA buffer pool.
  bool uses_cuda_pool_ = false;
  std::unordered_map<std::string, ImportedTopicVariant<ConverterT>> imported_;
  std::unordered_map<std::string, ExportedTopicVariant<ConverterT>> exported_;

  // Records the ros2_topic each port_name was wired to, alongside imported_/exported_ -- checked
  // by create_graph_topic() so re-wiring the same port_name to a *different* ros2_topic (a caller
  // bug the old config-list design couldn't even express, since a port_name's ros2_topic was
  // fixed once in config) throws instead of silently rebinding it.
  std::unordered_map<std::string, std::string> ingress_ros2_topics_;
  std::unordered_map<std::string, std::string> egress_ros2_topics_;

  // Built by from_ros2<Ros2T, HsdkT>; invoked once by start(). Returns the type-erased
  // subscription object that must be kept alive (same type create_ingress_subscription itself
  // returns).
  using IngressStarter = std::function<std::shared_ptr<void>(
      holoscan::ChannelRuntime&, const holoscan::ChannelEndpointCatalog&, rclcpp::Node&)>;
  // Built by to_ros2<Ros2T, HsdkT>; invoked once by start() with a stop_token from a
  // std::stop_source start() owns. Returns the spawned, joinable poll thread (same as
  // spawn_egress_poll_thread itself returns).
  using EgressStarter =
      std::function<std::thread(holoscan::ChannelRuntime&, const holoscan::ChannelEndpointCatalog&,
                                rclcpp::Node&, std::stop_token)>;
  std::unordered_map<std::string, IngressStarter> ingress_starters_;
  std::unordered_map<std::string, EgressStarter> egress_starters_;

  // Engaged by start(), disengaged by stop() -- see RuntimeState's own comment above.
  std::optional<RuntimeState> runtime_state_;
};

}  // namespace isaac::os::ros2_bridge
