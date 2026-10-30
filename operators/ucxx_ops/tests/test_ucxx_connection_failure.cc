/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <ucxx/api.h>

#include "joint_state_cpu_schema_traits.h"
#include "operators/ucxx_ops/ucxx_connect_op.h"
#include "operators/ucxx_ops/ucxx_receiver_op.h"

namespace isaac::os::ops {
namespace {

class TestConnectionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// A real local UCX peer, with no simulator or message producer required.
class LocalPeer {
 public:
  explicit LocalPeer(int port = 0) {
    context_ = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
    worker_ = context_->createWorker();
    listener_ = worker_->createListener(
        port,
        +[](ucp_conn_request_h request,
            void* data) {  // NOSONAR: UCX's C callback ABI requires void* user data.
          auto& peer = *static_cast<LocalPeer*>(data);
          std::lock_guard lock(peer.mutex_);
          try {
            peer.endpoint_ = peer.listener_->createEndpointFromConnRequest(request, true);
          } catch (const std::exception& error) {
            peer.error_ = error.what();
          }
          peer.ready_.notify_one();
        },
        this);
    worker_->startProgressThread(false);
  }
  ~LocalPeer() { worker_->stopProgressThread(); }
  int port() const { return listener_->getPort(); }
  void wait_for_connection() {
    std::unique_lock lock(mutex_);
    if (!ready_.wait_for(lock, std::chrono::seconds{5},
                         [&] { return endpoint_ || !error_.empty(); }))
      throw TestConnectionError("The local test peer did not accept the connection in time.");
    if (!error_.empty()) throw TestConnectionError(error_);
  }

  void send(std::uint8_t* data, std::size_t size, std::uint64_t tag) {
    auto request = endpoint_->tagSend(data, size, ucxx::Tag{tag});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!request->isCompleted() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    if (!request->isCompleted() || request->getStatus() != UCS_OK)
      throw TestConnectionError("The local test peer could not send a message in time.");
  }

 private:
  std::mutex mutex_;
  std::condition_variable ready_;
  std::string error_;
  std::shared_ptr<ucxx::Context> context_;
  std::shared_ptr<ucxx::Worker> worker_;
  std::shared_ptr<ucxx::Listener> listener_;
  std::shared_ptr<ucxx::Endpoint> endpoint_;
};

struct ReceivedCount {
  std::shared_ptr<std::atomic<int>> count;
};

TEST(UcxxConnectionFailure, InitialConnectionWaitsForLatePeer) {
  auto shutdown = holoscan::ManagedShutdownSignals::create();
  ASSERT_TRUE(shutdown);
  setenv("UCX_TLS", "tcp,self", 1);
  int port;
  {
    LocalPeer reservation;
    port = reservation.port();
  }
  UcxxEndpoint endpoint{"127.0.0.1", port, false};
  std::atomic connected{false};
  auto callback = std::make_shared<UcxxEndpoint::ConnectCallback>(
      [&connected](const auto&) { connected.store(true); });
  endpoint.notify_when_connected(callback);
  holoscan::Graph graph{"late-peer"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto connector = graph.op<UcxxConnectOp>(
      "connect", endpoint, UcxxConnectOptions{.retry_interval = std::chrono::milliseconds{20}});
  graph.partition("connection").add(connector);
  const auto plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = holoscan::run_async(plan, *shutdown);
  std::this_thread::sleep_for(std::chrono::milliseconds{200});
  EXPECT_FALSE(connected.load());
  EXPECT_FALSE(endpoint.endpoint());
  LocalPeer peer{port};
  peer.wait_for_connection();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (!connected.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  EXPECT_TRUE(connected.load());
  session.request_stop();
  session.wait();
  EXPECT_NE(session.termination(), holoscan::RunTermination::kFailed);
}

TEST(UcxxConnectionFailure, InitialConnectionTimesOut) {
  auto shutdown = holoscan::ManagedShutdownSignals::create();
  ASSERT_TRUE(shutdown);
  setenv("UCX_TLS", "tcp,self", 1);
  int port;
  {
    LocalPeer reservation;
    port = reservation.port();
  }
  UcxxEndpoint endpoint{"127.0.0.1", port, false};
  holoscan::Graph graph{"connection-timeout"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto connector =
      graph.op<UcxxConnectOp>("connect", endpoint,
                              UcxxConnectOptions{.timeout = std::chrono::milliseconds{200},
                                                 .retry_interval = std::chrono::milliseconds{20}});
  graph.partition("connection").add(connector);
  const auto plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = holoscan::run_async(plan, *shutdown);
  session.wait();
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kFailed);
  EXPECT_FALSE(endpoint.endpoint());
  std::string diagnostics;
  for (const auto& diagnostic : session.diagnostics()) {
    diagnostics += holoscan::format_diagnostic(diagnostic, plan.graph());
  }
  EXPECT_NE(diagnostics.find("Timed out"), std::string::npos) << diagnostics;
}

TEST(UcxxConnectionFailure, StopInterruptsInitialConnectionWait) {
  auto shutdown = holoscan::ManagedShutdownSignals::create();
  ASSERT_TRUE(shutdown);
  setenv("UCX_TLS", "tcp,self", 1);
  int port;
  {
    LocalPeer reservation;
    port = reservation.port();
  }
  UcxxEndpoint endpoint{"127.0.0.1", port, false};
  holoscan::Graph graph{"cancel-connection"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto connector = graph.op<UcxxConnectOp>(
      "connect", endpoint, UcxxConnectOptions{.retry_interval = std::chrono::milliseconds{10000}});
  graph.partition("connection").add(connector);
  const auto plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = holoscan::run_async(plan, *shutdown);
  std::this_thread::sleep_for(std::chrono::milliseconds{100});
  const auto start = std::chrono::steady_clock::now();
  session.request_stop();
  session.wait();
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds{2});
  EXPECT_NE(session.termination(), holoscan::RunTermination::kFailed);
}

class SlowImageSinkOp final : public holoscan::Operator<> {
 public:
  explicit SlowImageSinkOp(ReceivedCount state) : count_(std::move(state.count)) {}
  holoscan::Input<holoscan::schema::ImageT> input;
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice}});
  }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto message = input.receive();
    if (!message) return holoscan::make_unexpected(message.error());
    // Keep the selected frame alive while new camera messages arrive.
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    ++*count_;
    return {};
  }

 private:
  std::shared_ptr<std::atomic<int>> count_;
};

TEST(UcxxConnectionFailure, RejectsInvalidPort) {
  EXPECT_THROW((UcxxEndpoint{"127.0.0.1", 0, false}), std::invalid_argument);
  EXPECT_THROW((UcxxEndpoint{"127.0.0.1", 65536, false}), std::invalid_argument);
  EXPECT_THROW((UcxxEndpoint{"127.0.0.1", -1, false}), std::invalid_argument);
}

TEST(UcxxConnectionFailure, SlowCameraConsumerDoesNotFailSession) {
  auto shutdown = holoscan::ManagedShutdownSignals::create();
  ASSERT_TRUE(shutdown);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  setenv("UCX_TLS", "tcp,cuda_copy,self", 1);
  setenv("UCX_CM_USE_ALL_DEVICES", "n", 1);
  LocalPeer peer;
  UcxxEndpoint endpoint{"127.0.0.1", peer.port(), false};
  holoscan::Graph graph{"slow-camera-consumer"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  auto connector = graph.op<UcxxConnectOp>("ucx_connect", endpoint);
  auto receiver = graph.op<UcxxReceiverOp<holoscan::schema::ImageT>>(
      "image_rx", endpoint, 1234UL, 65536UL, 12UL, TimestampSource::kGraphClock);
  auto count = std::make_shared<std::atomic<int>>(0);
  auto sink = graph.op<SlowImageSinkOp>("slow_image_sink", ReceivedCount{count});
  graph.add_flow(receiver->out_message, sink->input,
                 {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.partition("connection").add(connector);
  graph.partition("receiver").add(receiver);
  graph.partition("sink").add(sink);
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      {.operator_path = "image_rx", .output_port = "out_message", .device = holoscan::DeviceId{0}});
  auto plan = holoscan::compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();

  std::uint8_t* pixels = nullptr;
  ASSERT_EQ(cudaMalloc(&pixels, 12), cudaSuccess);
  auto pixels_owner = std::shared_ptr<std::uint8_t>(pixels, [](std::uint8_t* p) { cudaFree(p); });
  ASSERT_EQ(cudaMemset(pixels, 17, 12), cudaSuccess);
  std::array<std::int64_t, 3> shape{2, 2, 3};
  DLManagedTensor storage{};
  storage.dl_tensor = {.data = pixels,
                       .device = {kDLCUDA, 0},
                       .ndim = 3,
                       .dtype = {kDLUInt, 8, 1},
                       .shape = shape.data(),
                       .strides = nullptr,
                       .byte_offset = 0};
  holoscan::schema::ImageT message;
  message.width = 2;
  message.height = 2;
  message.encoding = holoscan::schema::ImageEncoding_RGB8;
  message.data = std::make_shared<holoscan::Tensor>(&storage);
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(UcxxMessageTraits<holoscan::schema::ImageT>::encode(builder, message, 0));
  auto session = holoscan::run_async(plan, *shutdown);
  peer.wait_for_connection();
  for (int i = 0; i < 32 && session.is_running(); ++i) {
    peer.send(builder.GetBufferPointer(), builder.GetSize(), 1234);
    peer.send(pixels, 12, 1234);
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
  }
  std::this_thread::sleep_for(std::chrono::milliseconds{200});
  session.request_stop();
  session.wait();
  std::string diagnostics;
  for (const auto& diagnostic : session.diagnostics()) {
    diagnostics += holoscan::format_diagnostic(diagnostic, plan.graph());
  }
  EXPECT_NE(session.termination(), holoscan::RunTermination::kFailed) << diagnostics;
  EXPECT_GT(count->load(), 0);
}

class SlowImuSinkOp final : public holoscan::Operator<> {
 public:
  explicit SlowImuSinkOp(ReceivedCount state) : count_(std::move(state.count)) {}
  holoscan::Input<holoscan::schema::ImuT> input;
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input"); }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{20}});
    result.input(input, holoscan::LatestWithin{.max_age = std::chrono::milliseconds{25}});
    result.on_missing(input, holoscan::Skip{});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto message = input.receive();
    if (!message) return holoscan::make_unexpected(message.error());
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    ++*count_;
    return {};
  }

 private:
  std::shared_ptr<std::atomic<int>> count_;
};

void check_imu_burst(holoscan::QueuePolicy policy) {
  auto shutdown = holoscan::ManagedShutdownSignals::create();
  ASSERT_TRUE(shutdown);
  SCOPED_TRACE(static_cast<int>(policy));
  setenv("UCX_TLS", "tcp,self", 1);
  setenv("UCX_CM_USE_ALL_DEVICES", "n", 1);
  LocalPeer peer;
  UcxxEndpoint endpoint{"127.0.0.1", peer.port(), false};
  holoscan::Graph graph{"imu-burst"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  auto connector = graph.op<UcxxConnectOp>("ucx_connect", endpoint);
  auto receiver = graph.op<UcxxReceiverOp<holoscan::schema::ImuT>>(
      "imu_rx", endpoint, 1234UL, 65536UL, 0UL, TimestampSource::kGraphClock);
  auto count = std::make_shared<std::atomic<int>>(0);
  auto sink = graph.op<SlowImuSinkOp>("slow_imu_sink", ReceivedCount{count});
  graph.add_flow(receiver->out_message, sink->input, {.queue_policy = policy, .queue_depth = 2U});
  graph.partition("connection").add(connector);
  graph.partition("receiver").add(receiver);
  graph.partition("sink").add(sink);
  auto plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(UcxxMessageTraits<holoscan::schema::ImuT>::encode(builder, {}, 0));
  auto session = holoscan::run_async(plan, *shutdown);
  peer.wait_for_connection();
  // Clock ticks without IMU input must not invoke the consumer.
  std::this_thread::sleep_for(std::chrono::milliseconds{60});
  EXPECT_EQ(count->load(), 0);
  for (int i = 0; i < 32 && session.is_running(); ++i) {
    peer.send(builder.GetBufferPointer(), builder.GetSize(), 1234);
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
  }
  if (policy == holoscan::QueuePolicy::kDropOldest) {
    // Allow in-flight callbacks to finish and the retained sample to expire.
    std::this_thread::sleep_for(std::chrono::milliseconds{250});
    const auto after_burst = count->load();
    EXPECT_GT(after_burst, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    EXPECT_EQ(count->load(), after_burst) << "Expired IMU samples must not drive control.";
    // New samples must resume processing after the missing-input interval.
    for (int i = 0; i < 32 && session.is_running(); ++i) {
      peer.send(builder.GetBufferPointer(), builder.GetSize(), 1234);
      std::this_thread::sleep_for(std::chrono::milliseconds{5});
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    EXPECT_GT(count->load(), after_burst);
  }
  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination() == holoscan::RunTermination::kFailed,
            policy == holoscan::QueuePolicy::kBlocking);
  if (policy == holoscan::QueuePolicy::kDropOldest) EXPECT_GT(count->load(), 0);
}

TEST(UcxxConnectionFailure, ImuBurstBlockingFails) {
  check_imu_burst(holoscan::QueuePolicy::kBlocking);
}

TEST(UcxxConnectionFailure, ImuBurstDropOldestContinues) {
  check_imu_burst(holoscan::QueuePolicy::kDropOldest);
}

TEST(UcxxConnectionFailure, DeferredReceiverFailureTerminatesSession) {
  // Acquire signal ownership before CUDA or UCX can create helper threads.
  auto shutdown = holoscan::ManagedShutdownSignals::create();
  ASSERT_TRUE(shutdown);
  int devices = 0;
  ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
  ASSERT_GT(devices, 0);
  setenv("UCX_TLS", "tcp,cuda_copy,self", 1);
  setenv("UCX_CM_USE_ALL_DEVICES", "n", 1);
  LocalPeer peer;
  UcxxEndpoint endpoint{"127.0.0.1", peer.port(), false};
  ASSERT_EQ(endpoint.endpoint(), nullptr);
  // Finish peer acceptance before deliberately failing and tearing down the client.
  bool peer_accepted = false;
  auto accepted = std::make_shared<UcxxEndpoint::ConnectCallback>([&](const auto&) {
    peer.wait_for_connection();
    peer_accepted = true;
  });
  endpoint.notify_when_connected(accepted);

  holoscan::Graph graph{"deferred-receiver-failure"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  auto connector = graph.op<UcxxConnectOp>("ucx_connect", endpoint);
  // Exercise the actual receiver callback's error return -> throw path without
  // exhausting GPU memory: SIZE_MAX cannot be allocated on any supported GPU.
  auto receiver = graph.op<UcxxReceiverOp<holoscan::schema::ImageT>>(
      "image_rx", endpoint, 1234UL, 65536UL, std::numeric_limits<std::size_t>::max());
  graph.partition("connection").add(connector);
  graph.partition("image_transport").add(receiver);
  // Keep the SDK output pool small so the deliberate allocation failure happens
  // in the deferred receive callback, not during runtime pool initialization.
  graph.configure(receiver->out_message)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice},
          .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = 12U},
          .storage = holoscan::TensorOutputStorage::kRuntimePool});
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      {.operator_path = "image_rx", .output_port = "out_message", .device = holoscan::DeviceId{0}});
  const auto plan = holoscan::compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = holoscan::run_async(plan, *shutdown);
  // CTest's timeout makes a stranded receiver / non-terminating session fail.
  session.wait();
  EXPECT_TRUE(peer_accepted);
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kFailed);
  std::string diagnostics;
  for (const auto& diagnostic : session.diagnostics()) {
    diagnostics += holoscan::format_diagnostic(diagnostic, plan.graph());
  }
  EXPECT_NE(diagnostics.find("RUNTIME_OPERATOR_CALLBACK_FAILED"), std::string::npos) << diagnostics;
  EXPECT_NE(diagnostics.find("ucx_connect"), std::string::npos) << diagnostics;
}

TEST(UcxxConnectionFailure, SkippedConnectionCallbackDoesNotPreventSessionFailure) {
  auto shutdown = holoscan::ManagedShutdownSignals::create();
  ASSERT_TRUE(shutdown);
  setenv("UCX_TLS", "tcp,self", 1);
  setenv("UCX_CM_USE_ALL_DEVICES", "n", 1);
  LocalPeer peer;
  UcxxEndpoint endpoint{"127.0.0.1", peer.port(), false};
  auto accepted = std::make_shared<UcxxEndpoint::ConnectCallback>(
      [&](const auto&) { peer.wait_for_connection(); });
  endpoint.notify_when_connected(accepted);
  bool first_called = false;
  bool second_called = false;
  auto first = std::make_shared<UcxxEndpoint::ConnectCallback>([&first_called](const auto&) {
    first_called = true;
    throw TestConnectionError("The test receiver failed to initialize.");
  });
  auto second = std::make_shared<UcxxEndpoint::ConnectCallback>(
      [&second_called](const auto&) { second_called = true; });
  endpoint.notify_when_connected(first);
  endpoint.notify_when_connected(second);
  EXPECT_FALSE(first_called);
  EXPECT_FALSE(second_called);

  holoscan::Graph graph{"connection-callback-failure"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  auto connector = graph.op<UcxxConnectOp>("ucx_connect", endpoint);
  graph.partition("connection").add(connector);
  const auto plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = holoscan::run_async(plan, *shutdown);
  session.wait();
  EXPECT_TRUE(first_called);
  EXPECT_FALSE(second_called);
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kFailed);
}

struct ReceivedFrameIds {
  std::mutex mutex;
  std::vector<std::uint64_t> values;
};

class FrameIdSinkOp final : public holoscan::Operator<> {
 public:
  explicit FrameIdSinkOp(std::shared_ptr<ReceivedFrameIds> received)
      : received_(std::move(received)) {}
  holoscan::Input<isaac::JointStateCpuT> input;
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input"); }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto message = input.receive();
    if (!message) return holoscan::make_unexpected(message.error());
    std::lock_guard lock(received_->mutex);
    received_->values.push_back(message->metadata.frame_id);
    return {};
  }

 private:
  std::shared_ptr<ReceivedFrameIds> received_;
};

TEST(UcxxReceiver, GraphClockKeepsWireTimestampAsFrameId) {
  auto shutdown = holoscan::ManagedShutdownSignals::create();
  ASSERT_TRUE(shutdown);
  setenv("UCX_TLS", "tcp,self", 1);
  setenv("UCX_CM_USE_ALL_DEVICES", "n", 1);
  LocalPeer peer;
  UcxxEndpoint endpoint{"127.0.0.1", peer.port(), false};
  holoscan::Graph graph{"joint-state-frame-id"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  auto connector = graph.op<UcxxConnectOp>("ucx_connect", endpoint);
  auto receiver = graph.op<UcxxReceiverOp<isaac::JointStateCpuT>>(
      "joint_state_rx", endpoint, 1234UL, 65536UL, 0UL, TimestampSource::kGraphClock);
  auto received = std::make_shared<ReceivedFrameIds>();
  auto sink = graph.op<FrameIdSinkOp>("frame_id_sink", received);
  graph.add_flow(receiver->out_message, sink->input, {.queue_depth = 4U});
  graph.partition("connection").add(connector);
  graph.partition("receiver").add(receiver);
  graph.partition("sink").add(sink);
  auto plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = holoscan::run_async(plan, *shutdown);
  peer.wait_for_connection();
  // Let the receiver post its first receive before the peer sends.
  std::this_thread::sleep_for(std::chrono::milliseconds{60});
  isaac::JointStateCpuT state;
  state.position = {0.1F};
  state.velocity = {0.0F};
  for (const std::int64_t wire_time : {1000, 1000, 2000}) {
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(UcxxMessageTraits<isaac::JointStateCpuT>::encode(builder, state, wire_time));
    peer.send(builder.GetBufferPointer(), builder.GetSize(), 1234);
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
  }
  std::this_thread::sleep_for(std::chrono::milliseconds{200});
  session.request_stop();
  session.wait();
  std::lock_guard lock(received->mutex);
  EXPECT_EQ(received->values, (std::vector<std::uint64_t>{1000U, 1000U, 2000U}));
}

}  // namespace
}  // namespace isaac::os::ops
