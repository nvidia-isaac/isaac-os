/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "holoscan/channel/endpoint_catalog.hpp"
#include "holoscan/channel/runtime.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/core/external_topic.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/schema/descriptor_flatbuffer_payload.hpp"

#include "common/mcap_replayer/message.h"

namespace isaac::os::mcap_replayer {

// Type-erased so the replayer holds one port per topic in a single collection. import_into() and
// attach() are separate because a Graph exists only at compose time and the endpoint catalog only
// after compile() -- see McapReplayer::attach() and McapReplayer::start().
struct PortOwner {
  virtual ~PortOwner() = default;

  // Declares this topic on the graph under endpoint. Must run during the graph's compose(), like
  // Graph::import_topic(). No-op if already imported, so a repeated attach() reaches compile's
  // fan-out check instead of failing on a duplicate endpoint.
  virtual void import_into(holoscan::Graph& graph, std::string_view endpoint) = 0;

  // Creates the publisher for the endpoint import_into() declared. Throws std::runtime_error on
  // failure.
  virtual void attach(holoscan::ChannelRuntime& runtime,
                      const holoscan::ChannelEndpointCatalog& catalog,
                      std::string_view endpoint) = 0;

  // Publishes message (made by this port's builder, see decoder.h), consuming its contents.
  // Returns kNotReady if attach() has not run.
  [[nodiscard]] virtual holoscan::expected<void, holoscan::Error> emit(Message& message) = 0;

  // Refreshes the publisher's view of subscribers. A failure while the consuming graph is still
  // coming up means "not ready yet".
  [[nodiscard]] virtual holoscan::expected<void, holoscan::Error> update_connections() noexcept = 0;

  // Whether at least one subscriber was seen as of the last update_connections().
  [[nodiscard]] virtual bool has_subscribers() const noexcept = 0;
};

// What the two ports share. A port supplies only how to import its topic, create its publisher
// and publish a message; the state and the checks around them are here once.
template <typename T, typename Publisher>
class PortBase : public PortOwner {
 public:
  void import_into(holoscan::Graph& graph, std::string_view endpoint) final {
    if (imported_) {
      return;
    }
    imported_ = import_topic(graph, std::string(endpoint));
  }

  void attach(holoscan::ChannelRuntime& runtime, const holoscan::ChannelEndpointCatalog& catalog,
              std::string_view endpoint) final {
    auto created = create_publisher(runtime, catalog, endpoint);
    if (!created) {
      throw std::runtime_error(
          "Unable to attach the publisher for endpoint '" + std::string(endpoint) +
          "': " + std::string(holoscan::runtime_error_message(created.error())) + ".");
    }
    publisher_ = std::move(*created);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit(Message& message) final {
    if (!publisher_.has_value()) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotReady});
    }
    return publish(*publisher_, message);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> update_connections() noexcept final {
    if (!publisher_.has_value()) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotReady});
    }
    return publisher_->update_connections();
  }

  [[nodiscard]] bool has_subscribers() const noexcept final {
    return publisher_.has_value() && publisher_->has_subscribers();
  }

  [[nodiscard]] holoscan::Output<T>& output() { return imported_->output(); }

 private:
  virtual holoscan::ImportedTopic<T> import_topic(holoscan::Graph& graph,
                                                  const std::string& endpoint) = 0;

  [[nodiscard]] virtual holoscan::expected<Publisher, holoscan::Error> create_publisher(
      holoscan::ChannelRuntime& runtime, const holoscan::ChannelEndpointCatalog& catalog,
      std::string_view endpoint) = 0;

  [[nodiscard]] virtual holoscan::expected<void, holoscan::Error> publish(Publisher& publisher,
                                                                          Message& message) = 0;

  // Optional because ImportedTopic<T> has no public default constructor.
  std::optional<holoscan::ImportedTopic<T>> imported_;
  std::optional<Publisher> publisher_;
};

// The plain (non-tensor-schema) case: publishes materialized T values.
template <typename T>
class TypedPort final : public PortBase<T, holoscan::ChannelPublisher<T>> {
 private:
  holoscan::ImportedTopic<T> import_topic(holoscan::Graph& graph,
                                          const std::string& endpoint) override {
    return graph.import_topic<T>("holomq://" + endpoint,
                                 holoscan::TopicOptions{.endpoint_name = endpoint});
  }

  [[nodiscard]] holoscan::expected<holoscan::ChannelPublisher<T>, holoscan::Error> create_publisher(
      holoscan::ChannelRuntime& runtime, const holoscan::ChannelEndpointCatalog& catalog,
      std::string_view endpoint) override {
    return runtime.create_publisher<T>(catalog, endpoint);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> publish(
      holoscan::ChannelPublisher<T>& publisher, Message& message) override {
    assert(dynamic_cast<FlatbufferMessage<T>*>(&message) != nullptr);
    auto sent = publisher.emit(std::move(static_cast<FlatbufferMessage<T>&>(message).value));
    if (!sent) {
      return holoscan::make_unexpected(sent.error());
    }
    return {};
  }
};

namespace detail {

// The payload type the SDK derives from T's root type.
template <typename T>
using TensorSchemaPayload = holoscan::schema::DescriptorFlatBufferPayloadFor<
    typename holoscan::DescriptorTensorSchemaNativeTraits<T>::root_type>;

}  // namespace detail

// The tensor-schema case: publishes the recorded metadata bytes and rebuilt tensors as they are,
// bypassing T (see decoder.h). Importing needs the caller's DescriptorTensorTopicContract<T>
// (from Topic<T>), which import_into() hands to the graph.
template <typename T>
  requires holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>
class TypedTensorPort final
    : public PortBase<T, holoscan::ChannelTensorSchemaPublisher<detail::TensorSchemaPayload<T>>> {
 public:
  explicit TypedTensorPort(holoscan::DescriptorTensorTopicContract<T> topic_contract)
      : contract_(std::move(topic_contract)) {}

 private:
  using Payload = detail::TensorSchemaPayload<T>;

  holoscan::ImportedTopic<T> import_topic(holoscan::Graph& graph,
                                          const std::string& endpoint) override {
    return graph.import_topic<T>("holomq://" + endpoint,
                                 holoscan::TopicOptions{.endpoint_name = endpoint},
                                 std::move(contract_));
  }

  [[nodiscard]] holoscan::expected<holoscan::ChannelTensorSchemaPublisher<Payload>, holoscan::Error>
  create_publisher(holoscan::ChannelRuntime& runtime,
                   const holoscan::ChannelEndpointCatalog& catalog,
                   std::string_view endpoint) override {
    return runtime.create_tensor_schema_publisher<Payload>(catalog, endpoint);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> publish(
      holoscan::ChannelTensorSchemaPublisher<Payload>& publisher, Message& message) override {
    assert(dynamic_cast<TensorMessage*>(&message) != nullptr);
    auto& tensor_message = static_cast<TensorMessage&>(message);
    auto sent = publisher.emit(tensor_message.metadata, tensor_message.tensors);
    if (!sent) {
      return holoscan::make_unexpected(sent.error());
    }
    return {};
  }

  holoscan::DescriptorTensorTopicContract<T> contract_;
};

}  // namespace isaac::os::mcap_replayer
