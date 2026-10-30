/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_UCXX_OPS_UCXX_SENDER_OP_H
#define ISAAC_OS_OPERATORS_UCXX_OPS_UCXX_SENDER_OP_H

#include <cstdint>
#include <list>
#include <memory>
#include <thread>
#include <utility>

#include "flatbuffers/flatbuffers.h"
#include "ucxx/api.h"
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/schema/tensor_materialization.hpp>

#include "operators/ucxx_ops/message_traits.h"
#include "operators/ucxx_ops/ucxx_endpoint.h"

namespace isaac::os::ops {

/// Encodes graph payload `T` and sends it through a shared UcxxEndpoint.
///
/// One instantiation per message type. Consumers use the type aliases (see
/// `ucxx_type_aliases.h`) or instantiate directly:
///
///     const auto joint_command_sender =
///         graph.op<UcxxSenderOp<isaac::JointCommandCpuT>>(
///             "joint_command_sender", endpoint, /*tag=*/0x3ad708d3ull);
///
/// Port types use existing CPU joint or SDK sensor schema identities.
/// Serialization goes directly
/// through `Traits::encode(builder, native, timestamp)` — no runtime
/// `MessageReflection` dispatch. Messages carrying a device-memory tensor
/// (currently only `ImageT`) get a second `tagSend` on the SAME tag after
/// the FlatBuffer metadata to preserve zero-copy behavior on the network
/// fast path. UCX's per-tag in-order delivery guarantee is what lets the
/// receiver post metadata `tagRecv` then tensor `tagRecv` predictably; the
/// convention matches Isaac Sim's OgnUCXPublishImage GPU-direct path.
template <typename T>
class UcxxSenderOp final : public holoscan::Operator<> {
 public:
  using Traits = UcxxMessageTraits<T>;
  using RootType = typename Traits::RootType;
  using NativeType = typename Traits::NativeType;

  UcxxSenderOp(UcxxEndpoint endpoint, std::uint64_t tag)
      : endpoint_(std::move(endpoint)), tag_(tag) {}

  void setup(holoscan::OperatorSpec& spec) override {
    auto& input = spec.input(in_message, "in_message");
    if constexpr (Traits::kHasSeparateTensor) {
      input.expects_tensor(holoscan::TensorInputSpec{
          .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice}});
    }
    // kStop must drain outstanding tagSends before requests_ is destroyed:
    // each SendRequest owns the FlatBufferBuilder (metadata payload) and,
    // for kHasSeparateTensor, a shared_ptr<holoscan::Tensor> pinning the
    // pixel bytes. Letting the destructor tear those down while UCX is
    // still reading from the wire would race a live tagSend against
    // released storage.
    spec.lifecycle().stage(holoscan::LifecycleStage::kStop, &UcxxSenderOp::on_stop);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach{in_message});
    return contract;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto received = in_message.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    NativeType message = std::move(received->data);

    auto endpoint = endpoint_.endpoint();
    if (!endpoint) {
      // Either the peer never connected or it dropped mid-run
      // (on_endpoint_closed cleared state). UcxxEndpoint offers no
      // reconnect path -- surface as fatal so the session terminates
      // non-zero instead of running forever producing nothing.
      HOLOSCAN_LOG_ERROR("UCX sender: endpoint not connected; terminating.");
      return holoscan::make_unexpected(holoscan::Error{
          .code = holoscan::ErrorCode::kFailure, .message = "UCX sender: endpoint not connected."});
    }

    // Reap completed sends first, so a well-behaved peer keeps requests_
    // bounded at zero and each new compute() sees a fresh queue. Doing this
    // before enqueuing the new send is what makes the bounded-queue check
    // below useful -- without it, we would grow the queue with the new send
    // and then reap, and the cap would only ever trigger after an over-cap
    // spike.
    reap_completed_requests();

    // Backpressure: if the peer or network is slow enough that
    // requests_.size() reaches this cap even after reaping, drop the new
    // message and log. This keeps requests_ from growing without bound
    // (each entry retains a FlatBufferBuilder and, for ImageT, a device-
    // memory reference on the outgoing tensor). Dropping is preferable to
    // blocking here because compute() runs on a scheduler thread; a block
    // would stall the receiver -> sinus -> sender pipeline as a whole.
    if (requests_.size() >= kMaxInflightSends) {
      HOLOSCAN_LOG_ERROR(
          "UCX sender: {} in-flight requests reached cap ({}); dropping "
          "message. Peer or network is slower than the input rate.",
          requests_.size(), kMaxInflightSends);
      return {};
    }

    SendRequest& send = requests_.emplace_back();
    // Materialize small control tensors into the FlatBuffer while keeping
    // large image tensors metadata-only for the sibling UCX transfer below.
    // This uses HSDK5's canonical Tensor codec and its size-based gather mode.
    if constexpr (Traits::kHasSeparateTensor) {
      // Only ImageT currently trips this branch (see message_traits_inl.h).
      // The receiver posts one metadata tagRecv followed by one tensor
      // tagRecv per frame, keyed on the shared per-tag FIFO. If we send
      // the metadata half but omit the tensor half, the receiver blocks on
      // the missing tensor forever and then splices the next frame's
      // metadata into this frame's tensor slot -- a permanent desync. Drop
      // the whole frame here instead: pop the placeholder we already
      // reserved and log so the input rate mismatch is visible.
      if (!message.data || message.data->nbytes() == 0) {
        HOLOSCAN_LOG_ERROR(
            "UCX sender: kHasSeparateTensor message has no tensor payload; "
            "dropping frame to preserve metadata+tensor cadence on tag {}.",
            tag_);
        requests_.pop_back();
        return {};
      }
    }

    // Image pixels always travel on the second UCX transfer, never inline.
    holoscan::schema::WithTensorGather gather_scope(0U);
    auto offset = Traits::encode(send.flatbuffer_builder, message,
                                 received->metadata.capture_time().timestamp_ns);
    if constexpr (requires { Traits::kFileIdentifier; }) {
      send.flatbuffer_builder.Finish(offset, Traits::kFileIdentifier);
    } else {
      send.flatbuffer_builder.Finish(offset);
    }

    send.request = endpoint->tagSend(send.flatbuffer_builder.GetBufferPointer(),
                                     send.flatbuffer_builder.GetSize(), ucxx::Tag{tag_});

    if constexpr (Traits::kHasSeparateTensor) {
      // `ImageT::data` is a `std::shared_ptr<holoscan::Tensor>`. Ship its
      // raw byte payload on a second UCX tag so the pixel bytes are not
      // round-tripped through FlatBuffers metadata; UCX picks cuda_ipc
      // (zero-copy) when the peer is on the same host.
      //
      // Pin the tensor shared_ptr in the SendRequest so it outlives this
      // compute() call: `message` goes out of scope at the end of the
      // function, and if nothing upstream still holds message.data the
      // RawBufferMemoryReference would cudaFree the pixel bytes while the
      // async tagSend is still reading them. reap_completed_requests()
      // is what actually releases the SendRequest (and therefore this
      // shared_ptr) once ucxx marks the request complete.
      SendRequest& tensor_send = requests_.emplace_back();
      tensor_send.tensor = message.data;
      tensor_send.request = endpoint->tagSend(
          const_cast<void*>(tensor_send.tensor->data()),
          static_cast<std::size_t>(tensor_send.tensor->nbytes()), ucxx::Tag{tag_});
    }

    // Also reap after the just-emplaced send (unlikely to be already
    // complete, but cheap to check and keeps the tail predictable).
    reap_completed_requests();

    return {};
  }

  holoscan::Input<NativeType> in_message;

 private:
  holoscan::LifecycleStatus on_stop(holoscan::LifecycleContext&) noexcept {
    // ucxx::Request::cancel() is asynchronous (see UcxxReceiverOp::on_stop
    // for the long form): the UCX progress thread can still be reading the
    // FlatBufferBuilder / tensor storage until the request completes with
    // UCS_ERR_CANCELED or UCS_OK. Cancel every in-flight send, spin to
    // completion, then let requests_ drop its owned payloads.
    for (auto& send : requests_) {
      if (send.request) {
        send.request->cancel();
      }
    }
    for (auto& send : requests_) {
      while (send.request && !send.request->isCompleted()) {
        std::this_thread::yield();
      }
    }
    requests_.clear();
    return holoscan::LifecycleStatus::kOk;
  }

  struct SendRequest {
    std::shared_ptr<ucxx::Request> request;
    flatbuffers::FlatBufferBuilder flatbuffer_builder;
    // Non-null only for the tensor half of an ImageT (or any future T with
    // kHasSeparateTensor). Pins the source pixel buffer alive for the
    // duration of the async tagSend; released when reap_completed_requests
    // erases this SendRequest.
    std::shared_ptr<holoscan::Tensor> tensor;
  };

  // Upper bound on in-flight tagSends. For ImageT this covers ~kMaxInflightSends/2
  // frames since each frame consumes two requests (metadata + tensor bytes).
  // 32 leaves comfortable headroom over normal ~30 fps camera + 200 Hz joint
  // command rates while still bounding memory in a stalled-peer scenario.
  static constexpr std::size_t kMaxInflightSends = 32U;

  void reap_completed_requests() {
    for (auto it = requests_.begin(); it != requests_.end();) {
      if (!it->request || !it->request->isCompleted()) {
        ++it;
        continue;
      }
      if (const ucs_status_t status = it->request->getStatus(); status != UCS_OK) {
        HOLOSCAN_LOG_ERROR("UCX tagSend failed: {}", ucs_status_string(status));
      }
      it = requests_.erase(it);
    }
  }

  UcxxEndpoint endpoint_;
  std::uint64_t tag_{};
  std::list<SendRequest> requests_;
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_UCXX_OPS_UCXX_SENDER_OP_H
