/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_OPERATORS_UCXX_OPS_UCXX_RECEIVER_OP_H
#define ISAAC_OS_OPERATORS_UCXX_OPS_UCXX_RECEIVER_OP_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "ucxx/api.h"
#include <cuda_runtime.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>

#include "operators/ucxx_ops/message_traits.h"
#include "operators/ucxx_ops/ucxx_endpoint.h"

namespace isaac::os {

/// Source of capture-time metadata for received UCX messages.
enum class TimestampSource {
  kGraphClock,  ///< Use the graph clock when the receiver processes the message; a nonzero
                ///< wire timestamp becomes the sample's frame ID.
  kMessage,     ///< Use the wire timestamp, which must share the graph's clock domain.
};

}  // namespace isaac::os

namespace isaac::os::ops {

/// Receives wire messages and decodes graph payload `T` through a shared
/// UcxxEndpoint. One instantiation per message type.
///
/// Async completion signalling uses HSDK5's `NotificationSource`: the UCX
/// `tagRecv` completion callback posts on a `NotificationSender` acquired at
/// `kArm`, which the runtime turns into a `compute()` activation via
/// `OnNotified{.event = recv_notification_}` in the operator's `Contract`.
///
/// For `ImageT` (`kHasSeparateTensor == true`), a second `tagRecv` for the
/// raw device-memory bytes is posted on the SAME tag as the FlatBuffer
/// metadata. UCX guarantees in-order delivery per tag, so the receiver
/// consistently sees metadata first, then tensor -- matching Isaac Sim's
/// OgnUCXPublishImage GPU-direct convention (see
/// isaacsim.ucx.nodes/nodes/OgnUCXPublishImage.cpp).
template <typename T>
class UcxxReceiverOp final : public holoscan::Operator<> {
 public:
  using Traits = UcxxMessageTraits<T>;
  using RootType = typename Traits::RootType;
  using NativeType = typename Traits::NativeType;

  UcxxReceiverOp(UcxxEndpoint endpoint, std::uint64_t tag, std::size_t flatbuffer_buffer_size,
                 std::size_t tensor_buffer_size = 0,
                 TimestampSource timestamp_source = TimestampSource::kMessage)
      : endpoint_(std::move(endpoint)),
        tag_(tag),
        flatbuffer_buffer_(flatbuffer_buffer_size),
        tensor_buffer_size_(tensor_buffer_size),
        timestamp_source_(timestamp_source) {
    if constexpr (Traits::kHasSeparateTensor) {
      // Fail at construction: T carries a sibling tensor stream but the
      // caller reserved no device-side receive buffer for it. Silently
      // skipping the tensor tagRecv would leave the peer's frame stranded
      // on the wire and the FBS metadata would arrive with no payload to
      // splice into.
      if (tensor_buffer_size == 0) {
        throw std::runtime_error(
            "UcxxReceiverOp: frames of this message type arrive as a separate "
            "transfer, but no receive buffer space is reserved for them. "
            "Reserve space matching the published frame size "
            "(width * height * bytes-per-pixel).");
      }
    }
  }

  void setup(holoscan::OperatorSpec& spec) override {
    auto& output = spec.output(out_message, "out_message");
    if constexpr (Traits::kHasSeparateTensor) {
      output.produces_tensor(holoscan::TensorOutputSpec{
          .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice},
          .bounds = holoscan::TensorBounds{.max_rank = 8U, .max_byte_span = tensor_buffer_size_},
          .storage = holoscan::TensorOutputStorage::kRuntimePool});
    }
    spec.notification_source(recv_notification_, "ucxx_recv")
        .capacity(2U)  // one for metadata, one for tensor
        .sender_reference_capacity(2U);
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kArm, &UcxxReceiverOp::on_arm)
        .stage(holoscan::LifecycleStage::kStart, &UcxxReceiverOp::on_start)
        .stage(holoscan::LifecycleStage::kStop, &UcxxReceiverOp::on_stop)
        .stage(holoscan::LifecycleStage::kRelease, &UcxxReceiverOp::on_release);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnNotified{.event = recv_notification_});
    return contract;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto ep = endpoint_.endpoint();
    if (!ep) {
      // The initial connection callback posted receives with a live endpoint,
      // so a null return here means the peer dropped (on_endpoint_closed cleared state) and
      // there is no reconnect path. Surface as fatal so the session
      // terminates non-zero instead of running forever producing nothing.
      HOLOSCAN_LOG_ERROR("UCX receiver: peer disconnected; terminating.");
      return holoscan::make_unexpected(holoscan::Error{
          .code = holoscan::ErrorCode::kFailure, .message = "UCX receiver: peer disconnected."});
    }

    const bool message_ready = message_request_ && message_request_->isCompleted();
    const bool tensor_ready = !tensor_request_ || tensor_request_->isCompleted();
    if (!message_ready || !tensor_ready) {
      return {};
    }

    const bool message_ok = message_request_ && message_request_->getStatus() == UCS_OK;
    const bool tensor_ok = !tensor_request_ || tensor_request_->getStatus() == UCS_OK;
    if (!message_ok || !tensor_ok) {
      HOLOSCAN_LOG_ERROR("UCX receive failed; dropping message.");
      return reprime(ep);
    }

    // Validate the FBS structure before walking it: UCX itself is byte-safe
    // (UCS_OK above), but the payload may be truncated, corrupted, or hostile.
    // GetRoot alone only fetches a pointer inside our own buffer; UnPackTo
    // would then chase offsets that could point outside valid data. Verifier
    // walks the entire object graph within our buffer bound and refuses to
    // step past it. ucxx::Request does not expose the actual received size
    // for tagRecv, so use the full receive-buffer capacity as the upper
    // bound -- Verifier's built-in offset checks catch anything malformed.
    flatbuffers::Verifier verifier(flatbuffer_buffer_.data(), flatbuffer_buffer_.size());
    if (!verifier.VerifyBuffer<RootType>(nullptr)) {
      HOLOSCAN_LOG_ERROR("UCX receiver: FlatBuffer verification failed; dropping.");
      return reprime(ep);
    }

    NativeType message;
    const auto* root = flatbuffers::GetRoot<RootType>(flatbuffer_buffer_.data());
    message = Traits::decode(*root);

    auto options = emit_options(context, *root);
    if (!options) return holoscan::make_unexpected(options.error());
    holoscan::expected<void, holoscan::Error> emitted;
    if constexpr (Traits::kHasSeparateTensor) {
      if (!valid_tensor_descriptor(message)) {
        HOLOSCAN_LOG_ERROR("UCX receiver: invalid CUDA image descriptor; dropping frame.");
        return reprime(ep);
      }
      // UCX completion makes the receive buffer ready. Copy into SDK-owned
      // storage so downstream CUDA consumers get readiness and lifetime tracking.
      const auto shape = message.data->shape();
      auto loan = out_message.allocate_tensor(
          holoscan::TensorLoanRequest{.shape = shape, .dtype = message.data->dtype()});
      if (!loan) {
        HOLOSCAN_LOG_ERROR("UCX image allocation failed: {}.", loan.error().message);
        return holoscan::make_unexpected(std::move(loan).error());
      }
      {
        auto writer = loan->write(context.cuda_stream());
        if (!writer) {
          HOLOSCAN_LOG_ERROR("UCX image write access failed: {}.", writer.error().message);
          return holoscan::make_unexpected(std::move(writer).error());
        }
        if (cudaMemcpyAsync(writer->data(), tensor_buffer_.get(), message.data->nbytes(),
                            cudaMemcpyDeviceToDevice, context.cuda_stream().get()) != cudaSuccess ||
            cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
          HOLOSCAN_LOG_ERROR("UCX image copy into SDK tensor storage failed.");
          return holoscan::make_unexpected(holoscan::Error{
              holoscan::ErrorCode::kFailure, "UCX image copy into SDK tensor storage failed."});
        }
        auto committed = std::move(*writer).commit();
        if (!committed) return holoscan::make_unexpected(std::move(committed).error());
      }
      // Finish the copy before reusing the receive buffer for another UCX frame.
      message.data.reset();
      emitted = out_message.emit_tensor(std::move(*loan), message, *options);
    } else {
      emitted = out_message.emit(std::move(message), *options);
    }
    if (!emitted) {
      HOLOSCAN_LOG_ERROR(
          "Unable to publish a received UCX message on tag {:#x} from {}:{}: {} (code {}). {}",
          tag_, endpoint_.hostname(), endpoint_.port(),
          holoscan::error_code_name(emitted.error().code), static_cast<int>(emitted.error().code),
          emitted.error().message);
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    return reprime(ep);
  }

  holoscan::Output<NativeType> out_message;

 private:
  // Graph-clock receivers also accept messages without a wire timestamp.
  [[nodiscard]] static std::optional<std::uint64_t> wire_frame_id(const RootType& root) {
    try {
      const std::int64_t wire_time = Traits::timestamp(root);
      if (wire_time <= 0) return std::nullopt;
      return static_cast<std::uint64_t>(wire_time);
    } catch (const std::runtime_error&) {
      return std::nullopt;
    }
  }

  [[nodiscard]] holoscan::expected<holoscan::EmitOptions, holoscan::Error> emit_options(
      holoscan::ExecutionContext& context, const RootType& root) const {
    auto timestamp = capture_time(context, root);
    if (!timestamp) return holoscan::make_unexpected(timestamp.error());
    holoscan::EmitOptions options{.capture_time = *timestamp};
    if (timestamp_source_ == TimestampSource::kGraphClock) options.frame_id = wire_frame_id(root);
    return options;
  }

  [[nodiscard]] holoscan::expected<holoscan::TimePoint, holoscan::Error> capture_time(
      holoscan::ExecutionContext& context, const RootType& root) const {
    if (timestamp_source_ == TimestampSource::kMessage) {
      return context.clock().at_ns(Traits::timestamp(root));
    }
    return context.clock().now();
  }

  [[nodiscard]] bool valid_tensor_descriptor(const NativeType& message) const
    requires(Traits::kHasSeparateTensor)
  {
    return message.data && message.data->nbytes() > 0 &&
           static_cast<std::size_t>(message.data->nbytes()) <= tensor_buffer_size_ &&
           message.data->device().device_type == kDLCUDA && message.data->byte_offset() == 0 &&
           message.data->is_contiguous();
  }

  holoscan::LifecycleStatus on_arm(holoscan::LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(recv_notification_);
    if (!sender) {
      HOLOSCAN_LOG_ERROR("UCX receiver: failed to acquire notification sender.");
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    recv_sender_ = std::move(*sender);
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus on_start(holoscan::LifecycleContext&) noexcept {
    // contract() only wakes compute() on recv_notification_, and a receive
    // notification only fires once a tagRecv has been posted. Deferred
    // startup now supplies that initial post through the connection callback;
    // it does not introduce a reconnect path. An already-connected endpoint
    // invokes the callback immediately, preserving initial-post error handling.
    try {
      connect_callback_ = std::make_shared<UcxxEndpoint::ConnectCallback>(
          [this](const std::shared_ptr<ucxx::Endpoint>& ep) {
            try {
              if (!message_request_) {
                auto posted = post_next_receive(ep);
                if (!posted) {
                  throw std::runtime_error(holoscan::runtime_error_message(posted.error()));
                }
              }
            } catch (const std::exception& error) {
              HOLOSCAN_LOG_ERROR("Unable to initialize UCX receiver for tag {:#x} at {}:{}: {}",
                                 tag_, endpoint_.hostname(), endpoint_.port(), error.what());
              throw;
            }
          });
      endpoint_.notify_when_connected(connect_callback_);
    } catch (const std::exception& error) {
      HOLOSCAN_LOG_ERROR("Unable to start the UCX receiver: {}", error.what());
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus on_stop(holoscan::LifecycleContext&) noexcept {
    connect_callback_.reset();
    // cancel() is asynchronous under UCP -- it marks the request for
    // cancellation but the UCX progress thread may still be writing into
    // the destination buffer until the request actually completes with
    // UCS_ERR_CANCELED. Spin here until both requests report isCompleted()
    // so on_release can safely drop the CUDA buffer without racing an
    // in-flight write. Cancelled requests complete promptly (they are
    // already marked for teardown), so this loop is bounded and only ever
    // runs during graph shutdown.
    auto wait_for_completion = [](std::shared_ptr<ucxx::Request>& req) {
      if (!req) return;
      req->cancel();
      while (!req->isCompleted()) {
        std::this_thread::yield();
      }
    };
    wait_for_completion(message_request_);
    wait_for_completion(tensor_request_);
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus on_release(holoscan::LifecycleContext&) noexcept {
    tensor_buffer_.reset();
    tensor_request_.reset();
    message_request_.reset();
    return holoscan::LifecycleStatus::kOk;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> reprime(
      const std::shared_ptr<ucxx::Endpoint>& ep) {
    // Keep the staging allocation for the next frame.
    tensor_request_.reset();
    message_request_.reset();
    return post_next_receive(ep);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> post_next_receive(
      const std::shared_ptr<ucxx::Endpoint>& ep) {
    message_request_ = ep->tagRecv(
        flatbuffer_buffer_.data(), flatbuffer_buffer_.size(), ucxx::Tag{tag_}, ucxx::TagMaskFull,
        /*enablePythonFuture=*/false,
        [sender = recv_sender_](ucs_status_t status, std::shared_ptr<void>) {
          if (status != UCS_OK && status != UCS_ERR_CANCELED) {
            HOLOSCAN_LOG_ERROR("UCX metadata tagRecv failed: {}", ucs_status_string(status));
          }
          (void)sender.post();
        });

    if constexpr (Traits::kHasSeparateTensor) {
      // Reuse this staging buffer after its synchronous copy into SDK storage.
      if (!tensor_buffer_) {
        void* dev_ptr = nullptr;
        cudaError_t err = cudaMalloc(&dev_ptr, tensor_buffer_size_);
        if (err != cudaSuccess) {
          return holoscan::make_unexpected(
              holoscan::Error{.code = holoscan::ErrorCode::kResourceExhausted,
                              .message = "cudaMalloc failed for UCX tensor buffer"});
        }
        tensor_buffer_ = std::shared_ptr<std::byte>(static_cast<std::byte*>(dev_ptr),
                                                    [](std::byte* p) { cudaFree(p); });
      }
      tensor_request_ = ep->tagRecv(
          tensor_buffer_.get(), tensor_buffer_size_, ucxx::Tag{tag_}, ucxx::TagMaskFull,
          /*enablePythonFuture=*/false,
          [sender = recv_sender_](ucs_status_t status, std::shared_ptr<void>) {
            if (status != UCS_OK && status != UCS_ERR_CANCELED) {
              HOLOSCAN_LOG_ERROR("UCX tensor tagRecv failed: {}", ucs_status_string(status));
            }
            (void)sender.post();
          });
    }
    return {};
  }

  UcxxEndpoint endpoint_;
  std::shared_ptr<UcxxEndpoint::ConnectCallback> connect_callback_;
  std::uint64_t tag_{};

  std::vector<std::uint8_t> flatbuffer_buffer_;
  std::size_t tensor_buffer_size_{};
  TimestampSource timestamp_source_{TimestampSource::kMessage};
  std::shared_ptr<std::byte> tensor_buffer_;

  std::shared_ptr<ucxx::Request> message_request_;
  std::shared_ptr<ucxx::Request> tensor_request_;

  holoscan::NotificationSource recv_notification_;
  holoscan::NotificationSender recv_sender_;
};

}  // namespace isaac::os::ops

#endif  // ISAAC_OS_OPERATORS_UCXX_OPS_UCXX_RECEIVER_OP_H
