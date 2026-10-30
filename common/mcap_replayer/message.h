/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"

namespace isaac::os::mcap_replayer {

// One decoded message waiting its turn, type-erased so a single buffer can hold every topic's.
// It carries only data: the topic's port, which made it and knows which derived type it is, is the
// one that publishes it.
struct Message {
  virtual ~Message() = default;

  std::uint64_t log_time_ns = 0;
  // The instant this message fell due, stamped by ReplayBuffer::release_if_due(). Per message so
  // lag is measured against each one's own due time when several are released back to back.
  std::chrono::steady_clock::time_point target;
};

// A topic published as the value of its type: the recorded flatbuffer, materialized.
template <typename T>
struct FlatbufferMessage final : Message {
  T value;
};

// A tensor-schema topic: the recorded metadata bytes and the rebuilt tensors, held as they are to
// be published, never materialized as the schema's type. Nothing here depends on that type.
struct TensorMessage final : Message {
  std::vector<std::byte> metadata;
  std::vector<holoscan::Tensor> tensors;
};

}  // namespace isaac::os::mcap_replayer
