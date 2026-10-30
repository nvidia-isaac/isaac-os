/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <functional>
#include <string_view>

namespace isaac::os::mcap_logger {

// Lets a string-keyed unordered container be searched by std::string_view, or a string literal,
// without first copying the key into a temporary std::string.
//
// A lookup that has to build the string pays for it every time: a topic name past the small-string
// buffer (16 characters here) is a heap allocation and a copy just to ask whether a channel
// exists. C++20 lets the container skip that, but only when both its hasher and its equality
// predicate are transparent. std::equal_to<> already is; the standard library provides no
// matching hasher, so this is it. Use the two together:
//
//   std::unordered_map<std::string, V, TransparentStringHash, std::equal_to<>>
//
// Only lookup is heterogeneous. Insertion still needs a real std::string key, because inserting
// with another key type is C++26.
struct TransparentStringHash {
  using is_transparent = void;

  [[nodiscard]] std::size_t operator()(std::string_view key) const noexcept {
    return std::hash<std::string_view>{}(key);
  }
};

}  // namespace isaac::os::mcap_logger
