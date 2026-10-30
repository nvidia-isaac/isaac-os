/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/utils/regex.h"

#include <unordered_map>

namespace isaac::utils {

std::vector<int> get_matching_ids(const std::string& pattern,
                                  const std::vector<std::string>& names) {
  std::vector<int> matching_ids;
  std::regex regex_pattern(pattern);

  if (names.empty()) {
    throw std::runtime_error("Names vector is empty");
  }

  for (size_t i = 0; i < names.size(); ++i) {
    if (std::regex_match(names[i], regex_pattern)) {
      matching_ids.push_back(static_cast<int>(i));
    }
  }
  return matching_ids;
}

std::vector<int> get_matching_ids(const std::vector<std::string>& patterns,
                                  const std::vector<std::string>& names) {
  std::vector<int> matching_ids;
  std::string combined_pattern;
  for (size_t i = 0; i < patterns.size(); ++i) {
    if (i > 0) {
      combined_pattern += "|";
    }
    combined_pattern += "(" + patterns[i] + ")";
  }

  if (!combined_pattern.empty()) {
    matching_ids = get_matching_ids(combined_pattern, names);
  }
  return matching_ids;
}

std::vector<int> get_reorder_indices(const std::vector<std::string>& from_names,
                                     const std::vector<std::string>& to_names) {
  std::unordered_map<std::string, int> from_name_to_index;
  for (size_t i = 0; i < from_names.size(); ++i) {
    from_name_to_index[from_names[i]] = static_cast<int>(i);
  }

  std::vector<int> reorder_indices;
  reorder_indices.reserve(to_names.size());

  for (const auto& to_name : to_names) {
    auto it = from_name_to_index.find(to_name);
    if (it == from_name_to_index.end()) {
      throw std::runtime_error("Cannot reorder joint names because '" + to_name +
                               "' is missing from the source joint list.");
    }
    reorder_indices.push_back(it->second);
  }

  return reorder_indices;
}

}  // namespace isaac::utils
