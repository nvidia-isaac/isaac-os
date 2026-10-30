/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace isaac::utils {

/// Returns the indices of elements in @p names that fully match @p pattern (std::regex_match).
/// Throws std::runtime_error if @p names is empty.
std::vector<int> get_matching_ids(const std::string& pattern,
                                  const std::vector<std::string>& names);

/// Returns the indices of elements in @p names that match any pattern in @p patterns.
/// Patterns are combined with OR before matching. Returns an empty vector if @p patterns is empty.
std::vector<int> get_matching_ids(const std::vector<std::string>& patterns,
                                  const std::vector<std::string>& names);

/// Returns indices into @p from_names that reorder it to match @p to_names.
/// Each element of @p to_names must appear in @p from_names; throws std::runtime_error otherwise.
std::vector<int> get_reorder_indices(const std::vector<std::string>& from_names,
                                     const std::vector<std::string>& to_names);

}  // namespace isaac::utils
