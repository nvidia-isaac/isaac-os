/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "operators/agile_wbc/wbc_params.h"

namespace isaac::os::ops {
namespace {

void validate_joint_order(const std::vector<std::string>& order, std::string_view field,
                          const std::unordered_set<std::string>* sdk_joints = nullptr) {
  if (order.empty()) {
    throw std::invalid_argument("'" + std::string{field} + "' must not be empty");
  }

  std::unordered_set<std::string> unique;
  for (const std::string& joint : order) {
    if (!unique.insert(joint).second) {
      throw std::invalid_argument("'" + std::string{field} + "' contains duplicate joint '" +
                                  joint + "'");
    }
    if (sdk_joints != nullptr && !sdk_joints->contains(joint)) {
      throw std::invalid_argument("'" + std::string{field} + "' contains joint '" + joint +
                                  "' that is not in 'sdk_joint_order'");
    }
  }
}

}  // namespace

void validate_wbc_params(const WbcParams& params) {
  if (params.onnx_path.empty()) {
    throw std::invalid_argument("'onnx_path' must not be empty");
  }

  validate_joint_order(params.sdk_joint_order, "sdk_joint_order");
  const std::unordered_set<std::string> sdk_joints(params.sdk_joint_order.begin(),
                                                   params.sdk_joint_order.end());
  validate_joint_order(params.policy_joint_order, "policy_joint_order", &sdk_joints);
  validate_joint_order(params.action_joint_order, "action_joint_order", &sdk_joints);

  if (params.history_terms.empty()) {
    throw std::invalid_argument("'history_terms' must not be empty");
  }

  std::unordered_set<std::string> input_names;
  std::unordered_set<std::string> output_names;
  for (const HistoryTerm& term : params.history_terms) {
    if (term.input_name.empty()) {
      throw std::invalid_argument("history term has an empty 'input_name'");
    }
    if (term.output_name.empty()) {
      throw std::invalid_argument("history term '" + term.input_name +
                                  "' has an empty 'output_name'");
    }
    if (term.dim <= 0) {
      throw std::invalid_argument("history term '" + term.input_name + "' has a nonpositive 'dim'");
    }
    if (term.length < 0) {
      throw std::invalid_argument("history term '" + term.input_name + "' has a negative 'length'");
    }
    if (!input_names.insert(term.input_name).second) {
      throw std::invalid_argument("duplicate history 'input_name' '" + term.input_name + "'");
    }
    if (!output_names.insert(term.output_name).second) {
      throw std::invalid_argument("duplicate history 'output_name' '" + term.output_name + "'");
    }
  }

  for (const std::string& output_name : output_names) {
    if (input_names.contains(output_name)) {
      throw std::invalid_argument("history 'output_name' '" + output_name +
                                  "' collides with an input name");
    }
  }
  if (!input_names.contains(params.last_actions_input_name)) {
    throw std::invalid_argument("'last_actions_input_name' value '" +
                                params.last_actions_input_name +
                                "' does not identify a history input");
  }
}

}  // namespace isaac::os::ops
