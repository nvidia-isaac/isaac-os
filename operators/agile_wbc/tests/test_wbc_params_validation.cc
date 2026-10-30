/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "operators/agile_wbc/wbc_params.h"

namespace isaac::os::ops {
namespace {

WbcParams valid_params() {
  return WbcParams{
      .onnx_path = "policy.onnx",
      .sdk_joint_order = {"hip", "knee", "ankle"},
      .policy_joint_order = {"hip", "knee"},
      .action_joint_order = {"hip", "ankle"},
      .last_actions_input_name = "last_actions",
      .history_terms =
          {
              {.input_name = "joint_pos", .output_name = "joint_pos_out", .dim = 2, .length = 5},
              {.input_name = "last_actions", .output_name = "last_actions_out", .dim = 2},
          },
  };
}

struct InvalidCase {
  const char* name;
  void (*mutate)(WbcParams&);
  const char* expected_field;
};

TEST(WbcParamsValidation, AcceptsValidParams) {
  EXPECT_NO_THROW(validate_wbc_params(valid_params()));
}

TEST(WbcParamsValidation, RejectsInvalidRelationships) {
  const std::array cases{
      InvalidCase{"empty onnx path", [](WbcParams& params) { params.onnx_path.clear(); },
                  "onnx_path"},
      InvalidCase{"empty sdk order", [](WbcParams& params) { params.sdk_joint_order.clear(); },
                  "sdk_joint_order"},
      InvalidCase{"empty policy order",
                  [](WbcParams& params) { params.policy_joint_order.clear(); },
                  "policy_joint_order"},
      InvalidCase{"empty action order",
                  [](WbcParams& params) { params.action_joint_order.clear(); },
                  "action_joint_order"},
      InvalidCase{"duplicate sdk joint",
                  [](WbcParams& params) { params.sdk_joint_order.push_back("hip"); },
                  "sdk_joint_order"},
      InvalidCase{"duplicate policy joint",
                  [](WbcParams& params) { params.policy_joint_order.push_back("hip"); },
                  "policy_joint_order"},
      InvalidCase{"duplicate action joint",
                  [](WbcParams& params) { params.action_joint_order.push_back("hip"); },
                  "action_joint_order"},
      InvalidCase{"unknown policy joint",
                  [](WbcParams& params) { params.policy_joint_order.push_back("wrist"); },
                  "policy_joint_order"},
      InvalidCase{"unknown action joint",
                  [](WbcParams& params) { params.action_joint_order.push_back("wrist"); },
                  "action_joint_order"},
      InvalidCase{"empty history", [](WbcParams& params) { params.history_terms.clear(); },
                  "history_terms"},
      InvalidCase{"empty history input",
                  [](WbcParams& params) { params.history_terms.front().input_name.clear(); },
                  "input_name"},
      InvalidCase{"empty history output",
                  [](WbcParams& params) { params.history_terms.front().output_name.clear(); },
                  "output_name"},
      InvalidCase{"nonpositive history dimension",
                  [](WbcParams& params) { params.history_terms.front().dim = 0; }, "dim"},
      InvalidCase{"negative history length",
                  [](WbcParams& params) { params.history_terms.front().length = -1; }, "length"},
      InvalidCase{"duplicate history input",
                  [](WbcParams& params) {
                    params.history_terms.back().input_name =
                        params.history_terms.front().input_name;
                  },
                  "input_name"},
      InvalidCase{"duplicate history output",
                  [](WbcParams& params) {
                    params.history_terms.back().output_name =
                        params.history_terms.front().output_name;
                  },
                  "output_name"},
      InvalidCase{"history output and input collision",
                  [](WbcParams& params) {
                    params.history_terms.front().output_name =
                        params.history_terms.back().input_name;
                  },
                  "output_name"},
      InvalidCase{"unknown last actions input",
                  [](WbcParams& params) { params.last_actions_input_name = "missing"; },
                  "last_actions_input_name"},
  };

  for (const InvalidCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    WbcParams params = valid_params();
    test_case.mutate(params);
    try {
      validate_wbc_params(params);
      FAIL() << "Expected invalid parameters";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string{error.what()}.find(test_case.expected_field), std::string::npos)
          << error.what();
    }
  }
}

}  // namespace
}  // namespace isaac::os::ops
