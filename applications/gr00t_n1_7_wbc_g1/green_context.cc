/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "applications/gr00t_n1_7_wbc_g1/green_context.h"

#include <algorithm>
#include <format>
#include <stdexcept>
#include <string>

#include <holoscan/config/codec.hpp>
#include <holoscan/config/document.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>

namespace isaac::os {

std::optional<std::uint32_t> load_wbc_green_context_sms(const holoscan::config::Document& config,
                                                        std::optional<std::uint32_t> cli_override) {
  if (cli_override.has_value()) {
    return cli_override;
  }
  const auto green_context = config.root().find_member("green_context");
  if (!green_context) {
    return std::nullopt;
  }
  const auto value = green_context->find_member("wbc_sms");
  if (!value) {
    return std::nullopt;
  }
  return holoscan::config::decode_config<std::uint32_t>(*value);
}

std::uint32_t configure_wbc_green_context(holoscan::CompileOptions& options,
                                          std::optional<std::uint32_t> requested_sms,
                                          std::string_view partition_name) {
  if (requested_sms == 0U) {
    return 0U;
  }
  const auto capability = holoscan::query_cuda_green_context_capability(holoscan::DeviceId{0});
  if (!capability) {
    throw holoscan::RuntimeError(capability.error().code,
                                 "CUDA Green Context is unavailable on this device. "
                                 "Set the WBC SM count to zero to disable partitioning.");
  }
  const std::uint32_t quantum = capability->structured_sm_quantum;
  const std::uint32_t sm_count = requested_sms.value_or(quantum);
  const std::uint64_t required_sms = static_cast<std::uint64_t>(sm_count) + quantum;
  const std::uint64_t available_sms =
      static_cast<std::uint64_t>(capability->max_structured_context_count) * quantum;
  if (quantum == 0U || sm_count % quantum != 0U || required_sms > available_sms) {
    throw std::invalid_argument(std::format(
        "The requested WBC allocation of {} SMs is unsupported. This device allocates groups of "
        "{} SMs with {} available; at least one group must remain for other GPU work. "
        "Set the WBC SM count to zero to disable partitioning.",
        sm_count, quantum, available_sms));
  }
  options.deployment.bind_cuda_green_context(
      partition_name, {.device = holoscan::DeviceId{0}, .sm_count = sm_count});
  return sm_count;
}

void validate_wbc_green_context(const holoscan::ExecutionPlan& plan, std::uint32_t sm_count,
                                std::string_view partition_name) {
  if (sm_count == 0U) {
    return;
  }
  const auto request = std::ranges::find_if(plan.cuda_green_context_requests(),
                                            [partition_name](const auto& candidate) {
                                              return candidate.authored_name == partition_name;
                                            });
  if (request == plan.cuda_green_context_requests().end() || !request->accepted) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible,
                                 "The requested WBC CUDA Green Context was not accepted.");
  }
  if (request->accepted->sm_count != sm_count) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible,
                                 "The WBC CUDA Green Context was assigned " +
                                     std::to_string(request->accepted->sm_count) + " SMs, but " +
                                     std::to_string(sm_count) + " SMs were requested.");
  }
  if (plan.cuda_green_context_assignments().empty()) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kNotFound,
                                 "The WBC CUDA Green Context has no GPU assignments.");
  }
}

}  // namespace isaac::os
