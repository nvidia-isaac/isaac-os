/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_APPLICATIONS_GR00T_N1_7_WBC_G1_GREEN_CONTEXT_H
#define ISAAC_OS_APPLICATIONS_GR00T_N1_7_WBC_G1_GREEN_CONTEXT_H

#include <cstdint>
#include <optional>
#include <string_view>

namespace holoscan {
struct CompileOptions;
class ExecutionPlan;
namespace config {
class Document;
}
}  // namespace holoscan

namespace isaac::os {

// CLI overrides YAML; an absent count selects the device quantum.
[[nodiscard]] std::optional<std::uint32_t> load_wbc_green_context_sms(
    const holoscan::config::Document& config,
    std::optional<std::uint32_t> cli_override = std::nullopt);

// Returns the resolved count. Zero disables partitioning; other CUDA work shares the residual SMs.
[[nodiscard]] std::uint32_t configure_wbc_green_context(holoscan::CompileOptions& options,
                                                        std::optional<std::uint32_t> requested_sms,
                                                        std::string_view partition_name = "wbc");
void validate_wbc_green_context(const holoscan::ExecutionPlan& plan, std::uint32_t sm_count,
                                std::string_view partition_name = "wbc");

}  // namespace isaac::os

#endif  // ISAAC_OS_APPLICATIONS_GR00T_N1_7_WBC_G1_GREEN_CONTEXT_H
