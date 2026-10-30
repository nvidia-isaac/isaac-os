/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/teleop/teleop_params_config_codec.h"

#include <libevdev/libevdev.h>
#include <linux/input-event-codes.h>

#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace holoscan::config {
namespace {

ConfigResult<std::int64_t> decode_event_code(ObjectDecoder& object, ConfigView view,
                                             std::string_view key, unsigned int type,
                                             DecodeContext& context) {
  const auto field = view.find_member(key);
  if (!field || field->kind() != ConfigValueKind::kString) {
    auto code = object.required<std::int64_t>(key);
    if (!code) {
      return code;
    }
    if (*code < 0 || *code > libevdev_event_type_get_max(type) ||
        libevdev_event_code_get_name(type, static_cast<unsigned int>(*code)) == nullptr) {
      return tl::make_unexpected(ConfigError::make(
          ConfigErrorCode::kCodecFailure, std::format("Unknown gamepad input code {}.", *code),
          std::string{context.path()} + "/" + std::string{key}));
    }
    return code;
  }
  auto name = object.required<std::string>(key);
  if (!name) {
    return tl::make_unexpected(name.error());
  }
  const int code = libevdev_event_code_from_name(type, name->c_str());
  if (code < 0) {
    return tl::make_unexpected(ConfigError::make(
        ConfigErrorCode::kCodecFailure, "Unknown gamepad input code '" + *name + "'.",
        std::string{context.path()} + "/" + std::string{key}));
  }
  return code;
}

}  // namespace

ConfigResult<isaac::os::ops::TeleopAxis> ConfigCodec<isaac::os::ops::TeleopAxis>::decode(
    ConfigView view, const DecodeOptions& options, DecodeContext& context) {
  ObjectDecoder object{view, options, context};
  auto code = decode_event_code(object, view, "code", EV_ABS, context);
  if (!code) {
    return tl::make_unexpected(code.error());
  }
  auto invert = object.optional<bool>("invert");
  if (!invert) {
    return tl::make_unexpected(invert.error());
  }
  auto finished = object.finish();
  if (!finished) {
    return tl::make_unexpected(finished.error());
  }
  return isaac::os::ops::TeleopAxis{.code = *code, .invert = invert->value_or(true)};
}

ConfigResult<isaac::os::ops::TeleopDeadman> ConfigCodec<isaac::os::ops::TeleopDeadman>::decode(
    ConfigView view, const DecodeOptions& options, DecodeContext& context) {
  ObjectDecoder object{view, options, context};
  auto axis = decode_event_code(object, view, "axis", EV_ABS, context);
  if (!axis) {
    return tl::make_unexpected(axis.error());
  }
  auto threshold = object.optional<std::int64_t>("threshold");
  if (!threshold) {
    return tl::make_unexpected(threshold.error());
  }
  auto finished = object.finish();
  if (!finished) {
    return tl::make_unexpected(finished.error());
  }
  return isaac::os::ops::TeleopDeadman{.axis = *axis, .threshold = threshold->value_or(200)};
}

ConfigResult<isaac::os::ops::TeleopGamepadMapping>
ConfigCodec<isaac::os::ops::TeleopGamepadMapping>::decode(ConfigView view,
                                                          const DecodeOptions& options,
                                                          DecodeContext& context) {
  ObjectDecoder object{view, options, context};
  auto axes = object.required<std::vector<isaac::os::ops::TeleopAxis>>("axes");
  if (!axes) {
    return tl::make_unexpected(axes.error());
  }
  auto deadman = object.required<isaac::os::ops::TeleopDeadman>("deadman");
  if (!deadman) {
    return tl::make_unexpected(deadman.error());
  }
  auto lower = decode_event_code(object, view, "lower_body_enable_button", EV_KEY, context);
  if (!lower) {
    return tl::make_unexpected(lower.error());
  }
  std::int64_t upper = 0;
  if (view.find_member("upper_body_enable_button")) {
    auto code = decode_event_code(object, view, "upper_body_enable_button", EV_KEY, context);
    if (!code) {
      return tl::make_unexpected(code.error());
    }
    upper = *code;
  }
  auto finished = object.finish();
  if (!finished) {
    return tl::make_unexpected(finished.error());
  }
  return isaac::os::ops::TeleopGamepadMapping{
      .axes = std::move(*axes),
      .deadman = *deadman,
      .lower_body_enable_button = *lower,
      .upper_body_enable_button = upper,
  };
}

}  // namespace holoscan::config
