/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "operators/start_up/start_up_op.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/logger/logger.hpp>

#include "common/utils/regex.h"

namespace isaac::os::ops {
namespace {

constexpr float kNanosecondsToSeconds = 1.0e-9F;
constexpr auto kHandoverStateMaxAge = std::chrono::milliseconds{40};

std::vector<int> matching_joints(const std::string& pattern,
                                 const std::vector<std::string>& names) {
  auto indices = isaac::utils::get_matching_ids(pattern, names);
  if (indices.empty()) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "The startup joint pattern '" + pattern + "' matches no joint.");
  }
  return indices;
}

// Later patterns in sorted order override earlier matches.
std::vector<float> expand_joint_patterns(const std::map<std::string, double, std::less<>>& patterns,
                                         const std::vector<std::string>& names) {
  std::vector<float> values(names.size(), 0.0F);
  for (const auto& [pattern, value] : patterns) {
    for (const int index : matching_joints(pattern, names)) {
      values[index] = static_cast<float>(value);
    }
  }
  return values;
}

int duration_to_steps(double seconds, double control_period) {
  const double steps = seconds / control_period;
  if (!std::isfinite(steps) || seconds < 0.0 || steps > std::numeric_limits<int>::max()) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kInvalidArgument,
        "Startup durations must be nonnegative and fit in a control-step count.");
  }
  return std::max(1, static_cast<int>(steps));
}

bool take_enable_assertion(holoscan::Input<bool>& input, std::int64_t& last_signal_ns) {
  if (input.is_missing()) {
    return false;
  }
  const auto received = input.receive();
  if (!received) {
    return false;
  }
  const std::int64_t signal_ns = received->metadata.capture_time().timestamp_ns;
  if (signal_ns == last_signal_ns) {
    return false;
  }
  last_signal_ns = signal_ns;
  return received->data;
}

}  // namespace

void StartUpOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(joint_command, "in_joint_command").queue_depth(1U);
  spec.input(joint_state, "in_joint_state").queue_depth(1U);
  if (enable_signal_connected_) {
    spec.input(enable_lower, "in_enable_lower").queue_depth(1U);
    spec.input(enable_upper, "in_enable_upper").queue_depth(1U);
  }
  spec.output(command, "out_joint_command");
}

holoscan::Contract StartUpOp::contract() const {
  holoscan::Contract contract;
  contract.trigger(holoscan::OnEach{joint_command});
  contract.input(joint_state, holoscan::LatestWithin{.max_age = std::chrono::nanoseconds::max()});
  contract.on_missing(joint_state, holoscan::SafeDefault{});
  if (enable_signal_connected_) {
    contract.input(enable_lower,
                   holoscan::LatestWithin{.max_age = std::chrono::nanoseconds::max()});
    contract.on_missing(enable_lower, holoscan::SafeDefault{});
    contract.input(enable_upper,
                   holoscan::LatestWithin{.max_age = std::chrono::nanoseconds::max()});
    contract.on_missing(enable_upper, holoscan::SafeDefault{});
  }
  return contract;
}

void StartUpOp::start() {
  const std::size_t joints = joint_count();
  if (joints == 0U) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kInvalidArgument,
        "The startup routine needs the order the publisher expects joints in.");
  }
  const auto& config = params();
  initialize_upper_body();
  if (config.handover_only) {
    if (!config.waypoints.empty() || !upper_body_.has_any) {
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kInvalidArgument,
          "A handover-only startup needs upper-body joints and no waypoints.");
    }
    joint_state_seen_ = false;
    measured_time_ns_ = -1;
    state_ = State::kStartingUp;
    enable_latched_ = false;
    last_enable_lower_ns_ = -1;
    last_enable_upper_ns_ = -1;
    warned_command_size_ = false;
    if (enable_signal_connected_) {
      HOLOSCAN_LOG_INFO("The startup handover is waiting for lower- and upper-body enable.");
    }
    return;
  }
  if (!config.kp || !config.kd) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "The startup routine must define both 'kp' and 'kd' gain maps.");
  }
  if (!std::isfinite(config.control_dt) || config.control_dt <= 0.0) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kInvalidArgument,
        "The startup control period must be finite and greater than zero.");
  }
  kp_ = expand_joint_patterns(*config.kp, config.sdk_joint_order);
  kd_ = expand_joint_patterns(*config.kd, config.sdk_joint_order);
  hold_kp_ = config.hold_kp ? expand_joint_patterns(*config.hold_kp, config.sdk_joint_order) : kp_;
  hold_kd_ = config.hold_kd ? expand_joint_patterns(*config.hold_kd, config.sdk_joint_order) : kd_;
  if (std::ranges::all_of(kp_, [](float gain) { return gain == 0.0F; })) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kInvalidArgument,
        "The startup routine sets no position gain on any joint, which would leave the robot "
        "limp.");
  }
  if (!enable_signal_connected_ &&
      (!std::isfinite(config.auto_enable_delay) || config.auto_enable_delay < 0.0)) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "The automatic enable delay must be finite and nonnegative.");
  }
  const double auto_enable_delay = enable_signal_connected_ ? 0.0 : config.auto_enable_delay;
  if (!enable_signal_connected_ &&
      auto_enable_delay * 1.0e9 >= std::numeric_limits<std::int64_t>::max()) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "The automatic enable delay must fit in nanoseconds.");
  }
  auto_enable_delay_ns_ =
      enable_signal_connected_ ? -1 : static_cast<std::int64_t>(auto_enable_delay * 1.0e9);
  if (config.vel_settled_threshold < 0.0) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kInvalidArgument,
        "The velocity a joint counts as settled below cannot be negative.");
  }
  if (config.waypoints.empty()) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "The startup routine must contain at least one pose.");
  }

  const int dwell_steps = duration_to_steps(config.dwell_duration, config.control_dt);
  std::vector<WaypointSequencer::Waypoint> waypoints;
  waypoints.reserve(config.waypoints.size());
  for (const StartUpWaypoint& waypoint : config.waypoints) {
    waypoints.push_back(WaypointSequencer::Waypoint{
        .name = waypoint.name,
        .positions = expand_joint_patterns(waypoint.joint_positions, config.sdk_joint_order),
        .interp_steps = duration_to_steps(waypoint.duration, config.control_dt),
        .dwell_steps = dwell_steps,
    });
  }

  upper_body_.final_pose = waypoints.back().positions;
  if (upper_body_.final_pose.size() != joints) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "The final startup pose must give a position for every joint.");
  }
  upper_body_.held_pose = upper_body_.final_pose;
  sequencer_ = std::make_unique<WaypointSequencer>(std::move(waypoints),
                                                   config.vel_settled_threshold, joints);

  measured_position_.assign(joints, 0.0F);
  measured_velocity_.assign(joints, 0.0F);
  joint_state_seen_ = false;
  state_ = State::kStartingUp;
  enable_latched_ = false;
  final_reached_ns_ = -1;
  last_enable_lower_ns_ = 0;
  last_enable_upper_ns_ = 0;
  warned_command_size_ = false;

  if (upper_body_.has_any) {
    const auto held =
        std::ranges::count_if(upper_body_mask_, [](std::uint8_t value) { return value != 0U; });
    HOLOSCAN_LOG_INFO(
        "{} joints hold the startup pose until the upper-body enable is received, moving at up to "
        "{} rad/s.",
        held, UpperBodyHold::kMaxVelocity);
  }
}

void StartUpOp::initialize_upper_body() {
  const auto& config = params();
  if (config.handover_only &&
      (!std::isfinite(config.handover_duration) || config.handover_duration <= 0.0)) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kInvalidArgument,
                                 "The startup handover duration must be finite and positive.");
  }
  const auto joints = joint_count();
  upper_body_mask_.assign(joints, 0U);
  for (const auto& pattern : config.upper_body_joints) {
    for (const int index : matching_joints(pattern, config.sdk_joint_order)) {
      upper_body_mask_[index] = 1U;
    }
  }
  upper_body_ = {};
  upper_body_.has_any = std::ranges::any_of(upper_body_mask_, [](auto held) { return held != 0U; });
}

bool StartUpOp::prepare_handover(const isaac::JointCommandCpuT& value,
                                 const holoscan::SampleMetadata& metadata) {
  const auto time = metadata.capture_time();
  if (!metadata.valid() || !time.valid()) return false;
  for (std::size_t joint = 0; joint < joint_count(); ++joint) {
    if (upper_body_mask_[joint] == 0U && !std::isfinite(value.position[joint])) return false;
  }
  if (state_ != State::kStartingUp) return true;
  const bool fresh_state =
      joint_state_seen_ && measured_time_ns_ >= 0 && time.timestamp_ns >= measured_time_ns_ &&
      std::chrono::nanoseconds{time.timestamp_ns - measured_time_ns_} <= kHandoverStateMaxAge;
  if (upper_body_.held_pose.empty()) {
    if (!fresh_state) return false;
    upper_body_.held_pose = measured_position_;
  }
  // Hold only the upper body while waiting. Lower-body control must keep flowing so the
  // simulator can advance and deliver the observations needed by the learned policy.
  if (!fresh_state || metadata.degraded() ||
      !std::ranges::all_of(value.position,
                           [](float position) { return std::isfinite(position); }) ||
      (enable_signal_connected_ && (!enable_latched_ || !upper_body_.released)))
    return true;
  upper_body_.final_pose = measured_position_;
  upper_body_.held_pose = measured_position_;
  upper_body_.released = true;
  sequencer_ = std::make_unique<WaypointSequencer>(
      std::vector<WaypointSequencer::Waypoint>{{"handover", value.position, 1, 0}}, 0.0,
      joint_count());
  // Capture the measured initial pose before subsequent state samples arrive.
  sequencer_->step(&measured_position_, nullptr, WaypointSequencer::Handover{value.position, 0.0F});
  handover_start_ns_ = time.timestamp_ns;
  upper_body_.last_tick_ns = time.timestamp_ns;
  state_ = State::kHandingOver;
  HOLOSCAN_LOG_INFO("The startup handover is active from the measured pose.");
  return true;
}

bool StartUpOp::command_covers_every_joint(const isaac::JointCommandCpuT& value) const {
  const std::size_t joints = joint_count();
  return value.position.size() == joints && value.velocity.size() == joints &&
         value.effort.size() == joints && value.kp.size() == joints && value.kd.size() == joints;
}

void StartUpOp::cache_measured_state() {
  if (joint_state.is_missing()) {
    return;
  }
  auto received = joint_state.receive();
  if (!received) {
    return;
  }
  const isaac::JointStateCpuT& state = received->data;
  if (params().handover_only) {
    const auto time = received->metadata.capture_time();
    if (!received->metadata.valid() || !time.valid() ||
        !std::ranges::all_of(state.position,
                             [](float position) { return std::isfinite(position); }))
      return;
  }
  if (state.position.size() != joint_count() ||
      (!params().handover_only && state.velocity.size() != joint_count())) {
    return;
  }
  measured_time_ns_ = received->metadata.capture_time().timestamp_ns;
  measured_position_ = state.position;
  measured_velocity_ = state.velocity;
  joint_state_seen_ = true;
}

void StartUpOp::latch_enables() {
  if (!enable_signal_connected_) {
    return;
  }
  // Apply each held enable sample once, by capture time.
  const bool lower_asserted = take_enable_assertion(enable_lower, last_enable_lower_ns_);
  if (lower_asserted && !enable_latched_) {
    enable_latched_ = true;
    HOLOSCAN_LOG_INFO("The lower body is enabled.");
  }
  if (take_enable_assertion(enable_upper, last_enable_upper_ns_)) {
    upper_body_.released = !upper_body_.released;
    upper_body_.active = true;  // Slew toward the new target.
    // An interrupted initial blend joins the normal hold/release path from its last output.
    if (state_ == State::kHandingOver) state_ = State::kRunning;
    HOLOSCAN_LOG_INFO("The upper body is {}.",
                      upper_body_.released ? "enabled" : "holding its startup pose");
  }
}

void StartUpOp::advance_handover(const isaac::JointCommandCpuT& value, std::int64_t now_ns) {
  if (!upper_body_.active || now_ns <= upper_body_.last_tick_ns) return;
  upper_body_.last_tick_ns = now_ns;
  const double elapsed_s = static_cast<double>(now_ns - handover_start_ns_) * 1.0e-9;
  const auto fraction = static_cast<float>(elapsed_s / params().handover_duration);
  upper_body_.held_pose = sequencer_->step(&measured_position_, nullptr,
                                           WaypointSequencer::Handover{value.position, fraction});
  upper_body_.weight = 1.0F - smoothstep(fraction);
  if (sequencer_->at_final_waypoint()) {
    upper_body_.active = false;
    state_ = State::kRunning;
    HOLOSCAN_LOG_INFO("The startup handover is complete; controller commands now pass through.");
  }
}

void StartUpOp::advance_upper_hold(const isaac::JointCommandCpuT& value, std::int64_t now_ns) {
  if (!upper_body_.has_any) {
    return;
  }
  const float elapsed_s =
      upper_body_.last_tick_ns < 0
          ? 0.0F
          : static_cast<float>(now_ns - upper_body_.last_tick_ns) * kNanosecondsToSeconds;
  upper_body_.last_tick_ns = now_ns;

  const std::vector<std::uint8_t>& mask = upper_body_mask_;
  const std::size_t joints = joint_count();
  if (!upper_body_.active) {
    upper_body_.held_pose = value.position;
    return;
  }

  float span = 0.0F;
  float to_hold = 0.0F;
  float from_controller = 0.0F;
  for (std::size_t joint = 0; joint < joints; ++joint) {
    if (mask[joint] == 0U) {
      continue;
    }
    const float target =
        upper_body_.released ? value.position[joint] : upper_body_.final_pose[joint];
    span = std::max(span, std::abs(target - upper_body_.held_pose[joint]));
    to_hold =
        std::max(to_hold, std::abs(upper_body_.final_pose[joint] - upper_body_.held_pose[joint]));
    from_controller =
        std::max(from_controller, std::abs(upper_body_.final_pose[joint] - value.position[joint]));
  }
  // Fade velocity and effort with progress toward the hold pose.
  upper_body_.weight = from_controller > UpperBodyHold::kArrived
                           ? std::clamp(1.0F - to_hold / from_controller, 0.0F, 1.0F)
                           : 1.0F;

  if (span <= UpperBodyHold::kArrived) {
    for (std::size_t joint = 0; joint < joints; ++joint) {
      if (mask[joint] != 0U) {
        upper_body_.held_pose[joint] =
            upper_body_.released ? value.position[joint] : upper_body_.final_pose[joint];
      }
    }
    upper_body_.speed = 0.0F;
    upper_body_.active = !upper_body_.released;
    return;
  }

  const float ramp = UpperBodyHold::kMaxAcceleration * elapsed_s;
  const float approach = std::sqrt(2.0F * UpperBodyHold::kMaxAcceleration * span);
  upper_body_.speed = std::clamp(std::min(UpperBodyHold::kMaxVelocity, approach),
                                 upper_body_.speed - ramp, upper_body_.speed + ramp);
  const float progress = std::min(1.0F, upper_body_.speed * elapsed_s / span);
  for (std::size_t joint = 0; joint < joints; ++joint) {
    if (mask[joint] == 0U) {
      continue;
    }
    const float target =
        upper_body_.released ? value.position[joint] : upper_body_.final_pose[joint];
    upper_body_.held_pose[joint] += progress * (target - upper_body_.held_pose[joint]);
  }
}

void StartUpOp::hold_upper(isaac::JointCommandCpuT& value) const {
  const std::vector<std::uint8_t>& mask = upper_body_mask_;
  const float weight = upper_body_.weight;
  for (std::size_t joint = 0; joint < joint_count(); ++joint) {
    if (mask[joint] == 0U) {
      continue;
    }
    value.position[joint] = upper_body_.held_pose[joint];
    value.velocity[joint] -= weight * value.velocity[joint];
    value.effort[joint] -= weight * value.effort[joint];
  }
}

void StartUpOp::end_startup_if_enabled_or_delayed(std::int64_t now_ns) {
  if (final_reached_ns_ < 0) {
    final_reached_ns_ = now_ns;  // Start the hold timer.
  }
  const std::int64_t delay_ns = auto_enable_delay_ns_;
  const bool delayed = delay_ns >= 0 && now_ns - final_reached_ns_ >= delay_ns;
  if (!enable_latched_ && !delayed) {
    return;
  }
  if (delayed) {
    upper_body_.released = true;
    HOLOSCAN_LOG_INFO("Startup finished after holding the final pose; the robot is enabled.");
  } else {
    HOLOSCAN_LOG_INFO("Startup finished; the robot is enabled.");
  }
  state_ = State::kRunning;
}

bool StartUpOp::update_upper_body(isaac::JointCommandCpuT& value, std::int64_t capture_ns,
                                  std::int64_t now_ns) {
  if (params().handover_only && !std::ranges::all_of(value.position, [](float position) {
        return std::isfinite(position);
      })) {
    // Reject the whole upper-body target without interrupting valid lower-body control.
    for (std::size_t joint = 0; joint < joint_count(); ++joint) {
      if (upper_body_mask_[joint] == 0U) continue;
      value.position[joint] = upper_body_.held_pose[joint];
      value.velocity[joint] = 0.0F;
      value.effort[joint] = 0.0F;
    }
    return false;
  }
  if (state_ == State::kHandingOver) {
    advance_handover(value, capture_ns);
  } else if (params().handover_only && state_ == State::kRunning) {
    if (!upper_body_.active) {
      upper_body_.held_pose = value.position;
      upper_body_.last_tick_ns = std::max(upper_body_.last_tick_ns, capture_ns);
    } else if (enable_signal_connected_ && capture_ns > upper_body_.last_tick_ns) {
      advance_upper_hold(value, capture_ns);
    }
  } else if (!params().handover_only) {
    advance_upper_hold(value, now_ns);
  }
  if (upper_body_.has_any && upper_body_.active) {
    hold_upper(value);
  }
  return true;
}

holoscan::expected<void, holoscan::Error> StartUpOp::compute(holoscan::ExecutionContext& context) {
  try {
    auto result = compute_impl(context);
    if (!result) {
      HOLOSCAN_LOG_ERROR("The startup routine failed: {}.",
                         holoscan::runtime_error_message(result.error()));
    }
    return result;
  } catch (const std::exception& error) {
    HOLOSCAN_LOG_ERROR("The startup routine failed: {}.", error.what());
    throw;
  }
}

holoscan::expected<void, holoscan::Error> StartUpOp::compute_impl(
    holoscan::ExecutionContext& context) {
  auto now = context.clock().now();
  if (!now) {
    return holoscan::make_unexpected(std::move(now).error());
  }
  cache_measured_state();
  latch_enables();

  auto received = joint_command.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  isaac::JointCommandCpuT value = std::move(received->data);
  if (!command_covers_every_joint(value)) {
    if (!warned_command_size_) {
      warned_command_size_ = true;
      HOLOSCAN_LOG_ERROR(
          "A controller command does not carry one value per joint for all of position, velocity, "
          "effort, and gains; it was not passed on to the robot.");
    }
    return {};
  }
  if (params().handover_only && !prepare_handover(value, received->metadata)) return {};
  if (params().handover_only || state_ == State::kRunning) {
    const bool accepted =
        update_upper_body(value, received->metadata.capture_time().timestamp_ns, now->timestamp_ns);
    const auto& metadata = received->metadata;
    holoscan::EmitOptions options{.flags = metadata.flags};
    if (!accepted) options.flags |= holoscan::SampleFlags::kDegraded;
    if (metadata.capture_time().valid()) {
      options.capture_time = metadata.capture_time();
    }
    if (metadata.frame_id != 0U) {
      options.frame_id = metadata.frame_id;
    }
    if (metadata.trace_id != 0U) {
      options.trace = holoscan::TraceContext::root(metadata.trace_id);
    }
    return command.emit(std::move(value), options);
  }
  if (!joint_state_seen_) {
    return {};
  }
  const std::vector<float>& positions = sequencer_->step(&measured_position_, &measured_velocity_);
  const bool at_final = sequencer_->at_final_waypoint();

  isaac::JointCommandCpuT startup_command;
  startup_command.position = positions;
  startup_command.velocity.assign(joint_count(), 0.0F);
  startup_command.effort.assign(joint_count(), 0.0F);
  startup_command.kp = at_final ? hold_kp_ : kp_;
  startup_command.kd = at_final ? hold_kd_ : kd_;

  auto emitted =
      command.emit(std::move(startup_command), holoscan::EmitOptions{.capture_time = *now});
  if (!emitted) {
    return emitted;
  }
  if (at_final) {
    end_startup_if_enabled_or_delayed(now->timestamp_ns);
  }
  return {};
}

}  // namespace isaac::os::ops
