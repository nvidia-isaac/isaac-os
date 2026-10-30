/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// 137 Hz TargetProducer plus a 1 kHz ReadJointState -> MotionPlanner ->
// SafetyCheck -> WriteMotorCommand control path. ReadJointState also feeds
// TargetProducer. The GPU backend searches 1,024 local IK seeds in parallel.

#if defined(__linux__)
#include <sched.h>
#endif

#include <CLI/CLI.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <ios>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <holoscan/config/codec.hpp>
#include <holoscan/config/document.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "applications/robot_control_benchmark/gpu_contention_op.hpp"
#include "applications/robot_control_benchmark/payloads.hpp"
#include "common/robot_control_benchmark/arm_kinematics.hpp"
#include "common/robot_control_benchmark/control_loop.hpp"
#include "common/robot_control_benchmark/iceoryx2_config.hpp"
#include "common/robot_control_benchmark/joint_safety.hpp"
#include "common/robot_control_benchmark/latency_stats.hpp"
#include "common/robot_control_benchmark/managed_irq_guard.hpp"
#include "common/robot_control_benchmark/motion_backend.hpp"
#include "common/robot_control_benchmark/motor_protocol.hpp"
#include "common/robot_control_benchmark/payloads.hpp"
#include "common/robot_control_benchmark/runtime_support.hpp"
#include "common/robot_control_benchmark/trajectory.hpp"

namespace {

[[nodiscard]] holoscan::expected<void, holoscan::Error> tolerate_dropped_frame(
    holoscan::expected<void, holoscan::Error> published) {
  if (!published && (published.error().code == holoscan::ErrorCode::kBackpressured ||
                     published.error().code == holoscan::ErrorCode::kPublicationIndeterminate)) {
    return {};
  }
  return published;
}

struct BenchmarkState {
  BenchmarkState(isaac::os::benchmark::MotionBackend& motion_backend, std::int64_t warmup_ticks,
                 std::int64_t measured_ticks, std::int64_t control_period_ns)
      : stats(static_cast<std::size_t>(measured_ticks)),
        motion(motion_backend),
        warmup(warmup_ticks),
        samples(measured_ticks),
        period_ns(control_period_ns) {}

  isaac::os::benchmark::LatencyStats stats;
  isaac::os::benchmark::SimulatedRobot robot;
  isaac::os::benchmark::MotionBackend& motion;
  const std::int64_t warmup;
  const std::int64_t samples;
  const std::int64_t period_ns;
  std::uint64_t non_finite = 0;
  std::uint64_t range_clamps = 0;
  std::uint64_t step_clamps = 0;
  std::uint64_t recovery_steps = 0;
};

class BenchmarkStateHandle {
 public:
  explicit BenchmarkStateHandle(BenchmarkState& state) : state_(&state) {}

  [[nodiscard]] BenchmarkState* operator->() const { return state_; }

 private:
  BenchmarkState* state_;
};

}  // namespace

namespace isaac::os::ops {

/// Verifies that an operator starts on its configured partition CPU.
class CpuPlacementCheck {
 public:
  /// Records the CPU required for an operator's partition.
  explicit CpuPlacementCheck(int expected_cpu) : expected_cpu_(expected_cpu) {}

  /// Fails startup when the runtime placed the named operator on a different CPU.
  void verify(std::string_view operator_name) const {
    if (const int actual_cpu = sched_getcpu(); actual_cpu != expected_cpu_) {
      throw std::runtime_error("Operator '" + std::string(operator_name) +
                               "' started on unexpected CPU " + std::to_string(actual_cpu) + '.');
    }
  }

 private:
  int expected_cpu_;
};

class TargetProducerOp final : public holoscan::Operator<> {
 public:
  /// Creates the slower target source with its configured activation period and partition CPU.
  TargetProducerOp(std::int64_t period_ns, int expected_cpu)
      : cpu_placement_(expected_cpu), period_ns_(period_ns) {}

  /// Verifies that the target producer starts on its configured CPU.
  void start() override { cpu_placement_.verify("target producer"); }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in_state, "in_joint_state").queue_depth(1U);
    spec.output(out, "out_target").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::nanoseconds{period_ns_}})
        .input(in_state, holoscan::DrainAll{});
    return result;
  }

  /// Publishes the target for the runtime's scheduled activation after finite state is available.
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    const holoscan::TimePoint scheduled = context.activation_time();
    if (!scheduled.valid()) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kProtocolError,
                          "The target producer requires a valid scheduled activation time."});
    }
    if (!schedule_epoch_) {
      schedule_epoch_ = scheduled;
    }
    const std::int64_t elapsed_ns = scheduled.timestamp_ns - schedule_epoch_->timestamp_ns;
    if (scheduled.clock_id != schedule_epoch_->clock_id || elapsed_ns < 0 ||
        elapsed_ns % period_ns_ != 0) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kProtocolError,
          "Target activations must advance on one valid clock and the configured period grid."});
    }
    // Copy the payload: the selected batch view expires after this callback.
    if (auto batch = in_state.selected_batch(); batch && !batch->empty()) {
      latest_joint_state_ = batch->back().data;
    }
    if (!latest_joint_state_ ||
        !isaac::os::benchmark::measured_joints_fk_finite(latest_joint_state_->position)) {
      return {};
    }
    auto produced = context.clock().now();
    if (!produced) {
      return holoscan::make_unexpected(std::move(produced).error());
    }
    if (produced->clock_id != scheduled.clock_id) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kProtocolError,
                          "The target schedule and publication must use the same valid clock."});
    }
    const std::int64_t tick = elapsed_ns / period_ns_;
    const double trajectory_time_s = static_cast<double>(elapsed_ns) * 1e-9;
    isaac::os::benchmark::Pose6 pose = isaac::os::benchmark::trajectory_pose(trajectory_time_s);
    pose.seq = tick;
    pose.t_src_ns = produced->timestamp_ns;
    pose.produced_ns = produced->timestamp_ns;
    pose.version = ++version_;
    return tolerate_dropped_frame(out.emit(pose));
  }

  holoscan::Input<isaac::os::benchmark::JointStateSample> in_state;
  holoscan::Output<isaac::os::benchmark::Pose6> out;

 private:
  CpuPlacementCheck cpu_placement_;
  std::int64_t period_ns_;
  std::optional<holoscan::TimePoint> schedule_epoch_;
  std::optional<isaac::os::benchmark::JointStateSample> latest_joint_state_;
  std::uint64_t version_ = 0;
};

class ReadJointStateOp final : public holoscan::Operator<> {
 public:
  /// Creates the control source with shared robot state, its period, and its partition CPU.
  ReadJointStateOp(BenchmarkStateHandle state, std::int64_t period_ns, int expected_cpu)
      : cpu_placement_(expected_cpu), state_(state), period_ns_(period_ns) {}

  /// Verifies that the joint-state source starts on its configured CPU.
  void start() override { cpu_placement_.verify("joint-state source"); }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out_joint_state").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::nanoseconds{period_ns_}});
    return result;
  }

  /// Publishes joint state with the runtime's scheduled tick and observed callback-start time.
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    const holoscan::TimePoint scheduled = context.activation_time();
    auto callback_start = context.clock().now();
    if (!callback_start) {
      return holoscan::make_unexpected(std::move(callback_start).error());
    }
    if (!scheduled.valid() || scheduled.clock_id != callback_start->clock_id) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kProtocolError,
          "The scheduled activation and callback start must use the same valid clock."});
    }
    const auto position = state_->robot.read();
    isaac::os::benchmark::JointStateSample state{};
    state.sequence = next_sequence_++;
    state.scheduled_ns = scheduled.timestamp_ns;
    state.callback_start_ns = callback_start->timestamp_ns;
    std::copy(position.begin(), position.end(), state.position);
    return tolerate_dropped_frame(out.emit(state));
  }

  holoscan::Output<isaac::os::benchmark::JointStateSample> out;

 private:
  CpuPlacementCheck cpu_placement_;
  BenchmarkStateHandle state_;
  std::int64_t period_ns_;
  std::int64_t next_sequence_ = 0;
};

class MotionPlannerOp final : public holoscan::Operator<> {
 public:
  /// Creates the motion planner for its configured partition CPU.
  MotionPlannerOp(BenchmarkStateHandle state, int expected_cpu)
      : cpu_placement_(expected_cpu), state_(state) {
    initial_target_ = isaac::os::benchmark::trajectory_pose(0.0);
  }

  /// Verifies that the motion planner starts on its configured CPU.
  void start() override { cpu_placement_.verify("motion planner"); }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in_state, "in_joint_state").queue_depth(1U);
    spec.input(in_target, "in_target").queue_depth(1U);
    spec.output(out, "out_setpoints").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in_state}).input(in_target, holoscan::DrainAll{});
    return result;
  }

  /// Plans one control sample while preserving its authoritative source timing coordinates.
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto sample = in_state.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    // Keep the target payload between slower updates, including empty DrainAll batches.
    if (auto batch = in_target.selected_batch(); batch && !batch->empty()) {
      latest_target_ = batch->back().data;
    }
    const isaac::os::benchmark::Pose6* target =
        latest_target_ ? &*latest_target_ : &initial_target_;
    const isaac::os::benchmark::JointStateSample& state = sample->data;
    isaac::os::benchmark::MotionInput input{};
    std::copy(std::begin(state.position), std::end(state.position), input.q_warm_start.begin());
    std::copy(std::begin(state.position), std::end(state.position), input.measured_q.begin());
    input.target_p[0] = target->x;
    input.target_p[1] = target->y;
    input.target_p[2] = target->z;
    input.target_R = isaac::os::benchmark::rpy_to_rotation(target->rx, target->ry, target->rz);

    auto target_use = context.clock().now();
    if (!target_use) {
      return holoscan::make_unexpected(std::move(target_use).error());
    }
    const std::int64_t target_use_ns = target_use->timestamp_ns;
    state_->motion.compute(input, context.cuda_stream().get());

    isaac::os::benchmark::JointSetpoints sp{};
    sp.seq = state.sequence;
    sp.t_src_ns = state.scheduled_ns;
    for (int j = 0; j < isaac::os::benchmark::kNumJoints; ++j) {
      sp.q[j] = input.q_warm_start[static_cast<std::size_t>(j)];
      sp.measured_q[j] = state.position[j];
    }
    sp.callback_start_ns = state.callback_start_ns;
    sp.target_produced_ns = target->produced_ns;
    sp.target_use_ns = target_use_ns;
    sp.target_version = target->version;
    return tolerate_dropped_frame(out.emit(sp));
  }

  holoscan::Input<isaac::os::benchmark::JointStateSample> in_state;
  holoscan::Input<isaac::os::benchmark::Pose6> in_target;
  holoscan::Output<isaac::os::benchmark::JointSetpoints> out;

 private:
  CpuPlacementCheck cpu_placement_;
  BenchmarkStateHandle state_;
  std::optional<isaac::os::benchmark::Pose6> latest_target_;
  isaac::os::benchmark::Pose6 initial_target_{};
};

class SafetyCheckOp final : public holoscan::Operator<> {
 public:
  /// Creates the safety checker for its configured partition CPU.
  explicit SafetyCheckOp(int expected_cpu) : cpu_placement_(expected_cpu) {}

  /// Verifies that the safety checker starts on its configured CPU.
  void start() override { cpu_placement_.verify("safety checker"); }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in_setpoints").queue_depth(1U);
    spec.output(out, "out_commands").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  /// Clamps one setpoint sample and forwards its source timing with the encoded command batch.
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    isaac::os::benchmark::MotorCommandBatch batch{};
    const isaac::os::benchmark::JointSetpoints& sp = sample->data;
    std::array<double, isaac::os::benchmark::kNumJoints> safe_q{};
    std::copy(std::begin(sp.q), std::end(sp.q), safe_q.begin());
    isaac::os::benchmark::JointState measured{};
    measured.sequence = sp.seq;
    std::copy(std::begin(sp.measured_q), std::end(sp.measured_q), measured.position.begin());
    const auto safety_result = safety_.validate_and_clamp(safe_q, measured);
    isaac::os::benchmark::JointSetpoints safe_sp = sp;
    std::copy(safe_q.begin(), safe_q.end(), safe_sp.q);
    isaac::os::benchmark::encode_batch(safe_sp, batch);
    batch.callback_start_ns = sp.callback_start_ns;
    batch.target_produced_ns = sp.target_produced_ns;
    batch.target_use_ns = sp.target_use_ns;
    batch.target_version = sp.target_version;
    std::copy(safe_q.begin(), safe_q.end(), batch.safe_q);
    batch.safety_non_finite = safety_result.non_finite;
    batch.safety_range_clamps = safety_result.range_clamps;
    batch.safety_step_clamps = safety_result.step_clamps;
    batch.safety_recovery_steps = safety_result.recovery_steps;
    return tolerate_dropped_frame(out.emit(batch));
  }

  holoscan::Input<isaac::os::benchmark::JointSetpoints> in;
  holoscan::Output<isaac::os::benchmark::MotorCommandBatch> out;

 private:
  CpuPlacementCheck cpu_placement_;
  [[no_unique_address]] isaac::os::benchmark::JointSafety safety_;
};

class WriteMotorCommandOp final : public holoscan::Operator<> {
 public:
  /// Creates the motor-command sink for its configured partition CPU.
  WriteMotorCommandOp(BenchmarkStateHandle state, int expected_cpu)
      : cpu_placement_(expected_cpu), state_(state) {}

  /// Verifies that the motor-command sink starts on its configured CPU.
  void start() override { cpu_placement_.verify("motor-command sink"); }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in_commands").queue_depth(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  /// Writes one command and records latency, period, and delivery counters.
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto sample = in.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const isaac::os::benchmark::MotorCommandBatch& batch = sample->data;
    std::memcpy(send_buffer_.data(), batch.packets, isaac::os::benchmark::kBatchBytes);
    for (const std::byte value : send_buffer_) {
      checksum_ ^= std::to_integer<std::uint8_t>(value);
    }
    std::array<double, isaac::os::benchmark::kNumJoints> safe_q{};
    std::copy(std::begin(batch.safe_q), std::end(batch.safe_q), safe_q.begin());
    state_->robot.write(safe_q);
    if (state_->stats.count() >= static_cast<std::size_t>(state_->samples)) {
      return {};
    }
    auto completion = context.clock().now();
    if (!completion) {
      return holoscan::make_unexpected(std::move(completion).error());
    }
    const std::int64_t now = completion->timestamp_ns;
    // Track versions across warmup: a target whose true first command completed during warmup is
    // left-censored and must not be mislabeled as a target-response sample in the measured window.
    const bool first_command_for_target =
        target_response_tracker_.record(batch.target_produced_ns, batch.target_version);
    std::uint64_t coalesced_ticks = 0;
    std::uint64_t dropped_ticks = 0;
    if (last_sequence_ >= 0) {
      // A schedule gap includes runtime coalescing and downstream drops, while the source sequence
      // gap includes only downstream drops; their difference is the coalesced activation count.
      const std::int64_t sequence_advance = batch.seq - last_sequence_;
      const std::int64_t schedule_advance = batch.t_src_ns - last_scheduled_ns_;
      if (sequence_advance <= 0 || schedule_advance < state_->period_ns ||
          schedule_advance % state_->period_ns != 0) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kProtocolError,
            "Control activations must advance in sequence on the configured schedule."});
      }
      const std::int64_t schedule_ticks = schedule_advance / state_->period_ns;
      if (schedule_ticks < sequence_advance) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kProtocolError,
            "The scheduled control advance cannot be shorter than its sequence advance."});
      }
      coalesced_ticks = static_cast<std::uint64_t>(schedule_ticks - sequence_advance);
      dropped_ticks = static_cast<std::uint64_t>(sequence_advance - 1);
    }
    last_sequence_ = batch.seq;
    last_scheduled_ns_ = batch.t_src_ns;
    if (received_ >= state_->warmup) {
      if (!state_->stats.add_control_tick(isaac::os::benchmark::ControlTickObservation{
              .sequence = batch.seq,
              .scheduled_ns = batch.t_src_ns,
              .callback_start_ns = batch.callback_start_ns,
              .completion_ns = now,
              .target_produced_ns = batch.target_produced_ns,
              .target_use_ns = batch.target_use_ns,
              .target_version = batch.target_version,
              .first_command_for_target = first_command_for_target,
              .coalesced_ticks = coalesced_ticks,
              .dropped_ticks = dropped_ticks,
              .late_command = now > batch.t_src_ns + state_->period_ns,
          })) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kProtocolError,
                            "Unable to record the command: the observation buffer is full or its "
                            "timing is invalid."});
      }
      state_->non_finite += batch.safety_non_finite;
      state_->range_clamps += batch.safety_range_clamps;
      state_->step_clamps += batch.safety_step_clamps;
      state_->recovery_steps += batch.safety_recovery_steps;
      if (state_->stats.count() == static_cast<std::size_t>(state_->samples)) {
        context.request_completion();
      }
    }
    ++received_;
    return {};
  }

  holoscan::Input<isaac::os::benchmark::MotorCommandBatch> in;

 private:
  CpuPlacementCheck cpu_placement_;
  BenchmarkStateHandle state_;
  std::array<std::byte, isaac::os::benchmark::kBatchBytes> send_buffer_{};
  std::uint8_t checksum_ = 0;
  std::int64_t last_scheduled_ns_ = 0;
  std::int64_t last_sequence_ = -1;
  std::int64_t received_ = 0;
  isaac::os::benchmark::TargetResponseTracker target_response_tracker_;
};

}  // namespace isaac::os::ops

namespace isaac::os {
namespace {

struct Options {
  std::filesystem::path config_path;
  double rate_hz = 1000.0;
  double target_rate_hz = 137.0;
  std::int64_t samples = 20000;
  std::int64_t warmup = 1000;
  std::int64_t timeout_s = 60;
  std::string scenario = "clean";
  std::string motion_backend = "cpu-ik";
  std::string iceoryx2_prefix;
  std::string label;
  std::string label_prefix = "isaac-os-robot-control";
  std::string raw_observations_file;
  int target_cpu = -1;
  int control_cpu = -1;
  int load_cpu = -1;
  int pump_cpu = 11;
  int control_fifo_priority = 10;
};

/// Reports an invalid or unreadable benchmark configuration.
class BenchmarkConfigError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Unwraps one native Holoscan configuration result with its structured error text.
template <typename T>
T require_config(holoscan::config::ConfigResult<T> result) {
  if (!result) {
    throw BenchmarkConfigError(holoscan::config::format_error(result.error()));
  }
  return std::move(*result);
}

/// Applies an optional YAML value without replacing the built-in default when the key is absent.
template <typename T>
void apply_optional_config_default(const holoscan::config::Document& document,
                                   const CLI::Option* option, const char* path, T& value) {
  if (!option->empty()) {
    return;
  }
  auto configured = document.try_get<T>(path);
  if (configured) {
    value = std::move(*configured);
  } else if (configured.error().code() != holoscan::config::ConfigErrorCode::kPathNotFound) {
    throw BenchmarkConfigError(holoscan::config::format_error(configured.error()));
  }
}

/// Applies a required YAML value only when the corresponding CLI option was not supplied.
template <typename T>
void apply_config_default(const holoscan::config::Document& document, const CLI::Option* option,
                          const char* path, T& value) {
  if (option->empty()) {
    value = require_config(document.try_get<T>(path));
  }
}

/// Validates timing, sample, scenario, and CPU-placement options before graph construction.
void validate_options(const Options& options) {
  if (options.target_cpu < 0 || options.control_cpu < 0 || options.load_cpu < 0 ||
      options.pump_cpu < 0) {
    throw std::invalid_argument(
        "The target, control, load, and pump CPU assignments must be non-negative.");
  }
  if (options.target_cpu == options.control_cpu || options.target_cpu == options.load_cpu ||
      options.control_cpu == options.load_cpu) {
    throw std::invalid_argument("The target, control, and load paths must use distinct CPUs.");
  }
  if (options.control_fifo_priority < 1 ||
      options.control_fifo_priority > ::sched_get_priority_max(SCHED_FIFO)) {
    throw std::invalid_argument(
        "--control-fifo-priority must be a valid positive SCHED_FIFO priority.");
  }
  const auto valid_rate = [](double rate) {
    return std::isfinite(rate) && rate > 0.0 && rate <= 1e9;
  };
  if (!valid_rate(options.rate_hz)) {
    throw std::runtime_error("--rate must be finite, positive, and no greater than 1 GHz.");
  }
  if (!valid_rate(options.target_rate_hz)) {
    throw std::runtime_error("--target-rate must be finite, positive, and no greater than 1 GHz.");
  }
  if (options.samples <= 0) {
    throw std::runtime_error("--samples must be a positive integer.");
  }
  if (options.warmup < 0) {
    throw std::runtime_error("--warmup must be a non-negative integer.");
  }
  if (options.timeout_s <= 0) {
    throw std::runtime_error("--timeout must be a positive integer.");
  }
  if (options.scenario != "clean" && options.scenario != "contended" &&
      options.scenario != "isolated") {
    throw std::invalid_argument("--scenario must be clean, contended, or isolated.");
  }
  if (options.motion_backend != "cpu-ik" && options.motion_backend != "gpu-ik") {
    throw std::invalid_argument("--motion-backend must be cpu-ik or gpu-ik.");
  }
}

/// Parses CLI overrides and fills omitted values from an optional Holoscan YAML document.
void parse_options(CLI::App& app, Options& options, int argc, char** argv) {
  app.add_option("--config-path", options.config_path, "YAML defaults; CLI flags override")
      ->required();
  CLI::Option* rate = app.add_option("--rate", options.rate_hz, "Control-loop target rate in Hz")
                          ->check(CLI::PositiveNumber);
  CLI::Option* target_rate = app.add_option("--target-rate", options.target_rate_hz,
                                            "Independent target producer rate in Hz")
                                 ->check(CLI::PositiveNumber);
  CLI::Option* samples = app.add_option("--samples", options.samples, "Measured ticks after warmup")
                             ->check(CLI::PositiveNumber);
  CLI::Option* warmup =
      app.add_option("--warmup", options.warmup, "Ticks discarded before measuring")
          ->check(CLI::NonNegativeNumber);
  CLI::Option* timeout = app.add_option("--timeout", options.timeout_s,
                                        "Abort if the sample target is not met in time")
                             ->check(CLI::PositiveNumber);
  CLI::Option* scenario =
      app.add_option("--scenario", options.scenario, "GPU scenario: clean, contended, or isolated");
  CLI::Option* motion_backend = app.add_option("--motion-backend", options.motion_backend,
                                               "Motion planner IK backend: cpu-ik or gpu-ik");
  CLI::Option* target_cpu =
      app.add_option("--target-cpu", options.target_cpu, "CPU for the target partition")
          ->check(CLI::NonNegativeNumber);
  CLI::Option* control_cpu =
      app.add_option("--control-cpu", options.control_cpu, "CPU for the control partition")
          ->check(CLI::NonNegativeNumber);
  CLI::Option* load_cpu =
      app.add_option("--load-cpu", options.load_cpu, "CPU for the GPU-load partition")
          ->check(CLI::NonNegativeNumber);
  CLI::Option* pump_cpu =
      app.add_option("--pump-cpu", options.pump_cpu, "CPU for the channel pump (default: 11)")
          ->check(CLI::NonNegativeNumber);
  CLI::Option* control_fifo_priority =
      app.add_option("--control-fifo-priority", options.control_fifo_priority,
                     "Required SCHED_FIFO priority for the control partition")
          ->check(CLI::PositiveNumber);
  CLI::Option* iceoryx2_prefix =
      app.add_option("--iceoryx2-prefix", options.iceoryx2_prefix,
                     "Shared-memory namespace; defaults to a robot-control prefix scoped by UID");
  app.add_option("--label", options.label,
                 "Full BENCHCSV label; when empty, uses --label-prefix plus the scenario");
  CLI::Option* label_prefix =
      app.add_option("--label-prefix", options.label_prefix,
                     "BENCHCSV label prefix (scenario suffix appended unless --label is set)");
  CLI::Option* raw_observations =
      app.add_option("--raw-observations-file", options.raw_observations_file,
                     "Write one bench-observations/v3 CSV row per measured tick");
  app.parse(argc, argv);

  const holoscan::config::Document document =
      holoscan::config_io::load(options.config_path, holoscan::config_io::Format::kYaml);
  apply_config_default(document, rate, "/rate", options.rate_hz);
  apply_config_default(document, target_rate, "/target_rate", options.target_rate_hz);
  apply_config_default(document, samples, "/samples", options.samples);
  apply_config_default(document, warmup, "/warmup", options.warmup);
  apply_config_default(document, timeout, "/timeout", options.timeout_s);
  apply_config_default(document, scenario, "/scenario", options.scenario);
  apply_config_default(document, load_cpu, "/load_cpu", options.load_cpu);
  apply_config_default(document, target_cpu, "/target_cpu", options.target_cpu);
  apply_config_default(document, control_cpu, "/control_cpu", options.control_cpu);
  apply_optional_config_default(document, pump_cpu, "/pump_cpu", options.pump_cpu);
  apply_optional_config_default(document, control_fifo_priority, "/control_fifo_priority",
                                options.control_fifo_priority);
  apply_optional_config_default(document, iceoryx2_prefix, "/iceoryx2_prefix",
                                options.iceoryx2_prefix);
  apply_optional_config_default(document, motion_backend, "/motion_backend",
                                options.motion_backend);
  apply_optional_config_default(document, label_prefix, "/label_prefix", options.label_prefix);
  apply_optional_config_default(document, raw_observations, "/raw_observations_file",
                                options.raw_observations_file);

  validate_options(options);
  if (options.iceoryx2_prefix.empty()) {
    options.iceoryx2_prefix =
        "isaac_os_robot_control_u" + std::to_string(static_cast<unsigned long>(::getuid())) + "_";
  }
}

/// Selects the active CUDA device without assuming device zero.
holoscan::DeviceId current_cuda_device() {
  int ordinal = -1;
  const cudaError_t status = cudaGetDevice(&ordinal);
  if (status != cudaSuccess) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                 std::string("Unable to select the active CUDA device: ") +
                                     cudaGetErrorString(status) + ".");
  }
  const holoscan::DeviceId device{ordinal};
  if (!device.valid()) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kNotFound,
                                 "CUDA did not report a valid active device.");
  }
  return device;
}

struct GpuIsolationAllocation {
  std::uint32_t control_sm_count;
  std::uint32_t load_sm_count;
};

/// Gives control one structured SM quantum and assigns the remaining structured SMs to load.
GpuIsolationAllocation configure_gpu_isolation(holoscan::CompileOptions& options) {
  const holoscan::DeviceId device = current_cuda_device();
  const auto capability = holoscan::query_cuda_green_context_capability(device);
  if (!capability) {
    throw holoscan::RuntimeError(capability.error().code,
                                 "The isolated scenario requires CUDA Green Context support: " +
                                     holoscan::runtime_error_message(capability.error()));
  }
  if (capability->structured_sm_quantum == 0U || capability->max_structured_context_count < 2U) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kIncompatible,
        "The isolated scenario requires at least two structured CUDA SM groups.");
  }
  const std::uint64_t structured_sm_count =
      static_cast<std::uint64_t>(capability->max_structured_context_count) *
      capability->structured_sm_quantum;
  if (structured_sm_count > capability->total_sm_count) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kIncompatible,
        "CUDA reported inconsistent Green Context capacity for the active device.");
  }
  const std::uint32_t control_sm_count = capability->structured_sm_quantum;
  const auto load_sm_count = static_cast<std::uint32_t>(structured_sm_count - control_sm_count);
  options.deployment.bind_cuda_green_context("bench.control",
                                             {.device = device, .sm_count = control_sm_count});
  options.deployment.bind_cuda_green_context("bench.load",
                                             {.device = device, .sm_count = load_sm_count});
  return {.control_sm_count = control_sm_count, .load_sm_count = load_sm_count};
}

/// Confirms that compilation placed control and load on their separate requested SM sets.
void validate_gpu_isolation(const holoscan::ExecutionPlan& plan,
                            const GpuIsolationAllocation& allocation) {
  const auto& requests = plan.cuda_green_context_requests();
  const auto find_request = [&requests](std::string_view name) {
    return std::ranges::find_if(
        requests, [name](const auto& candidate) { return candidate.authored_name == name; });
  };
  const auto control_request = find_request("bench.control");
  const auto load_request = find_request("bench.load");
  if (control_request == requests.end() || load_request == requests.end() ||
      !control_request->accepted || !load_request->accepted ||
      control_request->accepted->sm_count != allocation.control_sm_count ||
      load_request->accepted->sm_count != allocation.load_sm_count ||
      control_request->accepted->device != load_request->accepted->device) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kIncompatible,
        "The isolated scenario did not receive both requested CUDA Green Contexts.");
  }

  const auto& partitions = plan.execution_partitions();
  const auto find_partition = [&partitions](std::string_view name) {
    return std::ranges::find_if(partitions, [name](const auto& candidate) {
      return candidate.authored_name.has_value() && *candidate.authored_name == name;
    });
  };
  const auto control_partition = find_partition("bench.control");
  const auto load_partition = find_partition("bench.load");
  if (control_partition == partitions.end() || load_partition == partitions.end()) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kNotFound,
                                 "The isolated scenario is missing a required partition.");
  }

  const auto& assignments = plan.cuda_green_context_assignments();
  const auto find_assignment = [&assignments](holoscan::ExecutionPartitionId partition) {
    return std::ranges::find_if(assignments, [partition](const auto& candidate) {
      return candidate.partition == partition;
    });
  };
  const auto control_assignment = find_assignment(control_partition->id);
  const auto load_assignment = find_assignment(load_partition->id);
  if (control_assignment == assignments.end() ||
      control_assignment->device != control_request->accepted->device ||
      control_assignment->source != holoscan::CudaGreenContextAssignmentSource::kDeveloperBinding) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible,
                                 "The isolated scenario did not bind the control CUDA resource.");
  }
  if (load_assignment == assignments.end() ||
      load_assignment->device != load_request->accepted->device ||
      load_assignment->pool_id != control_assignment->pool_id ||
      load_assignment->green_context_id == control_assignment->green_context_id ||
      load_assignment->source != holoscan::CudaGreenContextAssignmentSource::kDeveloperBinding) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible,
                                 "The isolated scenario did not bind the load CUDA resource.");
  }

  const auto& streams = plan.cuda_compute_streams();
  const auto find_stream = [&streams](holoscan::ExecutionPartitionId partition) {
    return std::ranges::find_if(streams, [partition](const auto& stream) {
      return stream.partition == partition && stream.green_context_id.has_value();
    });
  };
  const auto control_stream = find_stream(control_partition->id);
  const auto load_stream = find_stream(load_partition->id);
  if (control_stream == streams.end() || load_stream == streams.end() ||
      control_stream->device != control_assignment->device ||
      load_stream->device != load_assignment->device ||
      *control_stream->green_context_id != control_assignment->green_context_id ||
      *load_stream->green_context_id != load_assignment->green_context_id) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kIncompatible,
        "The isolated scenario did not materialize both partitioned CUDA streams.");
  }

  const auto& resources = plan.cuda_green_context_resources();
  const auto find_resource = [&resources](std::string_view green_context_id) {
    return std::ranges::find_if(resources, [green_context_id](const auto& resource) {
      return resource.green_context_id == green_context_id;
    });
  };
  const auto control_resource = find_resource(control_assignment->green_context_id);
  const auto load_resource = find_resource(load_assignment->green_context_id);
  if (control_resource == resources.end() || load_resource == resources.end() ||
      control_resource->device != control_assignment->device ||
      load_resource->device != load_assignment->device ||
      control_resource->pool_id != control_assignment->pool_id ||
      load_resource->pool_id != load_assignment->pool_id ||
      control_resource->role != holoscan::CudaGreenContextResourceRole::kDedicatedPartition ||
      load_resource->role != holoscan::CudaGreenContextResourceRole::kDedicatedPartition ||
      control_resource->accepted_sm_count != allocation.control_sm_count ||
      load_resource->accepted_sm_count != allocation.load_sm_count) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kIncompatible,
        "The isolated scenario did not separate the control and load CUDA resources.");
  }
}

/// Reports that the compiled runtime plan changed a required benchmark placement.
class ThreadPlacementError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Requires the compiled plan to retain one requested partition placement exactly.
void validate_thread_placement(const holoscan::ExecutionPlan& plan, std::string_view name, int cpu,
                               holoscan::SchedulingPolicy policy, std::optional<int> priority) {
  const auto placement = std::ranges::find_if(
      plan.partition_placements(),
      [name](const auto& candidate) { return candidate.authored_name == name; });
  if (placement == plan.partition_placements().end() || !placement->accepted ||
      !placement->accepted->affinity ||
      placement->accepted->affinity->cores !=
          std::vector<std::uint32_t>{static_cast<std::uint32_t>(cpu)} ||
      placement->accepted->scheduling.policy != policy ||
      placement->accepted->scheduling.priority != priority) {
    throw ThreadPlacementError(
        "The requested CPU and scheduling placement was not accepted for partition '" +
        std::string{name} + "'.");
  }
}

/// Describes an unsuccessful run after all callbacks have joined.
std::string_view run_completion_error(holoscan::RunTermination termination, bool timed_out,
                                      std::size_t samples, std::size_t expected_samples) {
  if (termination != holoscan::RunTermination::kStopped &&
      termination != holoscan::RunTermination::kCompleted &&
      termination != holoscan::RunTermination::kRequestedCompleted) {
    return "The robot-control graph did not run successfully.";
  }
  if (timed_out) {
    return "The control loop did not reach the sample target before the timeout.";
  }
  if (samples != expected_samples) {
    return "The control loop stopped before collecting all requested samples.";
  }
  return {};
}

/// Builds and runs the benchmark graph with validated runtime options.
int run(const Options& options) {
  std::vector<unsigned> irq_cpus{static_cast<unsigned>(options.target_cpu),
                                 static_cast<unsigned>(options.control_cpu),
                                 static_cast<unsigned>(options.pump_cpu)};
  if (options.scenario != "clean") {
    irq_cpus.push_back(static_cast<unsigned>(options.load_cpu));
  }
  const benchmark::ManagedIrqGuard irq_guard(std::move(irq_cpus), {"nvme"});
  benchmark::configure_iceoryx2(options.iceoryx2_prefix);
  const auto period_ns = static_cast<std::int64_t>(1e9 / options.rate_hz);
  const auto target_period_ns = static_cast<std::int64_t>(1e9 / options.target_rate_hz);

  auto motion_backend = isaac::os::benchmark::create_motion_backend(options.motion_backend);
  auto state =
      std::make_shared<BenchmarkState>(*motion_backend, options.warmup, options.samples, period_ns);

  holoscan::Graph graph{"robot-control-benchmark"};
  const auto target =
      graph.op<ops::TargetProducerOp>("target_producer", target_period_ns, options.target_cpu);
  const auto read = graph.op<ops::ReadJointStateOp>(
      "read_joint_state", BenchmarkStateHandle{*state}, period_ns, options.control_cpu);
  const auto motion = graph.op<ops::MotionPlannerOp>("motion_planner", BenchmarkStateHandle{*state},
                                                     options.control_cpu);
  const auto safety = graph.op<ops::SafetyCheckOp>("safety_check", options.control_cpu);
  const auto write = graph.op<ops::WriteMotorCommandOp>(
      "write_motor_command", BenchmarkStateHandle{*state}, options.control_cpu);

  graph.partition("bench.target").add(target);
  // The serial partition owns the robot and statistics; neither needs per-tick synchronization.
  graph.partition("bench.control").add(read).add(motion).add(safety).add(write);
  if (options.scenario != "clean") {
    const auto load = graph.op<ops::GpuContentionOp>("gpu_contention");
    graph.partition("bench.load").add(load);
  }

  const holoscan::ConnectionOptions drop_oldest_one{
      .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U};
  graph.add_flow(target->out, motion->in_target, drop_oldest_one);
  graph.add_flow(read->out, target->in_state, drop_oldest_one);
  graph.add_flow(read->out, motion->in_state, drop_oldest_one);
  graph.add_flow(motion->out, safety->in, drop_oldest_one);
  graph.add_flow(safety->out, write->in, drop_oldest_one);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions compile_options;
  compile_options.deployment.bind_thread_placement(
      "bench.target",
      holoscan::ThreadPlacement{
          .affinity = holoscan::CPUAffinity{static_cast<std::uint32_t>(options.target_cpu)}});
  compile_options.deployment.bind_thread_placement(
      "bench.control",
      holoscan::ThreadPlacement{
          .affinity = holoscan::CPUAffinity{static_cast<std::uint32_t>(options.control_cpu)},
          .scheduling = holoscan::ThreadScheduling{
              .policy = holoscan::SchedulingPolicy::kFIFO,
              .priority = options.control_fifo_priority,
          }});
  if (options.scenario != "clean") {
    compile_options.deployment.bind_thread_placement(
        "bench.load", holoscan::ThreadPlacement{.affinity = holoscan::CPUAffinity{
                                                    static_cast<std::uint32_t>(options.load_cpu)}});
  }
  compile_options.deployment.bind_support_thread_affinity(
      holoscan::SupportThreadRole::kChannelPump,
      holoscan::CPUAffinity{static_cast<std::uint32_t>(options.pump_cpu)});
  std::optional<GpuIsolationAllocation> isolation;
  if (options.scenario == "isolated") {
    isolation = configure_gpu_isolation(compile_options);
  }

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("Unable to compile the robot-control graph.\n{}", plan.json());
    return 1;
  }
  if (isolation) {
    validate_gpu_isolation(plan, *isolation);
    HOLOSCAN_LOG_INFO(
        "CUDA Green Context allocation: control receives {} SMs and GPU contention receives {} "
        "SMs.",
        isolation->control_sm_count, isolation->load_sm_count);
  }
  HOLOSCAN_LOG_INFO("Authored partitions: bench.target, bench.control{}.",
                    options.scenario == "clean" ? "" : ", bench.load");

  validate_thread_placement(plan, "bench.target", options.target_cpu,
                            holoscan::SchedulingPolicy::kOther, std::nullopt);
  validate_thread_placement(plan, "bench.control", options.control_cpu,
                            holoscan::SchedulingPolicy::kFIFO,
                            std::optional<int>{options.control_fifo_priority});
  if (options.scenario != "clean") {
    validate_thread_placement(plan, "bench.load", options.load_cpu,
                              holoscan::SchedulingPolicy::kOther, std::nullopt);
  }
  HOLOSCAN_LOG_INFO(
      "Isaac OS robot-control: loop {:.0f} Hz, target {:.0f} Hz, motion={}, scenario={}, "
      "control CPU={}, FIFO priority={}, {} ticks after {} warmup ticks.",
      options.rate_hz, options.target_rate_hz, motion_backend->name(), options.scenario,
      options.control_cpu, options.control_fifo_priority, options.samples, options.warmup);

  holoscan::RunSession session = holoscan::run_async(plan);
  HOLOSCAN_LOG_INFO(
      "Channel pump affinity requested on CPU {}; main and runtime controller retain inherited "
      "affinity.",
      options.pump_cpu);

  const auto print_diagnostics = [&] {
    const auto vertices = plan.graph().vertices();
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      std::string_view vertex_name{"unknown"};
      for (const auto& vertex : vertices) {
        if (vertex.id == diagnostic.source.vertex) {
          vertex_name = vertex.name;
          break;
        }
      }
      HOLOSCAN_LOG_ERROR("{} at {} (service {}).", diagnostic.token, vertex_name,
                         diagnostic.service);
    }
  };

  const bool timed_out = !session.wait_for(std::chrono::seconds{options.timeout_s});
  if (timed_out) {
    session.request_stop();
  }
  // Join every callback before inspecting the partition-owned statistics and safety counters.
  session.wait();

  // Capture before raw CSV output or reporting can cause disk IRQs.
  irq_guard.check();

  const std::string_view run_error =
      run_completion_error(session.termination(), timed_out, state->stats.count(),
                           static_cast<std::size_t>(options.samples));
  if (!run_error.empty()) {
    HOLOSCAN_LOG_ERROR("{}", run_error);
    print_diagnostics();
  }

  if (!options.raw_observations_file.empty()) {
    try {
      state->stats.write_observations_csv(options.raw_observations_file);
    } catch (const std::ios_base::failure& error) {
      HOLOSCAN_LOG_ERROR("{}", error.what());
      return 1;
    }
  }
  if (!run_error.empty()) {
    return 1;
  }

  const std::string label =
      options.label.empty()
          ? options.label_prefix + "-" + motion_backend->name() + "-" + options.scenario
          : options.label;
  state->stats.report(label, options.scenario, options.rate_hz, options.target_rate_hz);
  HOLOSCAN_LOG_INFO(
      "Safety checks reported {} non-finite results, {} range clamps, {} step clamps, and {} "
      "recovery steps.",
      state->non_finite, state->range_clamps, state->step_clamps, state->recovery_steps);
  return 0;
}

}  // namespace
}  // namespace isaac::os

int main(int argc, char** argv) {
  // Allow up to 32 CUDA compute/copy work queues to reduce serialization between streams.
  // This improved isolated control latency on Thor. Set before CUDA initialization and
  // preserve explicit environment overrides.
  setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", /*overwrite=*/0);

  isaac::os::Options options;
  CLI::App app{"Measure 7-DoF arm control-loop latency on Holoscan 5"};
  try {
    isaac::os::parse_options(app, options, argc, argv);
    return isaac::os::run(options);
  } catch (const CLI::ParseError& error) {
    return app.exit(error);
  } catch (const std::exception& error) {
    HOLOSCAN_LOG_ERROR("{}", error.what());
    return 1;
  }
}
