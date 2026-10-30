/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/memory/tensor_storage.hpp>

#include "operators/gr00t_n1_7/gr00t_n1_7_fused_op.h"

namespace isaac::os::ops {
namespace {

Gr00tN17Params valid_params() {
  return Gr00tN17Params{
      .sdk_joint_order = {"joint_0", "joint_1"},
      .state_output_spec = {{"arm", {"joint_0", "joint_1"}}},
      .view_name = "ego_view",
      .crop_height = 2,
      .crop_width = 3,
      .inference = {.num_joints = 2,
                    .models = {{.path = "model.onnx",
                                .inputs = {{"input", "input.video.ego_view"}},
                                .outputs = {{"output", "output.arm"}}}}},
      .num_motors = 2,
      .prediction_horizon = 2,
      .joint_order = {"joint_0", "joint_1"},
      .input_spec = {{"arm", {"joint_0", "joint_1"}}},
      .kp = {{"arm", {10.0F}}},
      .kd = {{"arm", {1.0F}}},
      .inference_period = 0.25,
  };
}

TEST(Gr00tN17Params, AcceptsConsistentConfiguration) {
  EXPECT_TRUE(validate_gr00t_n17_params(valid_params()));
}

TEST(Gr00tN17Params, RejectsMissingModelsAndInconsistentJointConfiguration) {
  Gr00tN17Params params = valid_params();
  params.inference.models.clear();
  EXPECT_FALSE(validate_gr00t_n17_params(params));

  params = valid_params();
  params.num_motors = 3;
  EXPECT_FALSE(validate_gr00t_n17_params(params));

  params = valid_params();
  params.input_spec.at("arm").push_back("unknown_joint");
  EXPECT_FALSE(validate_gr00t_n17_params(params));
}

TEST(Gr00tN17Params, RejectsInvalidDimensionsRateAndGains) {
  Gr00tN17Params params = valid_params();
  params.crop_height = 0;
  EXPECT_FALSE(validate_gr00t_n17_params(params));

  params = valid_params();
  params.prediction_horizon = 0;
  EXPECT_FALSE(validate_gr00t_n17_params(params));

  params = valid_params();
  params.inference_period = 0.0;
  EXPECT_FALSE(validate_gr00t_n17_params(params));

  params = valid_params();
  params.kp.at("arm") = {std::numeric_limits<double>::max()};
  EXPECT_FALSE(validate_gr00t_n17_params(params));
}

TEST(Gr00tN17Params, RejectsPeriodsThatCannotBeRepresentedInNanoseconds) {
  for (const double period : {1e-10, std::numeric_limits<double>::max()}) {
    auto params = valid_params();
    params.align_observations = true;
    params.inference_period = period;
    EXPECT_FALSE(validate_gr00t_n17_params(params));
  }
}

using namespace std::chrono_literals;

std::vector<std::chrono::nanoseconds> selected_frames(
    std::initializer_list<std::chrono::nanoseconds> capture_times) {
  std::optional<std::int64_t> last_period;
  std::vector<std::chrono::nanoseconds> selected;
  constexpr auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(700ms).count();
  for (const auto capture_time : capture_times) {
    if (select_gr00t_n17_frame(capture_time.count(), period_ns, last_period,
                               /*defer_initial_interval=*/true)) {
      selected.push_back(capture_time);
    }
  }
  return selected;
}

TEST(Gr00tN17Cadence, SelectsFirstMatchedFrameAcrossBoundariesWithoutDrift) {
  // Selecting just after 0.7 s must not delay the next boundary until 1.400000001 s.
  EXPECT_EQ(selected_frames({0ns, 700ms - 1ns, 700ms + 1ns, 710ms, 1400ms - 1ns, 1400ms,
                             1400ms + 1ns, 2100ms - 1ns, 2100ms + 1ns}),
            (std::vector<std::chrono::nanoseconds>{700ms + 1ns, 1400ms, 2100ms + 1ns}));
}

TEST(Gr00tN17Cadence, SkipsMissedIntervalsWithoutCatchUpOrMovingTheNextBoundary) {
  // No pair is available in [1.4, 2.1) s. Select once at 2.79 s, then again at 2.8 s.
  EXPECT_EQ(selected_frames({0ms, 710ms, 2790ms, 2790ms, 2795ms, 2800ms, 2810ms}),
            (std::vector<std::chrono::nanoseconds>{710ms, 2790ms, 2800ms}));
}

TEST(Gr00tN17Cadence, ResumesAfterCameraInterruptionOnTheOriginalCadence) {
  // Without camera frames there are no aligned pairs. The next pair arrives at 5.04 s.
  EXPECT_EQ(selected_frames({0ms, 700ms, 720ms, 5040ms, 5050ms, 5599ms, 5600ms, 5610ms}),
            (std::vector<std::chrono::nanoseconds>{700ms, 5040ms, 5600ms}));
}

TEST(Gr00tN17Cadence, WaitsForTheNextBoundaryWhenTheFirstPairArrivesLate) {
  EXPECT_EQ(selected_frames({1680ms, 2099ms, 2101ms, 2110ms, 2800ms}),
            (std::vector<std::chrono::nanoseconds>{2101ms, 2800ms}));
}

std::vector<std::chrono::nanoseconds> selected_unaligned_frames(
    std::initializer_list<std::chrono::nanoseconds> capture_times) {
  std::optional<std::int64_t> last_period;
  std::vector<std::chrono::nanoseconds> selected;
  constexpr auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(200ms).count();
  for (const auto capture_time : capture_times) {
    if (select_gr00t_n17_frame(capture_time.count(), period_ns, last_period,
                               /*defer_initial_interval=*/false)) {
      selected.push_back(capture_time);
    }
  }
  return selected;
}

TEST(Gr00tN17Cadence, RealAndMockAcceptFirstImageImmediately) {
  EXPECT_EQ(selected_unaligned_frames({0ms, 33ms, 199ms, 200ms}),
            (std::vector<std::chrono::nanoseconds>{0ms, 200ms}));
  EXPECT_EQ(selected_unaligned_frames({1680ms, 1799ms, 1800ms}),
            (std::vector<std::chrono::nanoseconds>{1680ms, 1800ms}));
}

TEST(Gr00tN17Cadence, RealAndMockKeepFixedBoundariesDespiteCameraJitter) {
  EXPECT_EQ(selected_unaligned_frames({1ms, 199ms, 201ms, 399ms, 400ms, 599ms, 601ms}),
            (std::vector<std::chrono::nanoseconds>{1ms, 201ms, 400ms, 601ms}));
}

TEST(Gr00tN17Cadence, RealAndMockSkipFramesLostDuringInferenceOrCameraInterruption) {
  // Skipped frames leave the original 200 ms boundaries unchanged.
  EXPECT_EQ(selected_unaligned_frames({0ms, 201ms, 1590ms, 1590ms, 1599ms, 1600ms}),
            (std::vector<std::chrono::nanoseconds>{0ms, 201ms, 1590ms, 1600ms}));
}

TEST(Gr00tN17Cadence, RealAndMockIgnoreRepeatedOrOlderCaptureIntervals) {
  EXPECT_EQ(selected_unaligned_frames({200ms, 200ms, 199ms, 0ms, 399ms, 400ms}),
            (std::vector<std::chrono::nanoseconds>{200ms, 400ms}));
}

TEST(Gr00tN17Output, BuildsJointMajorCpuChunkFromKnownHostBuffers) {
  const std::array<float, 4> position{1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<float, 4> effort{-1.0F, -2.0F, -3.0F, -4.0F};
  const std::array<float, 2> kp{10.0F, 20.0F};
  const std::array<float, 2> kd{1.0F, 2.0F};

  auto chunk = make_joint_command_chunk_cpu(position, effort, kp, kd, 2U, 2U, 123456789LL);

  ASSERT_TRUE(chunk);
  EXPECT_EQ(chunk->joint_count, 2U);
  EXPECT_EQ(chunk->action_count, 2U);
  EXPECT_EQ(chunk->position, std::vector<float>(position.begin(), position.end()));
  EXPECT_EQ(chunk->effort, std::vector<float>(effort.begin(), effort.end()));
  EXPECT_EQ(chunk->image_capture_time_ns, 123456789LL);
  EXPECT_EQ(chunk->kp, std::vector<float>(kp.begin(), kp.end()));
  EXPECT_EQ(chunk->kd, std::vector<float>(kd.begin(), kd.end()));
}

TEST(Gr00tN17Output, RejectsMismatchedHostBufferDimensions) {
  const std::array<float, 3> position{1.0F, 2.0F, 3.0F};
  const std::array<float, 4> effort{-1.0F, -2.0F, -3.0F, -4.0F};
  const std::array<float, 2> gains{1.0F, 2.0F};

  EXPECT_FALSE(make_joint_command_chunk_cpu(position, effort, gains, gains, 2U, 2U, 0));
}

TEST(Gr00tN17Input, AcceptsCpuJointStateAndPackedUint8Image) {
  Gr00tN17Params params = valid_params();
  isaac::JointStateCpuT state;
  state.position = {0.1F, 0.2F};
  std::array<std::uint8_t, 18> pixels{};
  const std::array<std::int64_t, 3> shape{2, 3, 3};
  auto image = holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                                     holoscan::MemoryStorageType::kHost, pixels.data());
  ASSERT_TRUE(image);

  EXPECT_TRUE(validate_gr00t_n17_inputs(params, state, *image));
}

TEST(Gr00tN17Input, RejectsMissingJointsAndInvalidImageLayout) {
  Gr00tN17Params params = valid_params();
  isaac::JointStateCpuT state;
  state.position = {0.1F};
  std::array<float, 18> pixels{};
  const std::array<std::int64_t, 3> shape{2, 3, 3};
  auto image = holoscan::make_tensor(shape, DLDataType{kDLFloat, 32U, 1U}, std::nullopt,
                                     holoscan::MemoryStorageType::kHost, pixels.data());
  ASSERT_TRUE(image);

  EXPECT_FALSE(validate_gr00t_n17_inputs(params, state, *image));

  state.position = {0.1F, 0.2F};
  EXPECT_FALSE(validate_gr00t_n17_inputs(params, state, *image));
}

}  // namespace
}  // namespace isaac::os::ops
