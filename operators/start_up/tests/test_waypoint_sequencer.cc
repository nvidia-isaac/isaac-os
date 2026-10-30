/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "operators/start_up/waypoint_sequencer.h"

namespace isaac::os::ops {
namespace {

constexpr std::size_t kJointCount = 2U;

std::vector<WaypointSequencer::Waypoint> make_waypoints() {
  return {
      WaypointSequencer::Waypoint{
          .name = "crouch", .positions = {1.0F, -1.0F}, .interp_steps = 4, .dwell_steps = 2},
      WaypointSequencer::Waypoint{
          .name = "stand", .positions = {2.0F, -2.0F}, .interp_steps = 4, .dwell_steps = 2},
  };
}

TEST(Smoothstep, EasesInAndOutWithinTheUnitInterval) {
  EXPECT_FLOAT_EQ(smoothstep(0.0F), 0.0F);
  EXPECT_FLOAT_EQ(smoothstep(1.0F), 1.0F);
  EXPECT_FLOAT_EQ(smoothstep(0.5F), 0.5F);
  EXPECT_FLOAT_EQ(smoothstep(-1.0F), 0.0F);
  EXPECT_FLOAT_EQ(smoothstep(2.0F), 1.0F);
}

TEST(WaypointSequencer, RejectsARoutineThatDoesNotCoverEveryJoint) {
  auto waypoints = make_waypoints();
  waypoints.front().positions.pop_back();
  EXPECT_THROW(WaypointSequencer(std::move(waypoints), 0.01, kJointCount), std::runtime_error);
  EXPECT_THROW(WaypointSequencer({}, 0.01, kJointCount), std::runtime_error);
}

TEST(WaypointSequencer, HoldsUntilAMeasuredPoseArrives) {
  WaypointSequencer sequencer{make_waypoints(), 0.01, kJointCount};
  const std::vector<float>& command = sequencer.step(nullptr, nullptr);
  EXPECT_EQ(command, (std::vector<float>{0.0F, 0.0F}));
}

TEST(WaypointSequencer, StartsFromTheMeasuredPoseSoTheRobotFeelsNoStep) {
  WaypointSequencer sequencer{make_waypoints(), 0.01, kJointCount};
  const std::vector<float> position{0.25F, -0.25F};
  const std::vector<float> velocity{0.0F, 0.0F};
  EXPECT_EQ(sequencer.step(&position, &velocity), position);
}

TEST(WaypointSequencer, WalksThroughEveryWaypointAndHoldsTheLast) {
  WaypointSequencer sequencer{make_waypoints(), 0.01, kJointCount};
  const std::vector<float> position{0.0F, 0.0F};
  const std::vector<float> velocity{0.0F, 0.0F};

  std::vector<float> command;
  for (int step = 0; step < 100; ++step) {
    command = sequencer.step(&position, &velocity);
  }
  EXPECT_TRUE(sequencer.at_final_waypoint());
  EXPECT_FLOAT_EQ(command[0], 2.0F);
  EXPECT_FLOAT_EQ(command[1], -2.0F);
}

TEST(WaypointSequencer, DoesNotAdvanceWhileTheJointsAreStillMoving) {
  WaypointSequencer sequencer{make_waypoints(), 0.01, kJointCount};
  const std::vector<float> position{0.0F, 0.0F};
  const std::vector<float> moving{5.0F, 5.0F};

  for (int step = 0; step < 100; ++step) {
    sequencer.step(&position, &moving);
  }
  EXPECT_FALSE(sequencer.at_final_waypoint());
}

TEST(WaypointSequencer, StopsAdvancingWhileMeasurementsAreStale) {
  WaypointSequencer sequencer{make_waypoints(), 0.01, kJointCount};
  const std::vector<float> position{0.0F, 0.0F};
  const std::vector<float> velocity{0.0F, 0.0F};
  sequencer.step(&position, &velocity);
  sequencer.step(&position, &velocity);
  const std::vector<float> first_stale = sequencer.step(&position, nullptr);
  const std::vector<float> second_stale = sequencer.step(&position, nullptr);
  EXPECT_EQ(second_stale, first_stale);
  EXPECT_EQ(sequencer.step(nullptr, &velocity), first_stale);
}

}  // namespace
}  // namespace isaac::os::ops
