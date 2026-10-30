/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "applications/so101_sinus_motion/real/motion_profile.h"

namespace isaac::os {
namespace {

struct FakeClock {
  std::chrono::steady_clock::time_point time{};
  auto now() const { return time; }
  void sleep_until(std::chrono::steady_clock::time_point next) { time = next; }
};

struct FakeDriver {
  std::vector<std::vector<float>> targets;
  int reads = 0;
  bool fail_feedback = false;
  bool fail_write = false;
  bool stationary = false;
  bool read_present_positions(std::vector<float>& position) const {
    if (position.size() != 6U) throw MotionProfileError("Incorrect feedback buffer size.");
    position = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};
    return true;
  }
  std::vector<std::string> get_ordered_joint_names() const {
    return {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"};
  }
  bool send_action(const std::vector<float>& target) {
    targets.push_back(target);
    return !fail_write;
  }
  bool read_present_positions_and_velocities(std::vector<float>& position,
                                             std::vector<float>& velocity) {
    if (position.size() != 6U || velocity.size() != 6U)
      throw MotionProfileError("Incorrect feedback buffer size.");
    velocity.assign(6U, 0.0F);
    ++reads;
    if (!stationary) position = targets.back();
    return !fail_feedback;
  }
};

TEST(So101MotionProfile, WritesOnlySixTargetsAndPreservesOtherJoints) {
  FakeDriver driver;
  FakeClock clock;
  std::ostringstream output;
  measure_motion_profile(driver, output, clock, [] { return false; });
  ASSERT_EQ(driver.targets.size(), 6U);
  for (std::size_t phase = 0; phase < driver.targets.size(); ++phase) {
    EXPECT_FLOAT_EQ(driver.targets[phase][0], phase % 2 == 0 ? 0.3F : 0.1F);
    EXPECT_EQ((std::vector<float>(driver.targets[phase].begin() + 1, driver.targets[phase].end())),
              (std::vector<float>{0.2F, 0.3F, 0.4F, 0.5F, 0.6F}));
  }
  EXPECT_EQ(driver.reads, 1800);
  EXPECT_EQ(clock.time.time_since_epoch(), std::chrono::seconds(18));
  const auto csv = output.str();
  EXPECT_EQ(std::count(csv.begin(), csv.end(), '\n'), 1801);
}

TEST(So101MotionProfile, CancellationPreventsFurtherTargets) {
  FakeDriver driver;
  FakeClock clock;
  std::ostringstream output;
  measure_motion_profile(driver, output, clock, [&driver] { return driver.reads >= 2; });
  EXPECT_EQ(driver.targets.size(), 1U);
  EXPECT_EQ(driver.reads, 2);
}

TEST(So101MotionProfile, FeedbackFailureStopsImmediately) {
  FakeDriver driver;
  driver.fail_feedback = true;
  FakeClock clock;
  std::ostringstream output;
  EXPECT_THROW(measure_motion_profile(driver, output, clock, [] { return false; }),
               MotionProfileError);
  EXPECT_EQ(driver.targets.size(), 1U);
  EXPECT_EQ(driver.reads, 1);
}

TEST(So101MotionProfile, WriteFailureStopsImmediately) {
  FakeDriver driver;
  driver.fail_write = true;
  FakeClock clock;
  std::ostringstream output;
  EXPECT_THROW(measure_motion_profile(driver, output, clock, [] { return false; }),
               MotionProfileError);
  EXPECT_EQ(driver.targets.size(), 1U);
  EXPECT_EQ(driver.reads, 0);
}

TEST(So101MotionProfile, UnreachedTargetFailsInsteadOfReportingSuccess) {
  FakeDriver driver;
  driver.stationary = true;
  FakeClock clock;
  std::ostringstream output;
  EXPECT_THROW(measure_motion_profile(driver, output, clock, [] { return false; }),
               MotionProfileError);
  EXPECT_EQ(driver.targets.size(), 1U);
  EXPECT_EQ(driver.reads, 300);
}

TEST(So101MotionProfile, CancelledBeforeStartSendsNoTargets) {
  FakeDriver driver;
  FakeClock clock;
  std::ostringstream output;
  measure_motion_profile(driver, output, clock, [] { return true; });
  EXPECT_TRUE(driver.targets.empty());
}

TEST(So101MotionProfile, OutputFailureStopsImmediately) {
  FakeDriver driver;
  FakeClock clock;
  std::ostringstream output;
  output.setstate(std::ios::badbit);
  EXPECT_THROW(measure_motion_profile(driver, output, clock, [] { return false; }),
               MotionProfileError);
  EXPECT_TRUE(driver.targets.empty());
  EXPECT_EQ(driver.reads, 0);
}

TEST(So101MotionProfile, HoldRetriesUntilSuccessful) {
  struct Driver {
    int attempts = 0;
    bool hold_present_position() { return ++attempts == 2; }
  };
  Driver driver;
  EXPECT_TRUE(hold_after_motion_profile(driver));
  EXPECT_EQ(driver.attempts, 2);
}

TEST(So101MotionProfile, HoldFailureIsBounded) {
  struct Driver {
    int attempts = 0;
    bool hold_present_position() {
      ++attempts;
      return false;
    }
  };
  Driver driver;
  EXPECT_FALSE(hold_after_motion_profile(driver));
  EXPECT_EQ(driver.attempts, 3);
}
}  // namespace
}  // namespace isaac::os
