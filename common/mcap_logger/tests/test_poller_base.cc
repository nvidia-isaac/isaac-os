/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <string>
#include <utility>

#include "gtest/gtest.h"

#include "common/mcap_logger/topic_poller.h"

namespace isaac::os::mcap_logger {
namespace {

// PollerBase's constructor is protected; this exposes the one member under test.
class TestPoller final : public PollerBase {
 public:
  TestPoller(std::string topic, std::uint32_t queue_depth)
      : PollerBase(std::move(topic), queue_depth) {}
  using PollerBase::note_sequence;
};

template <typename Action>
std::string CaptureLogs(Action&& action) {
  ::testing::internal::CaptureStdout();
  ::testing::internal::CaptureStderr();
  action();
  return ::testing::internal::GetCapturedStdout() + ::testing::internal::GetCapturedStderr();
}

// Reported for the first drop and then every tenth, naming the topic, carrying the running total
// and saying which option raises the depth. A queue that is too shallow for a burst is fixed by
// reading the log, so the log has to say which topic and which field.
TEST(PollerBaseTest, ReportsDropsOnTheFirstAndEveryTenthNamingTheTopicAndTheOption) {
  TestPoller poller("front_camera", 16);

  const std::string logs = CaptureLogs([&] {
    poller.note_sequence(1);
    // Every later sample skips one: twelve gaps, one sample missed by each.
    for (std::uint64_t sequence = 3; sequence <= 25; sequence += 2) {
      poller.note_sequence(sequence);
    }
  });

  // The leading text keeps "1 so far" from also matching inside "11 so far".
  EXPECT_NE(logs.find("fell behind: 1 so far"), std::string::npos) << logs;
  EXPECT_NE(logs.find("fell behind: 10 so far"), std::string::npos) << logs;
  for (const char* unreported :
       {"fell behind: 2 so far", "fell behind: 5 so far", "fell behind: 9 so far",
        "fell behind: 11 so far", "fell behind: 12 so far"}) {
    EXPECT_EQ(logs.find(unreported), std::string::npos) << unreported << "\n" << logs;
  }
  EXPECT_NE(logs.find("'front_camera'"), std::string::npos) << logs;
  EXPECT_NE(logs.find("queue depth"), std::string::npos) << logs;
  EXPECT_NE(logs.find("16 samples held"), std::string::npos) << logs;
}

TEST(PollerBaseTest, SaysNothingWhileNoSampleIsMissed) {
  TestPoller poller("joint_state", 4);
  const std::string logs = CaptureLogs([&] {
    for (std::uint64_t sequence = 1; sequence <= 20; ++sequence) {
      poller.note_sequence(sequence);
    }
    poller.note_sequence(0);  // Unstamped: not a gap.
  });
  EXPECT_EQ(logs.find("fell behind"), std::string::npos) << logs;
}

// A publisher's sequence starts at 1, so a first sample above 1 means earlier ones were dropped
// before it was received.
TEST(PollerBaseTest, CountsSamplesDroppedBeforeTheFirstReceive) {
  TestPoller poller("front_camera", 4);
  const std::string logs = CaptureLogs([&] { poller.note_sequence(21); });
  EXPECT_NE(logs.find("fell behind: 20 so far"), std::string::npos) << logs;
}

TEST(PollerBaseTest, DoesNotCountANonAdvancingSequenceAsADrop) {
  TestPoller poller("joint_state", 4);
  const std::string logs = CaptureLogs([&] {
    poller.note_sequence(1);
    poller.note_sequence(1);
    poller.note_sequence(2);
    poller.note_sequence(1);  // A restart or reorder.
    poller.note_sequence(3);  // Measured from the highest sequence seen.
  });
  EXPECT_EQ(logs.find("fell behind"), std::string::npos) << logs;
}

}  // namespace
}  // namespace isaac::os::mcap_logger
