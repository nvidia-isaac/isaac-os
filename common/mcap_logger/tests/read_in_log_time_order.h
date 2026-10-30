/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gtest/gtest.h"

#include "mcap/reader.hpp"

namespace isaac::os::mcap_logger {

// Iterates a finished .mcap file in ascending log_time order rather than mcap's default
// ReadOrder::FileOrder (the physical order records happen to sit in the file).
//
// The logger does not write in log_time order and never promised to. A poll thread reads
// log_time from SampleMetadata::publish_timestamp_ns *before* it writes the sample. Samples of one
// topic are written in the order they were received, but every topic has its own poll thread, so
// records of different topics land in the file in whatever order the threads finish. Only log_time
// is authoritative across topics. A test that iterates in file order while asserting increasing
// log_time across topics is therefore asserting a property of thread scheduling, and fails whenever
// two topics' samples are genuinely in flight at once. Today that is rare only because these tests
// publish slowly enough that one sample is usually finished before the next arrives -- timing, not
// a guarantee.
//
// Safe on every file these tests produce, including one written without a chunk index:
// IndexedMessageReader falls back to scanning the file to build the index it needs
// (ReadSummaryMethod::AllowFallbackScan) when none is present.
inline mcap::LinearMessageView ReadInLogTimeOrder(mcap::McapReader& reader) {
  mcap::ReadMessageOptions options;
  options.readOrder = mcap::ReadMessageOptions::ReadOrder::LogTimeOrder;
  // A problem reading the file is a defect in what the logger wrote, so it fails the test rather
  // than being skipped. A truncated chunk or a record for an unknown channel would otherwise read
  // as fewer messages, and any test asserting on whatever did arrive would pass on a partial
  // recording.
  //
  // The one exception is a recording with no messages in it. It has no chunk message indexes to
  // order by, which mcap reports as a problem, and several tests write exactly that on purpose --
  // a sink that never received a sample, or a shutdown before the first one.
  //
  // Passed by const reference but stored by value in LinearMessageView::onProblem_, so this
  // local does not dangle once the view outlives this call.
  const auto on_problem = [](const mcap::Status& problem) {
    if (problem.code == mcap::StatusCode::NoMessageIndexesAvailable) {
      return;
    }
    ADD_FAILURE() << "Unable to read the recording cleanly: " << problem.message;
  };
  return reader.readMessages(on_problem, options);
}

// The same strict reading in the order records physically sit in the file. Only for a test that
// asserts something about that order -- a topic's own messages are written in the order they were
// received -- and never to line different topics up in time: use ReadInLogTimeOrder for that.
inline mcap::LinearMessageView ReadInFileOrder(mcap::McapReader& reader) {
  mcap::ReadMessageOptions options;
  options.readOrder = mcap::ReadMessageOptions::ReadOrder::FileOrder;
  const auto on_problem = [](const mcap::Status& problem) {
    if (problem.code == mcap::StatusCode::NoMessageIndexesAvailable) {
      return;
    }
    ADD_FAILURE() << "Unable to read the recording cleanly: " << problem.message;
  };
  return reader.readMessages(on_problem, options);
}

}  // namespace isaac::os::mcap_logger
