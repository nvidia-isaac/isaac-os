/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "common/pose_tree/pose_tree_history.hpp"

#include <mutex>

#include "common/pose_tree/math/interpolation.hpp"

namespace isaac::os::pose_tree {

PoseTreeEdgeHistory::PoseTreeEdgeHistory(const frame_t lhs, const frame_t rhs,
                                         const int32_t maximum_size, AccessMethod access_method,
                                         TimedPose* buffer)
    : edges_info_(buffer),
      lhs_(lhs),
      rhs_(rhs),
      maximum_size_(maximum_size),
      size_(0),
      pos_(0),
      default_access_method_(access_method) {}

PoseTreeEdgeHistory::PoseTreeEdgeHistory(const frame_t lhs, const frame_t rhs,
                                         const int32_t maximum_size, TimedPose* buffer)
    : edges_info_(buffer),
      lhs_(lhs),
      rhs_(rhs),
      maximum_size_(maximum_size),
      size_(0),
      pos_(0),
      default_access_method_(AccessMethod::kInterpolateLinearly) {}

PoseTreeEdgeHistory::expected_t<void> PoseTreeEdgeHistory::disconnect(const double time,
                                                                      const version_t version) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  const auto maybe_idx = reserve_new_pose(time, version);
  if (!maybe_idx) {
    return unexpected_t(maybe_idx.error());
  }
  const int32_t idx = maybe_idx.value();
  edges_info_[idx].valid = false;
  return expected_t<void>{};
}

PoseTreeEdgeHistory::expected_t<void> PoseTreeEdgeHistory::set(const double time,
                                                               const Pose3d& pose,
                                                               const version_t version) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  const auto maybe_idx = reserve_new_pose(time, version);
  if (!maybe_idx) {
    return unexpected_t(maybe_idx.error());
  }
  const int32_t idx = maybe_idx.value();
  edges_info_[idx].pose = pose;
  edges_info_[idx].valid = true;
  return expected_t<void>{};
}

PoseTreeEdgeHistory::expected_t<int32_t> PoseTreeEdgeHistory::reserve_new_pose(
    const double time, const version_t version) {
  if (maximum_size_ <= 0 || edges_info_ == nullptr) {
    return unexpected_t(Error::kOutOfRange);
  }
  const int32_t last_idx = (pos_ + size_ - 1) % maximum_size_;
  if (size_ != 0 &&
      (edges_info_[last_idx].time >= time || edges_info_[last_idx].version >= version)) {
    return unexpected_t{Error::kOutOfOrder};
  }
  int32_t idx;
  if (size_ == maximum_size_) {
    idx = pos_++;
    if (pos_ == maximum_size_) {
      pos_ = 0;
    }
  } else {
    idx = size_++;
  }
  edges_info_[idx].time = time;
  edges_info_[idx].version = version;
  return idx;
}

PoseTreeEdgeHistory::expected_t<PoseTreeEdgeHistory::TimedPose> PoseTreeEdgeHistory::latest()
    const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  if (size_ == 0) {
    return unexpected_t(Error::kOutOfRange);
  }
  return edges_info_[(pos_ + size_ - 1) % maximum_size_];
}

PoseTreeEdgeHistory::expected_t<PoseTreeEdgeHistory::TimedPose> PoseTreeEdgeHistory::at(
    const int32_t index) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  if (index < 0 || index >= size_) {
    return unexpected_t(Error::kInvalidArgument);
  }
  const int32_t idx = (pos_ + index) % maximum_size_;
  return edges_info_[idx];
}

int32_t PoseTreeEdgeHistory::last_index_before(const double time, const version_t version) const {
  // Perform a binary search in the array to find the last pose which time is lower than `time`.
  // If none exist, then index will be equal to pose_
  int32_t start = pos_;        // included
  int32_t end = pos_ + size_;  // excluded
  while (start + 1 < end) {
    const int32_t mid = (start + end) / 2;
    const auto& mid_info = edges_info_[mid >= maximum_size_ ? mid - maximum_size_ : mid];
    if (mid_info.time > time || mid_info.version > version) {
      end = mid;
    } else {
      start = mid;
    }
  }
  return start;
}

PoseTreeEdgeHistory::expected_t<Pose3d> PoseTreeEdgeHistory::pose_at_end(
    const int32_t index, const int32_t start, const double time, const AccessMethod method) const {
  switch (method) {
    case AccessMethod::kNearest:
    case AccessMethod::kPrevious:
    case AccessMethod::kInterpolateSlerp:
    case AccessMethod::kInterpolateLinearly: {
      return edges_info_[index].pose;
    }
    case AccessMethod::kExtrapolateSlerp:
    case AccessMethod::kExtrapolateLinearly: {
      // Extrapolation continues the motion of the last two entries, so there has to be one before
      // `index`, and it has to be connected.
      if (pos_ == start) {
        return unexpected_t(Error::kOutOfRange);
      }
      const auto& pose1 = edges_info_[(index + maximum_size_ - 1) % maximum_size_];
      if (!pose1.valid) {
        return unexpected_t(Error::kOutOfRange);
      }
      const auto& pose2 = edges_info_[index];
      const double delta_time = time - pose1.time;
      const double total_time = pose2.time - pose1.time;
      return method == AccessMethod::kExtrapolateSlerp
                 ? slerp_interpolate(delta_time / total_time, pose1.pose, pose2.pose)
                 : interpolate(delta_time / total_time, pose1.pose, pose2.pose);
    }
    case AccessMethod::kDefault: {
      return unexpected_t(Error::kInvalidArgument);
    }
  }
  return unexpected_t(Error::kInvalidArgument);
}

PoseTreeEdgeHistory::expected_t<Pose3d> PoseTreeEdgeHistory::pose_between(
    const int32_t index, const int32_t next_index, const double time,
    const AccessMethod method) const {
  const auto& previous = edges_info_[index];
  const auto& next = edges_info_[next_index];
  const double delta_time = time - previous.time;
  const double total_time = next.time - previous.time;
  switch (method) {
    case AccessMethod::kPrevious: {
      return previous.pose;
    }
    case AccessMethod::kNearest: {
      return next.time - time < time - previous.time ? next.pose : previous.pose;
    }
    case AccessMethod::kInterpolateLinearly:
    case AccessMethod::kExtrapolateLinearly: {
      return interpolate(delta_time / total_time, previous.pose, next.pose);
    }
    case AccessMethod::kInterpolateSlerp:
    case AccessMethod::kExtrapolateSlerp: {
      return slerp_interpolate(delta_time / total_time, previous.pose, next.pose);
    }
    case AccessMethod::kDefault: {
      return unexpected_t(Error::kInvalidArgument);
    }
  }
  return unexpected_t(Error::kInvalidArgument);
}

PoseTreeEdgeHistory::expected_t<Pose3d> PoseTreeEdgeHistory::get(const double time,
                                                                 AccessMethod method,
                                                                 const version_t version) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  if (size_ == 0 || edges_info_[pos_].version > version || edges_info_[pos_].time > time) {
    return unexpected_t(Error::kFramesNotLinked);
  }

  const int32_t start = last_index_before(time, version);
  const int32_t index = start % maximum_size_;
  if (!edges_info_[index].valid) {
    return unexpected_t(Error::kFramesNotLinked);
  }
  const int32_t next_index = (start + 1) % maximum_size_;

  if (method == AccessMethod::kDefault) {
    method = default_access_method_;
  }

  // Nothing usable follows `index`, either because it is the newest entry held, or because what
  // follows it is invisible to this version or disconnected. Either way there is no second pose to
  // interpolate towards, which is a case of its own.
  if (next_index == (pos_ + size_) % maximum_size_ || edges_info_[next_index].version > version ||
      !edges_info_[next_index].valid) {
    return pose_at_end(index, start, time, method);
  }
  return pose_between(index, next_index, time, method);
}

bool PoseTreeEdgeHistory::connected() const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return size_ > 0 && edges_info_[(pos_ + size_ - 1) % maximum_size_].valid;
}

void PoseTreeEdgeHistory::reset() {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  pos_ = 0;
  size_ = 0;
}

}  // namespace isaac::os::pose_tree
