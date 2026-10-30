/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Array subscript access for frame name copying is intentional.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)

#include "common/pose_tree/pose_tree.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace isaac::os::pose_tree {

namespace {

/// Reserve a key list up front, reporting failure rather than throwing.
///
/// Each of these shadows a map and is reserved to the same maximum, so this runs once during
/// init() and the list never grows again.
template <typename T>
bool reserve_keys(std::vector<T>& keys, std::size_t capacity) {
  try {
    keys.reserve(capacity);
  } catch (const std::exception&) {
    return false;
  }
  return keys.capacity() >= capacity;
}

/// Append to a key list without letting it grow.
///
/// The bound is the point: the list is reserved to the capacity of the map it shadows, so running
/// out of room here means that map is full too, and silently reallocating would hide it.
template <typename T, typename U>
[[nodiscard]] bool push_key(std::vector<T>& keys, U&& key) {
  if (keys.size() >= keys.capacity()) {
    return false;
  }
  keys.push_back(std::forward<U>(key));
  return true;
}

}  // namespace

const char* PoseTree::error_to_str(Error error) {
  switch (error) {
    case Error::kInvalidArgument:
      return "Invalid Argument";
    case Error::kOutOfMemory:
      return "Out of Memory";
    case Error::kFrameNotFound:
      return "Frame not found";
    case Error::kAlreadyExists:
      return "Edge already exists";
    case Error::kCyclingDependency:
      return "Cycling dependency";
    case Error::kFramesNotLinked:
      return "Frames are not linked";
    case Error::kPoseOutOfOrder:
      return "Pose updated out of order";
    case Error::kLogicError:
      return "Logic Error";
  }
  return "Invalid Error Code";
}

PoseTree::expected_t<void> PoseTree::init(const int32_t number_frames, const int32_t number_edges,
                                          const int32_t history_length,
                                          const int32_t default_number_edges,
                                          const int32_t default_history_length,
                                          const int32_t edges_chunk_size,
                                          const int32_t history_chunk_size) {
  // Check the parameters are valid.
  if (number_frames <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (number_edges <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (history_length <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (default_number_edges <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (default_history_length <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (edges_chunk_size <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (history_chunk_size <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }

  std::unique_lock<std::shared_timed_mutex> lock(mutex_);

  init_params_ = {number_frames,          number_edges,     history_length,    default_number_edges,
                  default_history_length, edges_chunk_size, history_chunk_size};
  default_number_edges_ = default_number_edges;
  default_history_length_ = default_history_length;
  {  // Initialize the history map
    const auto result = histories_map_.initialize(number_edges);
    if (!result) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the frame map
    const auto result = frame_map_.initialize(number_frames, number_frames * 2);
    if (!result) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the history memory management
    const auto result = histories_management_.allocate(number_edges, edges_chunk_size);
    if (!result) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the poses memory management
    const auto result = poses_management_.allocate(history_length, history_chunk_size);
    if (!result) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the callbacks
    // TODO(bbutin): We should use a parameter for the size of the callbacks
    const bool result1 = reserve_keys(set_edge_callbacks_keys_, 1024);
    const auto result2 = set_edge_callbacks_.initialize(1024);
    if (!result1 || !result2) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the create frame callbacks
    // TODO(bbutin): We should use a parameter for the size of the callbacks
    const bool result1 = reserve_keys(create_frame_callbacks_keys_, 1024);
    const auto result2 = create_frame_callbacks_.initialize(1024);
    if (!result1 || !result2) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the edges map
    const auto result1 = edges_map_.initialize(number_edges, 2 * number_edges);
    const bool result2 = reserve_keys(edges_map_keys_, number_edges);
    if (!result1 || !result2) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the edges map
    const auto result1 = name_to_uid_map_.initialize(number_frames, 2 * number_frames);
    const bool result2 = reserve_keys(name_to_uid_map_keys_, number_frames);
    if (!result1 || !result2) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }

  frames_stack_.reset(new (std::nothrow) frame_t[number_frames]);
  if (frames_stack_.get() == nullptr) {
    return unexpected_t(Error::kOutOfMemory);
  }

  hint_version_ = 0;
  version_ = 1;
  frame_cb_latest_uid_ = 0;
  edge_cb_latest_uid_ = 0;
  next_frame_id_ = 1;
  frame_id_increment_ = 1;

  return expected_t<void>{};
}

PoseTree::expected_t<void> PoseTree::set_multithreading_info(const frame_t start_frame_id,
                                                             const frame_t increment) {
  if (start_frame_id == 0) {
    HOLOSCAN_LOG_DEBUG(
        "Invalid start_frame_id: {}. Value must be greater than 0 since frame ID 0 "
        "is reserved for invalid frames",
        start_frame_id);
    return unexpected_t(Error::kInvalidArgument);
  }
  if (increment == 0 || std::numeric_limits<uint64_t>::max() / increment <
                            static_cast<uint64_t>(frame_map_.capacity())) {
    HOLOSCAN_LOG_DEBUG("Invalid increment: {}. Value must be between 1 and {}", increment,
                       std::numeric_limits<uint64_t>::max() / frame_map_.capacity());
    return unexpected_t(Error::kInvalidArgument);
  }
  next_frame_id_ = start_frame_id;
  frame_id_increment_ = increment;
  return {};
}

void PoseTree::deinit() {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  frames_stack_.reset(nullptr);
  // Replaced rather than emptied, which is what actually returns them to their starting state.
  // initialize() gives a map fresh storage but leaves its recorded size alone, so a map merely
  // re-initialized would carry the old count into the new tree and reach capacity that much
  // early; and until it is replaced its contents are still there to be found, so a frame from
  // the previous tree would answer to find_frame after the tree it belonged to was torn down.
  frame_map_ = HashMap<frame_t, FrameInfo>();
  name_to_uid_map_ = HashMap<FrameName, frame_t, FrameNameHash>();
  edges_map_ = HashMap<std::pair<frame_t, frame_t>, history_t, PairHash<frame_t, frame_t>>();
  histories_map_ = UniqueIndexMap<PoseTreeEdgeHistory>();
  create_frame_callbacks_ = UniqueIndexMap<CreateFrameCallback>();
  set_edge_callbacks_ = UniqueIndexMap<SetEdgeCallback>();
  histories_management_ = FirstFitAllocator<history_t>();
  poses_management_ = FirstFitAllocator<PoseTreeEdgeHistory::TimedPose>();
  edges_map_keys_.clear();
  name_to_uid_map_keys_.clear();
  create_frame_callbacks_keys_.clear();
  set_edge_callbacks_keys_.clear();
}

PoseTree::version_t PoseTree::get_pose_tree_version() const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return version_;
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::find_frame(std::string_view name) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return find_frame_impl(name);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::find_frame_impl(std::string_view name) const {
  const auto it = name_to_uid_map_.get(FrameName(name));
  if (it) {
    return it.value();
  }
  return unexpected_t(Error::kFrameNotFound);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::find_or_create_frame(std::string_view name) {
  return create_frame_impl(name, default_number_edges_, true, nullptr);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::find_or_create_frame(std::string_view name,
                                                                       const int32_t number_edges) {
  return create_frame_impl(name, number_edges, true, nullptr);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::create_frame_with_id(const frame_t id,
                                                                       std::string_view name) {
  return create_frame_impl(name, default_number_edges_, false, &id);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::create_frame(std::string_view name,
                                                               const int32_t number_edges) {
  return create_frame_impl(name, number_edges, false, nullptr);
}
PoseTree::expected_t<PoseTree::frame_t> PoseTree::create_frame(std::string_view name) {
  return create_frame_impl(name, default_number_edges_, false, nullptr);
}
PoseTree::expected_t<PoseTree::frame_t> PoseTree::create_frame(const int32_t number_edges) {
  return create_frame_impl({}, number_edges, false, nullptr);
}
PoseTree::expected_t<PoseTree::frame_t> PoseTree::create_frame() {
  return create_frame_impl({}, default_number_edges_, false, nullptr);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::reserve_frame_uid(const std::string_view name,
                                                                    const frame_t* id) {
  if (id != nullptr) {
    if (*id == 0) {
      HOLOSCAN_LOG_ERROR("Invalid frame_id requested 0.");
      return unexpected_t(Error::kInvalidArgument);
    }
    if (frame_map_.has(*id)) {
      HOLOSCAN_LOG_DEBUG("ID {} requested for '{}' conflicts with existing frame '{}'", *id, name,
                         frame_map_.get(*id).value().name);
      return unexpected_t(Error::kAlreadyExists);
    }
    return *id;
  }
  // The map is known not to be full by the time we get here, so one of the next size() + 1
  // candidates has to be free.
  for (int i = 0; i <= frame_map_.size(); i++) {
    const frame_t uid = next_frame_id_;
    next_frame_id_ += frame_id_increment_;
    if (!frame_map_.has(uid)) {
      return uid;
    }
  }
  // This can only happen if the increment * number_frames is bigger than 2^64
  return unexpected_t(Error::kLogicError);
}

PoseTree::expected_t<void> PoseTree::assign_frame_name(FrameInfo* frame,
                                                       const std::string_view name) const {
  if (!name.empty()) {
    for (size_t index = 0; index < name.size(); index++) {
      frame->name[index] = name[index];
    }
    frame->name_length = static_cast<int32_t>(name.size());
    return {};
  }
  // Auto generate a name
  const int length_required =
      snprintf(nullptr, 0, "%s%zu", kAutoGeneratedFrameNamePrefix, frame->uid);
  if (length_required > kFrameNameMaximumLength) {
    return unexpected_t(Error::kOutOfMemory);
  }
  const int length_used = snprintf(frame->name, length_required + 1, "%s%zu",
                                   kAutoGeneratedFrameNamePrefix, frame->uid);
  if (length_required != length_used) {
    return unexpected_t(Error::kLogicError);
  }
  frame->name_length = length_used;
  return {};
}

void PoseTree::notify_create_frame_callbacks(const frame_t uid) const {
  std::shared_lock<std::shared_timed_mutex> callbacks_lock(create_frame_callbacks_mutex_);
  for (const uid_t cid : create_frame_callbacks_keys_) {
    const auto cb = create_frame_callbacks_.try_get(cid);
    if (cb) {
      (*cb.value())(uid);
    }
  }
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::create_frame_impl(std::string_view name,
                                                                    const int32_t number_edges,
                                                                    const bool find_or_create,
                                                                    const frame_t* id) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  if (auto existing = find_frame_impl(name)) {
    if (find_or_create) {
      return existing;
    }
    HOLOSCAN_LOG_ERROR("Frame {} already exists.", name);
    return unexpected_t(Error::kAlreadyExists);
  }
  if (frame_map_.size() == frame_map_.capacity()) {
    return unexpected_t(Error::kOutOfMemory);
  }
  // A name may neither start with the character reserved for generated names nor be longer than a
  // frame can store.
  if (!name.empty() &&
      (name[0] == kAutoGeneratedFrameNamePrefix[0] || name.size() > kFrameNameMaximumLength)) {
    return unexpected_t(Error::kInvalidArgument);
  }
  const auto maybe_uid = reserve_frame_uid(name, id);
  if (!maybe_uid) {
    return unexpected_t(maybe_uid.error());
  }
  const frame_t uid = maybe_uid.value();

  const auto history = histories_management_.acquire(number_edges);
  if (!history) {
    return unexpected_t(Error::kOutOfMemory);
  }
  // Everything the frame has taken so far, handed back. Called on each failure below, because a
  // frame that is only half created has to leave nothing behind: no allocation, no name in the
  // lookup, no frame. Leaving any of it would let a later create_frame see the name as taken, or
  // the allocator run out against a frame nobody can reach.
  const auto unwind = [&] {
    histories_management_.release(history.value().first);
    frame_map_.erase(uid);
  };
  const auto maybe_frame = frame_map_.insert(uid, {});
  if (!maybe_frame) {
    unwind();
    return unexpected_t(Error::kLogicError);
  }
  FrameInfo* frame = maybe_frame.value();
  frame->history = history.value().first;
  frame->maximum_number_edges = history.value().second;
  frame->number_edges = 0;
  frame->hint_version = version_;
  frame->distance_to_root = 0;
  frame->node_to_root = uid;
  frame->uid = uid;
  frame->root = frame->uid;
  if (const auto named = assign_frame_name(frame, name); !named) {
    unwind();
    return unexpected_t(named.error());
  }
  if (!name_to_uid_map_.insert(FrameName(frame->name_view()), frame->uid)) {
    unwind();
    return unexpected_t(Error::kOutOfMemory);
  }
  if (!push_key(name_to_uid_map_keys_, uid)) {
    name_to_uid_map_.erase(FrameName(frame->name_view()));
    unwind();
    return unexpected_t(Error::kOutOfMemory);
  }
  // Release the lock to avoid deadlock in the callback
  lock.unlock();
  notify_create_frame_callbacks(uid);

  return uid;
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(const frame_t lhs,
                                                                 const frame_t rhs,
                                                                 const int32_t maximum_length) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_edges_impl(lhs, rhs, maximum_length, PoseTreeEdgeHistory::AccessMethod::kDefault);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(const frame_t lhs,
                                                                 const frame_t rhs) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_edges_impl(lhs, rhs, default_history_length_,
                           PoseTreeEdgeHistory::AccessMethod::kDefault);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(
    const frame_t lhs, const frame_t rhs, const int32_t maximum_length,
    PoseTreeEdgeHistory::AccessMethod method) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_edges_impl(lhs, rhs, maximum_length, method);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(
    const frame_t lhs, const frame_t rhs, PoseTreeEdgeHistory::AccessMethod method) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_edges_impl(lhs, rhs, default_history_length_, method);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(const std::string_view lhs,
                                                                 const std::string_view rhs,
                                                                 const int32_t maximum_length) {
  return create_edges(lhs, rhs, maximum_length, PoseTreeEdgeHistory::AccessMethod::kDefault);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(const std::string_view lhs,
                                                                 const std::string_view rhs) {
  return create_edges(lhs, rhs, default_history_length_,
                      PoseTreeEdgeHistory::AccessMethod::kDefault);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(
    const std::string_view lhs, const std::string_view rhs,
    PoseTreeEdgeHistory::AccessMethod method) {
  return create_edges(lhs, rhs, default_history_length_, method);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(
    const std::string_view lhs, const std::string_view rhs, const int32_t maximum_length,
    PoseTreeEdgeHistory::AccessMethod method) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  const auto lhs_frame = find_frame_impl(lhs);
  if (!lhs_frame) {
    HOLOSCAN_LOG_WARN("Pose frame {} not found", lhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  const auto rhs_frame = find_frame_impl(rhs);
  if (!rhs_frame) {
    HOLOSCAN_LOG_WARN("Pose frame {} not found", rhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  return create_edges_impl(lhs_frame.value(), rhs_frame.value(), maximum_length, method);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges_impl(
    frame_t lhs, frame_t rhs, const int32_t maximum_length,
    PoseTreeEdgeHistory::AccessMethod method) {
  if (lhs == rhs) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (lhs > rhs) {
    std::swap(lhs, rhs);
  }
  // Make sure the frames exists
  const auto lhs_it = frame_map_.try_get(lhs);
  const auto rhs_it = frame_map_.try_get(rhs);
  if (!lhs_it || !rhs_it) {
    return unexpected_t(Error::kFrameNotFound);
  }
  // Make sure the edges do not exist yet
  if (edges_map_.has({lhs, rhs})) {
    return unexpected_t(Error::kAlreadyExists);
  }
  // Check both frame have room for one more edge
  FrameInfo* lhs_frame = lhs_it.value();
  FrameInfo* rhs_frame = rhs_it.value();
  if (lhs_frame->number_edges == lhs_frame->maximum_number_edges ||
      rhs_frame->number_edges == rhs_frame->maximum_number_edges) {
    return unexpected_t(Error::kOutOfMemory);
  }
  const auto lhs_edge = poses_management_.acquire(maximum_length);
  if (!lhs_edge) {
    return unexpected_t(Error::kOutOfMemory);
  }
  const auto uid =
      method == PoseTreeEdgeHistory::AccessMethod::kDefault
          ? histories_map_.emplace(lhs, rhs, lhs_edge.value().second, lhs_edge.value().first)
          : histories_map_.emplace(lhs, rhs, lhs_edge.value().second, method,
                                   lhs_edge.value().first);
  if (!uid) {
    poses_management_.release(lhs_edge.value().first);
    return unexpected_t(Error::kOutOfMemory);
  }
  // The edge now exists in three places, and each later failure has to undo everything before it.
  // An edge left in edges_map_ without its key would be invisible to get_edge_names but still
  // reported as existing, so a retry would fail with kAlreadyExists against an edge nobody can
  // reach; a history left behind holds its pose storage for the life of the tree.
  auto res = edges_map_.insert({lhs, rhs}, uid.value());
  if (!res) {
    histories_map_.erase(uid.value());
    poses_management_.release(lhs_edge.value().first);
    return unexpected_t(Error::kOutOfMemory);
  }
  if (!push_key(edges_map_keys_, std::pair<frame_t, frame_t>{lhs, rhs})) {
    edges_map_.erase({lhs, rhs});
    histories_map_.erase(uid.value());
    poses_management_.release(lhs_edge.value().first);
    return unexpected_t(Error::kOutOfMemory);
  }
  lhs_frame->history[lhs_frame->number_edges++] = uid.value();
  rhs_frame->history[rhs_frame->number_edges++] = uid.value();
  return ++version_;
}

// TODO(ben): This function could be smarter, calling delete_edge_impl on each edge require multiple
// access to a map + rerooting many time. This operation is quite rare, so for now we can keep a
// simpler implementation but we should consider making it smarter.
PoseTree::expected_t<PoseTree::version_t> PoseTree::delete_frame(const frame_t uid) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  const auto uid_it = frame_map_.try_get(uid);
  if (!uid_it) {
    return unexpected_t(Error::kFrameNotFound);
  }
  FrameInfo* frame = uid_it.value();
  while (frame->number_edges) {
    const auto history = histories_map_.try_get(frame->history[frame->number_edges - 1]);
    if (!history) {
      // This should never happen, assert?
      return unexpected_t(Error::kLogicError);
    }
    // Checked rather than ignored: delete_edge_impl reports failure without shortening the frame,
    // so an ignored error would leave this loop spinning on the same edge forever. The edges
    // already removed stay removed, which is the state a caller retrying delete_frame needs.
    const auto removed = delete_edge_impl(history.value()->lhs(), history.value()->rhs(), version_);
    if (!removed) {
      return unexpected_t(removed.error());
    }
  }

  if (!histories_management_.release(frame->history)) {
    // This should never happen, assert?
    return unexpected_t(Error::kLogicError);
  }

  name_to_uid_map_.erase(FrameName(frame->name_view()));
  for (size_t i = 0; i < name_to_uid_map_keys_.size(); i++) {
    if (name_to_uid_map_keys_[i] == uid) {
      name_to_uid_map_keys_.erase(name_to_uid_map_keys_.begin() + i);
      break;
    }
  }
  frame_map_.erase(uid);

  return ++version_;
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::delete_frame(const std::string_view name) {
  frame_t uid;
  {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    const auto maybe_uid = find_frame_impl(name);
    if (!maybe_uid) {
      return unexpected_t(Error::kFrameNotFound);
    }
    uid = maybe_uid.value();
  }
  return delete_frame(uid);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::delete_edge(const frame_t lhs,
                                                                const frame_t rhs) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return delete_edge_impl(lhs, rhs, ++version_);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::delete_edge(const std::string_view lhs,
                                                                const std::string_view rhs) {
  frame_t lhs_uid, rhs_uid;
  {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    const auto maybe_lhs_uid = find_frame_impl(lhs);
    if (!maybe_lhs_uid) {
      return unexpected_t(Error::kFrameNotFound);
    }
    const auto maybe_rhs_uid = find_frame_impl(rhs);
    if (!maybe_rhs_uid) {
      return unexpected_t(Error::kFrameNotFound);
    }
    lhs_uid = maybe_lhs_uid.value();
    rhs_uid = maybe_rhs_uid.value();
  }
  return delete_edge(lhs_uid, rhs_uid);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::delete_edge_impl(frame_t lhs, frame_t rhs,
                                                                     const version_t version) {
  if (lhs > rhs) {
    std::swap(lhs, rhs);
  }
  // Make sure both the frame and edge exists.
  const auto lhs_it = frame_map_.try_get(lhs);
  const auto rhs_it = frame_map_.try_get(rhs);
  if (!lhs_it || !rhs_it) {
    return unexpected_t(Error::kFrameNotFound);
  }
  const auto lhs_rhs = edges_map_.get({lhs, rhs});
  if (!lhs_rhs) {
    return unexpected_t(Error::kFramesNotLinked);
  }
  // Helper pointer to the frame infos.
  FrameInfo* lhs_frame = lhs_it.value();
  FrameInfo* rhs_frame = rhs_it.value();

  const auto history = histories_map_.try_get(lhs_rhs.value());
  if (!history) {
    // This should never happen, assert?
    return unexpected_t(Error::kLogicError);
  }
  // Locate the edge in both frames before touching either. Searching and shrinking in one pass
  // would leave lhs already shortened when rhs turns out not to hold the edge, and the frames
  // would then disagree about how many edges they have with no way back.
  int32_t lhs_slot = -1;
  for (int32_t id = lhs_frame->number_edges - 1; id >= 0; id--) {
    if (lhs_frame->history[id] == lhs_rhs.value()) {
      lhs_slot = id;
      break;
    }
  }
  int32_t rhs_slot = -1;
  for (int32_t id = rhs_frame->number_edges - 1; id >= 0; id--) {
    if (rhs_frame->history[id] == lhs_rhs.value()) {
      rhs_slot = id;
      break;
    }
  }
  if (lhs_slot < 0 || rhs_slot < 0) {
    // The edge is in edges_map_ but not in one of the frames it relates, which no sequence of
    // public calls can produce. Nothing has been modified yet, so the tree is left as it was.
    return unexpected_t(Error::kLogicError);
  }
  // Everything that can fail has now been checked, so the rest runs to completion. Freeing the
  // pose storage is the one step that cannot be undone, which is why it waits until here.
  if (!poses_management_.release(history.value()->data())) {
    // This should never happen, assert?
    return unexpected_t(Error::kLogicError);
  }
  // Move the last edge into the freed slot and drop the last one.
  lhs_frame->history[lhs_slot] = lhs_frame->history[--lhs_frame->number_edges];
  rhs_frame->history[rhs_slot] = rhs_frame->history[--rhs_frame->number_edges];

  edges_map_.erase({lhs, rhs});
  histories_map_.erase(lhs_rhs.value());
  for (size_t i = 0; i < edges_map_keys_.size(); i++) {
    if (edges_map_keys_[i].first == lhs && edges_map_keys_[i].second == rhs) {
      edges_map_keys_.erase(edges_map_keys_.begin() + i);
      break;
    }
  }

  // Recompute the root of the subtree which got disconnected.
  if (lhs_frame->node_to_root == rhs) {
    update_root(lhs);
  } else {
    update_root(rhs);
  }

  return version;
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::disconnect_edge(frame_t lhs, frame_t rhs,
                                                                    const double time) {
  if (lhs > rhs) {
    std::swap(lhs, rhs);
  }
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  // Make sure both the frame and edge exists.
  const auto lhs_it = frame_map_.try_get(lhs);
  const auto rhs_it = frame_map_.try_get(rhs);
  if (!lhs_it || !rhs_it) {
    return unexpected_t(Error::kFrameNotFound);
  }
  const auto lhs_rhs = edges_map_.get({lhs, rhs});
  if (!lhs_rhs) {
    return unexpected_t(Error::kFramesNotLinked);
  }
  const auto history = histories_map_.try_get(lhs_rhs.value());
  if (!history) {
    // This should never happen, assert?
    return unexpected_t(Error::kLogicError);
  }
  const auto result = history.value()->disconnect(time, version_ + 1);
  if (!result) {
    return unexpected_t(Error::kPoseOutOfOrder);
  }
  // Recompute the root of the subtree which got disconnected.
  if (lhs_it.value()->node_to_root == rhs) {
    update_root(lhs);
  } else {
    update_root(rhs);
  }
  return ++version_;
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::disconnect_edge(const std::string_view lhs,
                                                                    const std::string_view rhs,
                                                                    const double time) {
  frame_t lhs_uid, rhs_uid;
  {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    const auto maybe_lhs_uid = find_frame_impl(lhs);
    if (!maybe_lhs_uid) {
      return unexpected_t(Error::kFrameNotFound);
    }
    const auto maybe_rhs_uid = find_frame_impl(rhs);
    if (!maybe_rhs_uid) {
      return unexpected_t(Error::kFrameNotFound);
    }
    lhs_uid = maybe_lhs_uid.value();
    rhs_uid = maybe_rhs_uid.value();
  }
  return disconnect_edge(lhs_uid, rhs_uid, time);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::disconnect_frame(const frame_t uid,
                                                                     const double time) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  const auto uid_it = frame_map_.try_get(uid);
  if (!uid_it) {
    HOLOSCAN_LOG_WARN("No pose frame found with UID {}", uid);
    return unexpected_t(Error::kFrameNotFound);
  }
  FrameInfo* frame = uid_it.value();
  // Go through all the edges and attempt to delete them
  bool failed = false;
  for (int32_t edge = 0; edge < frame->number_edges; edge++) {
    const auto history = histories_map_.try_get(frame->history[edge]);
    if (!history) {
      // This should never happen, assert?
      return unexpected_t(Error::kLogicError);
    }
    if (!history.value()->disconnect(time, version_ + 1)) {
      failed = true;
      continue;
    }
    if (history.value()->rhs() == uid) {
      update_root(history.value()->lhs());
    } else {
      update_root(history.value()->rhs());
    }
  }
  update_root(uid);
  ++version_;
  if (failed) {
    return unexpected_t(Error::kPoseOutOfOrder);
  }
  return version_;
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::disconnect_frame(const std::string_view name,
                                                                     const double time) {
  frame_t uid;
  {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    const auto maybe_uid = find_frame_impl(name);
    if (!maybe_uid) {
      return unexpected_t(Error::kFrameNotFound);
    }
    uid = maybe_uid.value();
  }
  return disconnect_frame(uid, time);
}

PoseTree::expected_t<std::string_view> PoseTree::get_frame_name(const frame_t uid) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_frame_name_unlocked(uid);
}

PoseTree::expected_t<std::string_view> PoseTree::get_frame_name_unlocked(const frame_t uid) const {
  const auto uid_it = frame_map_.try_get(uid);
  if (!uid_it) {
    HOLOSCAN_LOG_WARN("No pose frame found with UID {}", uid);
    return unexpected_t(Error::kFrameNotFound);
  }
  return uid_it.value()->name_view();
}

PoseTree::expected_t<PoseTree::InitParameters> PoseTree::get_init_parameters() const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  if (frames_stack_ == nullptr) {
    return unexpected_t(Error::kLogicError);
  }
  return init_params_;
}

PoseTree::expected_t<void> PoseTree::update_root(const frame_t root) {
  int32_t size = 0;
  frame_t* stack = frames_stack_.get();
  auto it = frame_map_.try_get(root);
  if (!it || stack == nullptr) {
    return unexpected_t(Error::kLogicError);
  }
  it.value()->node_to_root = root;
  it.value()->root = root;
  it.value()->distance_to_root = 0;
  it.value()->hint_version = hint_version_;
  stack[size++] = root;
  bool valid = true;
  // DFS implementation
  while (size > 0) {
    auto* frame = frame_map_.try_get(stack[--size]).value();
    // Go through all the edges
    for (int edge = 0; edge < frame->number_edges; edge++) {
      const auto history = histories_map_.try_get(frame->history[edge]);
      if (!history) {
        continue;
      }
      // Ignore not connected edges
      if (!history.value()->connected()) {
        continue;
      }
      it = frame_map_.try_get(history.value()->rhs() == frame->uid ? history.value()->lhs()
                                                                   : history.value()->rhs());
      if (!it) {
        valid = false;
        continue;
      }
      // Make sure we haven't visited this edge before (it means it is where we come from).
      if (it.value()->hint_version == hint_version_) {
        continue;
      }
      // Update the information
      it.value()->root = root;
      it.value()->hint_version = hint_version_;
      it.value()->node_to_root = frame->uid;
      it.value()->distance_to_root = 1 + frame->distance_to_root;
      // Add to the stack.
      stack[size++] = it.value()->uid;
    }
  }
  hint_version_++;
  if (!valid) {
    // Should never happen, assert?
    return unexpected_t(Error::kLogicError);
  }
  return expected_t<void>{};
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::set(std::string_view lhs, std::string_view rhs,
                                                        const double time,
                                                        const Pose3d& lhs_T_rhs) {
  frame_t lhs_frame, rhs_frame;
  {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    const auto maybe_lhs_frame = find_frame_impl(lhs);
    if (!maybe_lhs_frame) {
      HOLOSCAN_LOG_WARN("Pose frame {} not found", lhs);
      return unexpected_t(Error::kFrameNotFound);
    }
    lhs_frame = maybe_lhs_frame.value();
    const auto maybe_rhs_frame = find_frame_impl(rhs);
    if (!maybe_rhs_frame) {
      HOLOSCAN_LOG_WARN("Pose frame {} not found", rhs);
      return unexpected_t(Error::kFrameNotFound);
    }
    rhs_frame = maybe_rhs_frame.value();
  }
  return set(lhs_frame, rhs_frame, time, lhs_T_rhs);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::set(const frame_t lhs, const frame_t rhs,
                                                        const double time,
                                                        const Pose3d& lhs_T_rhs) {
  if (lhs > rhs) {
    return set(rhs, lhs, time, lhs_T_rhs.inverse());
  }
  if (!lhs_T_rhs.translation.allFinite() ||
      !is_almost_one(lhs_T_rhs.rotation.quaternion().squaredNorm())) {
    HOLOSCAN_LOG_WARN("Invalid pose between: {} and {}", lhs, rhs);
    return unexpected_t(Error::kInvalidArgument);
  }
  // Will contain the version to return
  version_t version;
  {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    // Make sure both the frame and edge exists.
    const auto lhs_it = frame_map_.try_get(lhs);
    const auto rhs_it = frame_map_.try_get(rhs);
    if (!lhs_it || !rhs_it) {
      HOLOSCAN_LOG_WARN("Pose frame UID {} or UID {} not found", lhs, rhs);
      return unexpected_t(Error::kFrameNotFound);
    }
    // Helper pointer to the frame infos.
    FrameInfo* lhs_frame = lhs_it.value();
    FrameInfo* rhs_frame = rhs_it.value();

    auto lhs_rhs = edges_map_.get({lhs, rhs});
    // If the history does not exist yet, let's create it
    if (!lhs_rhs) {
      auto result = create_edges_impl(lhs, rhs, default_history_length_,
                                      PoseTreeEdgeHistory::AccessMethod::kDefault);
      // It might fail due to a loop or memory issue.
      if (!result) {
        HOLOSCAN_LOG_DEBUG("Unable to set pose: Unable to create edges for '{}' ({}) -> '{}' ({})",
                           lhs_frame->name_view(), lhs, rhs_frame->name_view(), rhs);
        return result;
      }
      lhs_rhs = edges_map_.get({lhs, rhs});
      if (!lhs_rhs) {
        HOLOSCAN_LOG_DEBUG(
            "Unable to set pose: Unable to get edge map item for '{}' ({}) -> '{}' ({})",
            lhs_frame->name_view(), lhs, rhs_frame->name_view(), rhs);
        // Safety check, but this should never happen.
        return unexpected_t(Error::kLogicError);
      }
    }
    // Check if we are creating a new connection, if so we will need to update the connected
    // component and we should detect loop.
    auto history = histories_map_.try_get(lhs_rhs.value());
    if (!history) {
      HOLOSCAN_LOG_DEBUG("Unable to set pose: Unable to get history for '{}' ({}) -> '{}' ({})",
                         lhs_frame->name_view(), lhs, rhs_frame->name_view(), rhs);
      // Safety check, but this should never happen.
      return unexpected_t(Error::kLogicError);
    }
    const bool need_update = !history.value()->connected();
    if (need_update) {
      // The edge is disconnected at the moment, so we need to make sure there is no other
      // connection.
      if (lhs_frame->root == rhs_frame->root ||
          get_impl(lhs, rhs, time, PoseTreeEdgeHistory::AccessMethod::kNearest, version_)) {
        HOLOSCAN_LOG_DEBUG(
            "Unable to set pose: cycling dependency detected for '{}' ({}) -> '{}' ({})",
            lhs_frame->name_view(), lhs, rhs_frame->name_view(), rhs);
        return unexpected_t(Error::kCyclingDependency);
      }
    }
    // Store the pose in both histories
    const auto result = history.value()->set(time, lhs_T_rhs, version_ + 1);
    if (!result) {
      HOLOSCAN_LOG_DEBUG("Unable to set pose: out of order for '{}' ({}) -> '{}' ({})",
                         lhs_frame->name_view(), lhs, rhs_frame->name_view(), rhs);
      return unexpected_t(Error::kPoseOutOfOrder);
    }
    // Connect the components if needed
    if (need_update) {
      update_root(lhs);
    }
    version = ++version_;
  }

  {
    // Call each registered SetEdgeCallback function
    std::shared_lock<std::shared_timed_mutex> callbacks_lock(set_edge_callbacks_mutex_);
    for (const uid_t cid : set_edge_callbacks_keys_) {
      auto cb = set_edge_callbacks_.try_get(cid);
      if (cb) {
        (*cb.value())(lhs, rhs, time, lhs_T_rhs);
      }
    }
  }

  return version;
}

// Internal implementation that assumes the lock is already held
PoseTree::expected_t<std::pair<Pose3d, double>> PoseTree::get_latest_impl(frame_t lhs,
                                                                          frame_t rhs) const {
  if (lhs > rhs) {
    auto result = get_latest_impl(rhs, lhs);
    if (!result) {
      return result;
    }
    return std::make_pair(result.value().first.inverse(), result.value().second);
  }
  // Make sure both the frame and edge exists.
  const auto lhs_it = frame_map_.try_get(lhs);
  const auto rhs_it = frame_map_.try_get(rhs);
  if (!lhs_it || !rhs_it) {
    HOLOSCAN_LOG_WARN("Pose frame UID {} or UID {} not found", lhs, rhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  const auto lhs_rhs = edges_map_.get({lhs, rhs});
  // Check the edge exist
  if (!lhs_rhs) {
    return unexpected_t(Error::kFramesNotLinked);
  }
  const auto history = histories_map_.try_get(lhs_rhs.value());
  if (!history) {
    // Safety check, but this should never happen.
    return unexpected_t(Error::kLogicError);
  }
  const auto timed_pose = history.value()->latest();
  if (!timed_pose) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (!timed_pose.value().valid) {
    return unexpected_t(Error::kFramesNotLinked);
  }
  return std::make_pair(timed_pose.value().pose, timed_pose.value().time);
}

PoseTree::expected_t<std::pair<Pose3d, double>> PoseTree::get_latest(std::string_view lhs,
                                                                     std::string_view rhs) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  const auto lhs_frame = find_frame_impl(lhs);
  if (!lhs_frame) {
    HOLOSCAN_LOG_WARN("Pose frame {} not found", lhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  const auto rhs_frame = find_frame_impl(rhs);
  if (!rhs_frame) {
    HOLOSCAN_LOG_WARN("Pose frame {} not found", rhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  return get_latest_impl(lhs_frame.value(), rhs_frame.value());
}

PoseTree::expected_t<std::pair<Pose3d, double>> PoseTree::get_latest(frame_t lhs,
                                                                     frame_t rhs) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_latest_impl(lhs, rhs);
}

PoseTree::expected_t<Pose3d> PoseTree::get(std::string_view lhs, std::string_view rhs,
                                           const double time,
                                           const PoseTreeEdgeHistory::AccessMethod method,
                                           const version_t version) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  const auto lhs_frame = find_frame_impl(lhs);
  if (!lhs_frame) {
    HOLOSCAN_LOG_WARN("Pose frame {} not found", lhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  const auto rhs_frame = find_frame_impl(rhs);
  if (!rhs_frame) {
    HOLOSCAN_LOG_WARN("Pose frame {} not found", rhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  return get_impl(lhs_frame.value(), rhs_frame.value(), time, method, version);
}

PoseTree::expected_t<Pose3d> PoseTree::get(std::string_view lhs, std::string_view rhs,
                                           const double time, const version_t version) const {
  return get(lhs, rhs, time, PoseTreeEdgeHistory::AccessMethod::kDefault, version);
}

PoseTree::expected_t<Pose3d> PoseTree::get(std::string_view lhs, std::string_view rhs,
                                           const double time,
                                           const PoseTreeEdgeHistory::AccessMethod method) const {
  return get(lhs, rhs, time, method, get_pose_tree_version());
}

PoseTree::expected_t<Pose3d> PoseTree::get(std::string_view lhs, std::string_view rhs,
                                           const double time) const {
  return get(lhs, rhs, time, PoseTreeEdgeHistory::AccessMethod::kDefault, get_pose_tree_version());
}

PoseTree::expected_t<Pose3d> PoseTree::get(const std::string_view lhs,
                                           const std::string_view rhs) const {
  return get(lhs, rhs, std::numeric_limits<double>::max(),
             PoseTreeEdgeHistory::AccessMethod::kPrevious, get_pose_tree_version());
}

PoseTree::expected_t<Pose3d> PoseTree::get(const std::string_view lhs, const std::string_view rhs,
                                           const version_t version) const {
  return get(lhs, rhs, std::numeric_limits<double>::max(),
             PoseTreeEdgeHistory::AccessMethod::kPrevious, version);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const frame_t lhs, const frame_t rhs, const double time,
                                           const PoseTreeEdgeHistory::AccessMethod method,
                                           const version_t version) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_impl(lhs, rhs, time, method, version);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const frame_t lhs, const frame_t rhs, const double time,
                                           const version_t version) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_impl(lhs, rhs, time, PoseTreeEdgeHistory::AccessMethod::kDefault, version);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const frame_t lhs, const frame_t rhs, const double time,
                                           const PoseTreeEdgeHistory::AccessMethod method) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_impl(lhs, rhs, time, method, version_);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const frame_t lhs, const frame_t rhs,
                                           const double time) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_impl(lhs, rhs, time, PoseTreeEdgeHistory::AccessMethod::kDefault, version_);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const frame_t lhs, const frame_t rhs) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_impl(lhs, rhs, std::numeric_limits<double>::max(),
                  PoseTreeEdgeHistory::AccessMethod::kPrevious, version_);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const frame_t lhs, const frame_t rhs,
                                           const version_t version) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_impl(lhs, rhs, std::numeric_limits<double>::max(),
                  PoseTreeEdgeHistory::AccessMethod::kPrevious, version);
}

PoseTree::expected_t<Pose3d> PoseTree::get_via_root(const FrameInfo* lhs_frame,
                                                    const FrameInfo* rhs_frame, const double time,
                                                    const PoseTreeEdgeHistory::AccessMethod method,
                                                    const version_t version) const {
  Pose3d lhs_T_parent = Pose3d::identity();
  Pose3d rhs_T_parent = Pose3d::identity();

  const auto update_pose = [&](const FrameInfo** frame, Pose3d& pose) {
    const auto next = frame_map_.try_get((*frame)->node_to_root);
    if (!next) {
      return false;
    }
    const auto current_T_next = edge_pose((*frame)->uid, next.value()->uid, time, method, version);
    if (!current_T_next) {
      return false;
    }
    pose = pose * current_T_next.value();
    *frame = next.value();
    return true;
  };

  // Move lhs to the same distance as rhs
  while (lhs_frame->distance_to_root > rhs_frame->distance_to_root) {
    if (!update_pose(&lhs_frame, lhs_T_parent)) {
      return unexpected_t(Error::kFramesNotLinked);
    }
  }
  // Move rhs to the same distance as lhs
  while (lhs_frame->distance_to_root < rhs_frame->distance_to_root) {
    if (!update_pose(&rhs_frame, rhs_T_parent)) {
      return unexpected_t(Error::kFramesNotLinked);
    }
  }
  // Move in parallel until lhs == rhs
  while (lhs_frame->uid != rhs_frame->uid) {
    if (!update_pose(&lhs_frame, lhs_T_parent) || !update_pose(&rhs_frame, rhs_T_parent)) {
      return unexpected_t(Error::kFramesNotLinked);
    }
  }

  return lhs_T_parent * rhs_T_parent.inverse();
}

PoseTree::expected_t<Pose3d> PoseTree::get_impl(const frame_t lhs, const frame_t rhs,
                                                const double time,
                                                const PoseTreeEdgeHistory::AccessMethod method,
                                                const version_t version) const {
  if (lhs == rhs) {
    return Pose3d::identity();
  }
  if (lhs > rhs) {
    const auto rhs_T_lhs = get_impl(rhs, lhs, time, method, version);
    if (!rhs_T_lhs) {
      return rhs_T_lhs;
    }
    return rhs_T_lhs.value().inverse();
  }
  // Make sure both the frame exist.
  const auto lhs_it = frame_map_.try_get(lhs);
  const auto rhs_it = frame_map_.try_get(rhs);
  if (!lhs_it || !rhs_it) {
    HOLOSCAN_LOG_WARN("Pose frame UID {} or UID {} not found", lhs, rhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  // The pre-computed paths to the root are only a usable hint while both frames hang off the same
  // root; otherwise the two walks have nowhere to meet. A stale hint fails the same way, and both
  // fall back to searching the tree outright.
  if (lhs_it.value()->root == rhs_it.value()->root) {
    const auto pose = get_via_root(lhs_it.value(), rhs_it.value(), time, method, version);
    if (pose) {
      return pose.value();
    }
  }
  return get_dfs_impl(lhs, rhs, time, method, version);
}

PoseTree::expected_t<PoseTree::uid_t> PoseTree::add_create_frame_callback(
    const CreateFrameCallback& callback) {
  std::unique_lock<std::shared_timed_mutex> lock(create_frame_callbacks_mutex_);
  const auto cid = create_frame_callbacks_.insert(callback);
  if (!cid) {
    return unexpected_t(Error::kOutOfMemory);
  }
  if (!push_key(create_frame_callbacks_keys_, cid.value())) {
    // Without its key the callback would never be invoked, and never be removable either: the
    // remove path erases from the map and then looks for the key to confirm, so it would report
    // kInvalidArgument having already erased.
    create_frame_callbacks_.erase(cid.value());
    return unexpected_t(Error::kOutOfMemory);
  }
  return cid.value();
}

PoseTree::expected_t<void> PoseTree::remove_create_frame_callback(uid_t cid) {
  std::unique_lock<std::shared_timed_mutex> lock(create_frame_callbacks_mutex_);
  // If the cid does not have an associated callback function, return an error
  auto erase = create_frame_callbacks_.erase(cid);
  if (!erase) {
    return unexpected_t(Error::kInvalidArgument);
  }
  for (size_t i = 0; i < create_frame_callbacks_keys_.size(); i++) {
    if (create_frame_callbacks_keys_[i] == cid) {
      create_frame_callbacks_keys_[i] = create_frame_callbacks_keys_.back();
      create_frame_callbacks_keys_.pop_back();
      return expected_t<void>{};
    }
  }
  return unexpected_t(Error::kInvalidArgument);
}

PoseTree::expected_t<PoseTree::uid_t> PoseTree::add_set_edge_callback(
    const SetEdgeCallback& callback) {
  std::unique_lock<std::shared_timed_mutex> lock(set_edge_callbacks_mutex_);
  const auto cid = set_edge_callbacks_.insert(callback);
  if (!cid) {
    return unexpected_t(Error::kOutOfMemory);
  }
  if (!push_key(set_edge_callbacks_keys_, cid.value())) {
    // Without its key the callback would never be invoked, and never be removable either: the
    // remove path erases from the map and then looks for the key to confirm, so it would report
    // kInvalidArgument having already erased.
    set_edge_callbacks_.erase(cid.value());
    return unexpected_t(Error::kOutOfMemory);
  }
  return cid.value();
}

PoseTree::expected_t<void> PoseTree::remove_set_edge_callback(uid_t cid) {
  std::unique_lock<std::shared_timed_mutex> lock(set_edge_callbacks_mutex_);
  // If the cid does not have an associated callback function, return an error
  auto erase = set_edge_callbacks_.erase(cid);
  if (!erase) {
    return unexpected_t(Error::kInvalidArgument);
  }
  for (size_t i = 0; i < set_edge_callbacks_keys_.size(); i++) {
    if (set_edge_callbacks_keys_[i] == cid) {
      set_edge_callbacks_keys_[i] = set_edge_callbacks_keys_.back();
      set_edge_callbacks_keys_.pop_back();
      return expected_t<void>{};
    }
  }
  return unexpected_t(Error::kInvalidArgument);
}

PoseTree::expected_t<Pose3d> PoseTree::edge_pose(const frame_t lhs, const frame_t rhs,
                                                 const double time,
                                                 const PoseTreeEdgeHistory::AccessMethod method,
                                                 const version_t version) const {
  const bool forward = lhs < rhs;
  const auto history_it = edges_map_.get(forward ? std::pair{lhs, rhs} : std::pair{rhs, lhs});
  if (!history_it) {
    return unexpected_t(Error::kLogicError);
  }
  const auto history = histories_map_.try_get(history_it.value());
  if (!history) {
    return unexpected_t(Error::kLogicError);
  }
  const auto pose = history.value()->get(time, method, version);
  if (!pose) {
    return unexpected_t(Error::kFramesNotLinked);
  }
  return forward ? pose.value() : pose.value().inverse();
}

PoseTree::expected_t<Pose3d> PoseTree::dfs_path_pose(const FrameInfo* frame, const frame_t rhs,
                                                     const double time,
                                                     const PoseTreeEdgeHistory::AccessMethod method,
                                                     const version_t version) const {
  Pose3d pose = Pose3d::identity();
  while (frame->uid != rhs) {
    const auto current_T_next = edge_pose(frame->uid, frame->dfs_link, time, method, version);
    if (!current_T_next) {
      return unexpected_t(current_T_next.error());
    }
    pose = pose * current_T_next.value();
    frame = frame_map_.try_get(frame->dfs_link).value();
  }
  return pose;
}

int32_t PoseTree::push_unvisited_neighbors(const FrameInfo* frame, const double time,
                                           const PoseTreeEdgeHistory::AccessMethod method,
                                           const version_t version, frame_t* stack,
                                           int32_t size) const {
  // Go through all the edges
  for (int edge = 0; edge < frame->number_edges; edge++) {
    const auto history = histories_map_.try_get(frame->history[edge]);
    if (!history) {
      continue;
    }
    // Ignore not connected edges
    if (!history.value()->get(time, method, version)) {
      continue;
    }
    const frame_t other =
        history.value()->rhs() == frame->uid ? history.value()->lhs() : history.value()->rhs();
    const auto it = frame_map_.try_get(other);
    if (!it) {
      continue;
    }
    // Make sure we haven't visited this edge before (it means it is where we come from).
    if (it.value()->hint_version == hint_version_) {
      continue;
    }
    // Update the information
    it.value()->hint_version = hint_version_;
    it.value()->dfs_link = frame->uid;
    // Add to the stack.
    stack[size++] = it.value()->uid;
  }
  return size;
}

PoseTree::expected_t<Pose3d> PoseTree::get_dfs_impl(const frame_t lhs, const frame_t rhs,
                                                    const double time,
                                                    const PoseTreeEdgeHistory::AccessMethod method,
                                                    const version_t version) const {
  std::unique_lock<std::mutex> lock(dfs_mutex_);
  int32_t size = 0;
  frame_t* stack = frames_stack_.get();
  const auto start = frame_map_.try_get(rhs);
  if (!start) {
    return unexpected_t(Error::kLogicError);
  }
  start.value()->hint_version = hint_version_;
  stack[size++] = rhs;
  // DFS implementation
  while (size > 0) {
    const FrameInfo* frame = frame_map_.try_get(stack[--size]).value();
    if (frame->uid == lhs) {
      // The search is over whether or not the path reconstructs, so the visited marks are retired
      // either way. Leaving them behind would make the next search treat those frames as seen.
      const auto pose = dfs_path_pose(frame, rhs, time, method, version);
      hint_version_++;
      return pose;
    }
    size = push_unvisited_neighbors(frame, time, method, version, stack, size);
  }
  hint_version_++;
  return unexpected_t(Error::kFramesNotLinked);
}

}  // namespace isaac::os::pose_tree

// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
