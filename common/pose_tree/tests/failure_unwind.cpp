/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A call that fails must leave the tree exactly as it found it.
//
// These reach a failure by exhausting a capacity rather than by injecting a fault, and check what
// the tree looks like afterwards rather than that the call failed.
//
// What they cannot reach: the failures inside create_edges_impl and create_frame_impl that occur
// after the first mutation, where the unwinding is. init() ties histories_map_ and
// histories_management_ to the same number_edges, and every frame reserves default_number_edges
// slots from the second when it is created, so the edges a tree can hold are bounded by half its
// map capacity and the maps never fill. The per-frame slot check refuses the edge first, before
// anything has been written. Those unwind paths are therefore defensive, and reachable only if
// init's coupling changes or an allocator fails for another reason.

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "common/pose_tree/pose_tree.hpp"

namespace isaac::os::pose_tree {
namespace {

using Edge = std::pair<PoseTree::frame_t, PoseTree::frame_t>;

/// A tree with room for `frames` frames, each able to hold `slots` edges.
///
/// The edge pool is generous and the per-frame slot count is what binds, so an edge is refused by
/// the slot check rather than by an exhausted allocator.
std::vector<PoseTree::frame_t> build(PoseTree& tree, int frames, int slots) {
  EXPECT_TRUE(tree.init(frames, frames * slots * 4, 65536, slots, 64, 1, 8));
  std::vector<PoseTree::frame_t> ids;
  for (int i = 0; i < frames; i++) {
    auto uid = tree.create_frame("f" + std::to_string(i));
    EXPECT_TRUE(uid);
    if (uid) ids.push_back(*uid);
  }
  return ids;
}

/// Relate two frames and write one pose, which is what makes the edge connected. get_edge_uids
/// lists connected edges only, so an edge that was created and never written to is invisible to it.
bool link(PoseTree& tree, PoseTree::frame_t a, PoseTree::frame_t b) {
  if (!tree.create_edges(a, b)) return false;
  return tree.set(a, b, 1.0, Pose3d::identity()).has_value();
}

/// Edges the tree reports as existing.
std::vector<Edge> edges_of(const PoseTree& tree) {
  std::vector<Edge> listed;
  listed.reserve(1024);
  EXPECT_TRUE(tree.get_edge_uids(listed));
  return listed;
}

TEST(FailureUnwind, RefusedEdgeLeavesTheEdgeSetUnchanged) {
  PoseTree tree;
  const auto f = build(tree, 8, 2);
  ASSERT_EQ(f.size(), 8U);

  int made = 0;
  for (size_t i = 0; i < f.size(); i++)
    for (size_t j = i + 1; j < f.size(); j++)
      if (link(tree, f[i], f[j])) made++;
  ASSERT_GT(made, 0);

  // Every edge reported as existing must also be listed: one recorded in the map but missing from
  // the key list would be counted by the first and absent from the second.
  EXPECT_EQ((int)edges_of(tree).size(), made);
}

TEST(FailureUnwind, RefusedEdgeConsumesNothing) {
  PoseTree tree;
  const auto f = build(tree, 8, 2);
  ASSERT_EQ(f.size(), 8U);
  for (size_t i = 0; i < f.size(); i++)
    for (size_t j = i + 1; j < f.size(); j++) link(tree, f[i], f[j]);

  const auto listed = edges_of(tree);
  ASSERT_FALSE(listed.empty());
  // Freeing one edge must make room for exactly that edge again. Had a refused create taken a
  // history or its pose storage on the way out, the room would not come back.
  ASSERT_TRUE(tree.delete_edge(listed[0].first, listed[0].second));
  EXPECT_TRUE(link(tree, listed[0].first, listed[0].second));
  EXPECT_EQ(edges_of(tree).size(), listed.size());
}

TEST(FailureUnwind, DuplicateEdgeIsRejectedWithoutDisturbingTheTree) {
  PoseTree tree;
  const auto f = build(tree, 8, 4);
  ASSERT_TRUE(link(tree, f[0], f[1]));
  const auto before = edges_of(tree);

  auto again = tree.create_edges(f[0], f[1]);
  ASSERT_FALSE(again);
  EXPECT_EQ(again.error(), PoseTree::Error::kAlreadyExists);
  EXPECT_EQ(edges_of(tree).size(), before.size());
  // The edge that was already there still works. A later time than link() used, since the history
  // takes poses in order.
  EXPECT_TRUE(tree.set(f[0], f[1], 2.0, Pose3d::identity()));
}

TEST(FailureUnwind, DeletingAFrameWithEdgesTerminatesAndTakesThemWithIt) {
  PoseTree tree;
  const auto f = build(tree, 8, 4);
  for (size_t i = 1; i < 5; i++) ASSERT_TRUE(link(tree, f[0], f[i]));
  ASSERT_EQ(edges_of(tree).size(), 4U);

  // The frame's edges are removed one at a time. A removal that reported failure without
  // shortening the frame would leave this spinning rather than returning.
  ASSERT_TRUE(tree.delete_frame(f[0]));
  EXPECT_TRUE(edges_of(tree).empty());
  EXPECT_FALSE(tree.find_frame("f0"));
}

TEST(FailureUnwind, DeletedFrameReleasesItsNameAndItsCapacity) {
  PoseTree tree;
  const auto f = build(tree, 4, 4);
  ASSERT_EQ(f.size(), 4U);

  // The tree is full, so a fifth frame is refused without touching anything.
  auto overflow = tree.create_frame("f4");
  ASSERT_FALSE(overflow);
  EXPECT_EQ(overflow.error(), PoseTree::Error::kOutOfMemory);
  EXPECT_TRUE(tree.find_frame("f0"));

  ASSERT_TRUE(tree.delete_frame(f[2]));
  // The name and the frame have to go together: a name left behind would make this kAlreadyExists
  // against a frame that no longer exists, and a frame left behind would make it kOutOfMemory.
  EXPECT_TRUE(tree.create_frame("f2"));
  EXPECT_TRUE(tree.create_frame("f4").has_value() == false);
}

TEST(Deinit, ForgetsTheFramesItHeld) {
  PoseTree tree;
  const auto f = build(tree, 4, 4);
  ASSERT_EQ(f.size(), 4U);
  ASSERT_TRUE(tree.find_frame("f0"));

  tree.deinit();
  // A frame of a tree that no longer exists must not answer to a lookup.
  EXPECT_FALSE(tree.find_frame("f0"));
  EXPECT_FALSE(tree.find_frame("f3"));
}

TEST(Deinit, LeavesTheNextTreeItsFullCapacity) {
  PoseTree tree;
  ASSERT_EQ(build(tree, 4, 4).size(), 4U);  // fills the tree
  tree.deinit();

  // The second tree is the same size as the first, so it must hold as many frames. A count carried
  // over from the first would have it reach capacity that many frames early.
  const auto again = build(tree, 4, 4);
  EXPECT_EQ(again.size(), 4U) << "the new tree ran out of room early";
  EXPECT_TRUE(tree.find_frame("f0"));
}

TEST(Deinit, LeavesTheNextTreeItsEdgesAndCallbacks) {
  PoseTree tree;
  auto f = build(tree, 4, 4);
  ASSERT_TRUE(link(tree, f[0], f[1]));
  auto cb = tree.add_create_frame_callback([](PoseTree::frame_t) {
    // Empty on purpose: the test only cares that the slot was taken and later freed, never that
    // the callback ran. A body here would be invoked by frames other tests create and prove
    // nothing about deinit.
  });
  ASSERT_TRUE(cb);
  tree.deinit();

  f = build(tree, 4, 4);
  ASSERT_EQ(f.size(), 4U);
  EXPECT_TRUE(edges_of(tree).empty()) << "an edge outlived the tree it belonged to";
  EXPECT_TRUE(link(tree, f[0], f[1]));
  // The callback registered against the old tree must not be invoked by the new one, and its slot
  // must be free again.
  EXPECT_TRUE(tree.add_create_frame_callback([](PoseTree::frame_t) {
    // Empty on purpose: registering at all is the assertion, since it can only succeed if deinit
    // released the slot the previous callback held.
  }));
}

// Erasing from the frame map can relocate another entry into the freed slot. Nothing may hold a
// view into a frame's storage across that move.
TEST(Relocation, ErasingAFrameDoesNotRenameTheOneThatMovesIntoItsSlot) {
  PoseTree tree;
  // Four frames means eight buckets, so ids 1 and 9 land on the same one and the second is stored
  // beside the first. Erasing the first moves the second into the slot the first occupied.
  ASSERT_TRUE(tree.init(4, 16, 16, 1, 1, 1, 1));
  ASSERT_TRUE(tree.create_frame_with_id(1, "alpha"));
  ASSERT_TRUE(tree.create_frame_with_id(9, "bravo"));
  ASSERT_TRUE(tree.delete_frame(PoseTree::frame_t{1}));

  const auto name = tree.get_frame_name(9);
  ASSERT_TRUE(name);
  EXPECT_EQ(*name, "bravo") << "the relocated frame answered with the erased frame's name";
  auto found = tree.find_frame("bravo");
  ASSERT_TRUE(found) << "the relocated frame could no longer be found by name";
  EXPECT_EQ(*found, PoseTree::frame_t{9});
  EXPECT_FALSE(tree.find_frame("alpha"));
}

TEST(Relocation, ListingsSurviveTheMoveToo) {
  PoseTree tree;
  ASSERT_TRUE(tree.init(4, 16, 16, 1, 1, 1, 1));
  ASSERT_TRUE(tree.create_frame_with_id(1, "alpha"));
  ASSERT_TRUE(tree.create_frame_with_id(9, "bravo"));
  ASSERT_TRUE(tree.delete_frame(PoseTree::frame_t{1}));

  std::vector<std::string_view> names;
  names.reserve(16);
  ASSERT_TRUE(tree.get_frame_names(names));
  ASSERT_EQ(names.size(), 1U);
  EXPECT_EQ(names[0], "bravo");

  std::vector<PoseTree::frame_t> uids;
  uids.reserve(16);
  ASSERT_TRUE(tree.get_frame_uids(uids));
  ASSERT_EQ(uids.size(), 1U);
  EXPECT_EQ(uids[0], PoseTree::frame_t{9});
}

TEST(Relocation, ANameTooLongToStoreMatchesNothing) {
  PoseTree tree;
  ASSERT_TRUE(tree.init(8, 64, 1024, 4, 64, 1, 8));
  const std::string stored(100, 'x');
  ASSERT_TRUE(tree.create_frame(stored));

  // Shares every character of the stored name and then some. Truncating it to what a frame can
  // hold would make it match; it must not.
  const std::string longer(200, 'x');
  EXPECT_FALSE(tree.find_frame(longer));
  EXPECT_TRUE(tree.find_frame(stored));
}

}  // namespace
}  // namespace isaac::os::pose_tree
