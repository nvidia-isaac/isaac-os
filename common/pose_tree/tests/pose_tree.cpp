/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "common/pose_tree/pose_tree.hpp"

#include <sys/time.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "common/pose_tree/math/interpolation.hpp"
#include "common/pose_tree/math/pose3.hpp"

static inline double get_time() {
  timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define EXPECT_VEC_EQ(A, B)                 \
  {                                         \
    auto _a = (A);                          \
    auto _b = (B);                          \
    ASSERT_EQ(_a.size(), _b.size());        \
    const size_t a_length = _a.size();      \
    for (size_t i = 0; i < a_length; i++) { \
      EXPECT_EQ(_a[i], _b[i]);              \
    }                                       \
  }

#define EXPECT_VEC_NEAR(A, B, T)            \
  {                                         \
    auto _a = (A);                          \
    auto _b = (B);                          \
    ASSERT_EQ(_a.size(), _b.size());        \
    const size_t a_length = _a.size();      \
    for (size_t i = 0; i < a_length; i++) { \
      EXPECT_NEAR(_a[i], _b[i], T);         \
    }                                       \
  }

#define EXPECT_MAT_NEAR(A, B, T)            \
  {                                         \
    auto _a = (A);                          \
    auto _b = (B);                          \
    ASSERT_EQ(_a.rows(), _b.rows());        \
    ASSERT_EQ(_a.cols(), _b.cols());        \
    for (int i = 0; i < _a.rows(); i++) {   \
      for (int j = 0; j < _a.cols(); j++) { \
        EXPECT_NEAR(_a(i, j), _b(i, j), T); \
      }                                     \
    }                                       \
  }

#define ASSERT_VEC_NEAR(A, B, T)            \
  {                                         \
    auto _a = (A);                          \
    auto _b = (B);                          \
    ASSERT_EQ(_a.size(), _b.size());        \
    const size_t a_length = _a.size();      \
    for (size_t i = 0; i < a_length; i++) { \
      ASSERT_NEAR(_a[i], _b[i], T);         \
    }                                       \
  }

#define EXPECT_VEC_NEAR_ZERO(A, T)          \
  {                                         \
    auto _a = (A);                          \
    const size_t a_length = _a.size();      \
    for (size_t i = 0; i < a_length; i++) { \
      EXPECT_NEAR(_a[i], 0, T);             \
    }                                       \
  }

#define EXPECT_SO_NEAR_ID(A, T) EXPECT_NEAR((A).angle(), 0.0, T);

#define EXPECT_POSE_NEAR_ID(A, T)             \
  {                                           \
    EXPECT_SO_NEAR_ID((A).rotation, T);       \
    EXPECT_VEC_NEAR_ZERO((A).translation, T); \
  }

#define EXPECT_SO_NEAR(A, B, T) EXPECT_SO_NEAR_ID((A) * (B).inverse(), T);

#define EXPECT_POSE_NEAR(A, B, T) EXPECT_POSE_NEAR_ID((A) * (B).inverse(), T);

namespace isaac::os::pose_tree {
namespace {
// Poses for the tests that need a stream of them. Written out here rather than drawn from
// <random>: these tests want poses that are varied and awkward to compose, not poses that are
// unpredictable. Nothing derives a key, token, identifier or nonce from them, and a fixed sequence
// is the point -- a failing assertion reproduces exactly on the next run.
//
// Each test holds its own sequence, so what one test asserts against does not depend on how many
// poses the tests before it happened to draw.
class PoseSequence {
 public:
  explicit PoseSequence(uint64_t seed = 0x0123456789abcdefULL) : state_{seed} {}

  // The next pose: a rotation of up to 1.2 radians about an arbitrary axis, and a translation
  // inside a box three units across, which is roughly the spread the tests were written against.
  Pose3d next() {
    const Vector3d axis = next_axis();
    const double angle = 1.2 * next_scalar();
    const double x = 3.0 * next_scalar();
    const double y = 3.0 * next_scalar();
    const double z = 3.0 * next_scalar();
    return Pose3d{SO3d::from_axis_angle(axis, angle), Vector3d(x, y, z)};
  }

 private:
  // The next value in [-1, 1), from a 64-bit linear congruential generator with Knuth's MMIX
  // constants. The low bits of an LCG are its weakest, so the mantissa comes from the top 53.
  double next_scalar() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    const double fraction = static_cast<double>(state_ >> 11) / static_cast<double>(1ULL << 53);
    return 2.0 * fraction - 1.0;
  }

  // An axis to turn about. from_axis_angle normalises whatever it is handed, so this only has to be
  // long enough that normalising it does not amplify rounding. The sequence is fixed, so the
  // fallback either never fires or always does, and the pose is a valid one either way.
  Vector3d next_axis() {
    const Vector3d axis(next_scalar(), next_scalar(), next_scalar());
    return axis.norm() < 0.25 ? Vector3d::UnitZ() : axis;
  }

  uint64_t state_;
};

// For the tests that need no more than a handful of poses, picked by hand so the values an
// assertion rests on are visible in the file rather than a function of where a sequence had got to.
// They turn about different axes and differ in every translation component, which is all those
// tests ask of them.
const Pose3d kPoseA0{SO3d::from_axis_angle(Vector3d(0.0, 0.0, 1.0), 0.35),
                     Vector3d(1.5, -0.25, 0.75)};
const Pose3d kPoseA1{SO3d::from_axis_angle(Vector3d(1.0, 0.0, 0.0), -0.8),
                     Vector3d(-0.5, 2.25, 1.0)};
const Pose3d kPoseB0{SO3d::from_axis_angle(Vector3d(0.0, 1.0, 0.0), 1.1),
                     Vector3d(0.25, 0.5, -1.75)};
const Pose3d kPoseB1{SO3d::from_axis_angle(Vector3d(1.0, 1.0, 1.0), 0.6),
                     Vector3d(-1.25, -0.75, 2.0)};

}  // namespace

TEST(error_to_str, errors) {
  ASSERT_EQ(PoseTree::error_to_str(PoseTree::Error::kInvalidArgument),
            std::string("Invalid Argument"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kOutOfMemory), std::string("Out of Memory"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kFrameNotFound),
            std::string("Frame not found"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kAlreadyExists),
            std::string("Edge already exists"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kCyclingDependency),
            std::string("Cycling dependency"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kFramesNotLinked),
            std::string("Frames are not linked"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kPoseOutOfOrder),
            std::string("Pose updated out of order"));
  EXPECT_EQ(PoseTree::error_to_str(PoseTree::Error::kLogicError), std::string("Logic Error"));
  EXPECT_EQ(PoseTree::error_to_str(static_cast<PoseTree::Error>(-1)),
            std::string("Invalid Error Code"));
}

TEST(PoseTree, init) {
  EXPECT_EQ(PoseTree().init(0, 256, 1024, 32, 32, 4, 4).error(), PoseTree::Error::kInvalidArgument);
  EXPECT_EQ(PoseTree().init(16, 0, 1024, 32, 32, 4, 4).error(), PoseTree::Error::kInvalidArgument);
  EXPECT_EQ(PoseTree().init(16, 256, 0, 32, 32, 4, 4).error(), PoseTree::Error::kInvalidArgument);
  EXPECT_EQ(PoseTree().init(16, 256, 1024, 0, 32, 4, 4).error(), PoseTree::Error::kInvalidArgument);
  EXPECT_EQ(PoseTree().init(16, 256, 1024, 32, 0, 4, 4).error(), PoseTree::Error::kInvalidArgument);
  EXPECT_EQ(PoseTree().init(16, 256, 1024, 32, 32, 0, 4).error(),
            PoseTree::Error::kInvalidArgument);
  EXPECT_EQ(PoseTree().init(16, 256, 1024, 32, 32, 4, 0).error(),
            PoseTree::Error::kInvalidArgument);
  EXPECT_TRUE(PoseTree().init(16, 256, 1024, 32, 32, 4, 4));
}

TEST(PoseTree, Id) {
  for (int i = 0; i < 10; i++) {
    PoseTree pg;
    ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
    const PoseTree::version_t version = pg.get_pose_tree_version();
    const PoseTree::frame_t lhs = pg.create_frame("a").value();
    const PoseTree::frame_t rhs = pg.create_frame("b").value();
    EXPECT_FALSE(pg.create_frame("a"));
    EXPECT_FALSE(pg.create_frame("b"));
    ASSERT_TRUE(pg.find_frame("a"));
    ASSERT_TRUE(pg.find_frame("b"));
    EXPECT_EQ(pg.find_frame("a").value(), lhs);
    EXPECT_EQ(pg.find_frame("b").value(), rhs);
    EXPECT_FALSE(pg.find_frame("aa"));
    EXPECT_TRUE(pg.create_frame("aa"));
    const Pose3d actual = kPoseA0;
    EXPECT_EQ(pg.get_pose_tree_version(), version);
    ASSERT_TRUE(pg.set("a", "b", 0.0, actual));
    EXPECT_GT(pg.get_pose_tree_version(), version);
    auto maybe = pg.get(rhs, rhs, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), Pose3d::identity(), 1e-12);
    maybe = pg.get(lhs, lhs, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), Pose3d::identity(), 1e-12);
  }
}

TEST(PoseTree, create_frames) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(8, 256, 1024, 4, 4, 4, 4));
  ASSERT_TRUE(pg.create_frame("a"));
  ASSERT_EQ(pg.create_frame("a").error(), PoseTree::Error::kAlreadyExists);
  ASSERT_EQ(pg.create_frame("_a").error(), PoseTree::Error::kInvalidArgument);
  const std::string long_name =
      (std::string("abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz01234569") +
       std::string("abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz01234569") +
       std::string("abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz01234569"));
  ASSERT_EQ(pg.create_frame(long_name.c_str()).error(), PoseTree::Error::kInvalidArgument);
  auto focb = pg.find_or_create_frame("b");
  ASSERT_TRUE(focb);
  ASSERT_TRUE(pg.find_or_create_frame("b").value() == focb.value());
  ASSERT_TRUE(pg.find_or_create_frame("b", 8).value() == focb.value());
  ASSERT_TRUE(pg.create_frame("c", 8));
  // Adds unnamed variable, should always work (assuming enough memory)
  ASSERT_TRUE(pg.create_frame(8));
  ASSERT_TRUE(pg.create_frame(8));
  ASSERT_TRUE(pg.create_frame());
  ASSERT_TRUE(pg.create_frame());
  // Check run out of memory
  ASSERT_EQ(pg.create_frame(256).error(), PoseTree::Error::kOutOfMemory);
  ASSERT_TRUE(pg.create_frame(8));

  ASSERT_EQ(pg.create_frame().error(), PoseTree::Error::kOutOfMemory);
}

TEST(PoseTree, set_multithreading_info) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(8, 256, 1024, 4, 4, 4, 4));
  ASSERT_TRUE(pg.set_multithreading_info(42, 15));
  for (int i = 0; i < 8; i++) {
    auto uid = pg.create_frame();
    ASSERT_TRUE(uid);
    EXPECT_EQ(uid.value(), 42 + i * 15);
  }
}

TEST(PoseTree, create_frame_with_id) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(8, 256, 1024, 4, 4, 4, 4));
  ASSERT_EQ(pg.create_frame_with_id(42, "a"), 42);
  ASSERT_EQ(pg.create_frame_with_id(2, "b"), 2);
  ASSERT_EQ(pg.create_frame_with_id(23, "c"), 23);
  ASSERT_EQ(pg.create_frame_with_id(4, "d"), 4);
  ASSERT_EQ(pg.create_frame("e"), 1);
  ASSERT_EQ(pg.create_frame("f"), 3);
  ASSERT_EQ(pg.create_frame("g"), 5);
  ASSERT_FALSE(pg.create_frame_with_id(1, "h"));
  ASSERT_EQ(pg.create_frame("h"), 6);
  ASSERT_EQ(pg.create_frame("z").error(), PoseTree::Error::kOutOfMemory);
}

TEST(PoseTree, create_edges) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(8, 256, 1024, 4, 4, 1, 1));
  const PoseTree::frame_t a = pg.create_frame("a", 1).value();
  const PoseTree::frame_t b = pg.create_frame("b").value();
  const PoseTree::frame_t c = pg.create_frame("c").value();
  const PoseTree::frame_t d = pg.create_frame("d").value();
  const PoseTree::frame_t e = pg.create_frame("e").value();
  ASSERT_TRUE(pg.delete_frame(e));
  ASSERT_TRUE(pg.create_edges(a, b));
  ASSERT_EQ(pg.create_edges(a, b).error(), PoseTree::Error::kAlreadyExists);
  ASSERT_EQ(pg.create_edges(b, a).error(), PoseTree::Error::kAlreadyExists);
  ASSERT_TRUE(pg.create_edges(b, c, 16));
  ASSERT_EQ(pg.create_edges(c, d, 1024).error(), PoseTree::Error::kOutOfMemory);
  ASSERT_EQ(pg.create_edges(d, e).error(), PoseTree::Error::kFrameNotFound);
  ASSERT_EQ(pg.create_edges(d, a).error(), PoseTree::Error::kOutOfMemory);
  ASSERT_TRUE(pg.create_edges(d, c, 16));
}

TEST(PoseTree, TwoNodes) {
  PoseSequence poses;
  for (int i = 0; i < 10; i++) {
    PoseTree pg;
    ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
    const PoseTree::frame_t lhs = pg.create_frame().value();
    const PoseTree::frame_t rhs = pg.create_frame().value();
    const Pose3d actual = poses.next();
    ASSERT_TRUE(pg.set(lhs, rhs, 0.0, actual));
    auto maybe = pg.get(lhs, rhs, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), actual, 1e-12);
    maybe = pg.get(rhs, lhs, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), actual.inverse(), 1e-12);
  }
}

TEST(PoseTree, ThreeNodes) {
  PoseSequence poses;
  for (int i = 0; i < 10; i++) {
    PoseTree pg;
    ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
    const PoseTree::frame_t a = pg.create_frame().value();
    const PoseTree::frame_t b = pg.create_frame().value();
    const PoseTree::frame_t c = pg.create_frame().value();
    const Pose3d aTb = poses.next();
    const Pose3d bTc = poses.next();
    const Pose3d aTc = aTb * bTc;
    ASSERT_TRUE(pg.set(a, b, 0.0, aTb));
    ASSERT_TRUE(pg.set(b, c, 0.0, bTc));
    auto maybe = pg.get(a, c, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), aTc, 1e-12);
    maybe = pg.get(c, a, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), aTc.inverse(), 1e-12);
  }
}

// Hangs `branching` new frames off `parent`, each at a random pose. Returns the first of them,
// which is the one the test's path continues down, together with the pose that reaches it.
std::pair<PoseTree::frame_t, Pose3d> add_tree_level(PoseTree& pg, const PoseTree::frame_t parent,
                                                    const int branching, PoseSequence& poses) {
  PoseTree::frame_t first{};
  Pose3d parent_T_first = Pose3d::identity();
  for (int b = 0; b < branching; b++) {
    const Pose3d pose = poses.next();
    const PoseTree::frame_t child = pg.create_frame().value();
    EXPECT_TRUE(pg.set(parent, child, 0.0, pose));
    if (b == 0) {
      first = child;
      parent_T_first = pose;
    }
  }
  return {first, parent_T_first};
}

TEST(PoseTree, Tree) {
  // Create a pose graph which looks like a tree and go from top to bottom.
  constexpr int kRepetitions = 10;
  constexpr int kDepth = 10;
  constexpr int kBranching = 10;
  PoseSequence poses;
  for (int i = 0; i < kRepetitions; i++) {
    PoseTree pg;
    ASSERT_TRUE(pg.init(1024, 1024 * 16, 1024 * 1024, 16, 16, 4, 4));
    const PoseTree::frame_t root = pg.create_frame().value();
    // Walk one branch from the root to a leaf, composing the poses taken along the way. Every level
    // is fully populated, so the path is one of many, not the only one the tree has to offer.
    PoseTree::frame_t leaf = root;
    Pose3d root_T_leaf = Pose3d::identity();
    for (int d = 1; d < kDepth; d++) {
      const auto [child, parent_T_child] = add_tree_level(pg, leaf, kBranching, poses);
      root_T_leaf = root_T_leaf * parent_T_child;
      leaf = child;
    }
    auto maybe = pg.get(root, leaf, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), root_T_leaf, 1e-12);
    maybe = pg.get(leaf, root, 0.0);
    ASSERT_TRUE(maybe);
    EXPECT_POSE_NEAR(maybe.value(), root_T_leaf.inverse(), 1e-12);
  }
}

TEST(PoseTree, getPose2XY) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame("a").value();
  const PoseTree::frame_t b = pg.create_frame("b").value();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::from_translation(Vector3d(1.0, 2.0, 3.0))));
  EXPECT_POSE_NEAR(pg.get_pose2_xy(a, b, 1.0).value(), Pose2d::from_translation(1.0, 2.0), 1e-12);
  EXPECT_POSE_NEAR(pg.get_pose2_xy("a", "b", 1.0).value(), Pose2d::from_translation(1.0, 2.0),
                   1e-12);
}

TEST(PoseTree, CheckCycle) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::identity()));
  EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::identity()));
  EXPECT_FALSE(pg.set(a, c, 0.0, Pose3d::identity()));
}

TEST(PoseTree, CheckInterpolation) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame("a").value();
  const PoseTree::frame_t b = pg.create_frame("b").value();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0))));
  EXPECT_POSE_NEAR(pg.get(a, b, 0.5).value(), Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)),
                   1e-12);
  EXPECT_POSE_NEAR(pg.get("a", "b", 0.5).value(), Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)),
                   1e-12);
  EXPECT_POSE_NEAR(
      pg.get(a, b, 0.5, PoseTreeEdgeHistory::AccessMethod::kInterpolateLinearly).value(),
      Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get("a", "b", 0.5, PoseTreeEdgeHistory::AccessMethod::kInterpolateLinearly).value(),
      Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get(a, b, 1.5, PoseTreeEdgeHistory::AccessMethod::kInterpolateLinearly).value(),
      Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get("a", "b", 1.5, PoseTreeEdgeHistory::AccessMethod::kInterpolateLinearly).value(),
      Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get(a, b, 0.5, PoseTreeEdgeHistory::AccessMethod::kExtrapolateLinearly).value(),
      Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get("a", "b", 0.5, PoseTreeEdgeHistory::AccessMethod::kExtrapolateLinearly).value(),
      Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get(a, b, 1.5, PoseTreeEdgeHistory::AccessMethod::kExtrapolateLinearly).value(),
      Pose3d::from_translation(Vector3d(4.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(
      pg.get("a", "b", 1.5, PoseTreeEdgeHistory::AccessMethod::kExtrapolateLinearly).value(),
      Pose3d::from_translation(Vector3d(4.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.4, PoseTreeEdgeHistory::AccessMethod::kNearest).value(),
                   Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(pg.get("a", "b", 0.4, PoseTreeEdgeHistory::AccessMethod::kNearest).value(),
                   Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.6, PoseTreeEdgeHistory::AccessMethod::kNearest).value(),
                   Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)), 1e-12);
  EXPECT_POSE_NEAR(pg.get("a", "b", 0.6, PoseTreeEdgeHistory::AccessMethod::kNearest).value(),
                   Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)), 1e-12);
}

TEST(PoseTree, CheckVersioning) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const auto v1 = pg.get_pose_tree_version();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));
  const auto v2 = pg.get_pose_tree_version();
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0))));
  EXPECT_FALSE(pg.get(a, b, 0.0, v1));
  EXPECT_POSE_NEAR(pg.get(a, b, 1.0).value(), Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)),
                   1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 1.0, v2).value(), Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)),
                   1e-12);
}

TEST(PoseTree, CheckOrderedUpdates) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0))));
  EXPECT_FALSE(pg.set(a, b, 1.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0))));
  EXPECT_POSE_NEAR(pg.get(a, b, 1.0).value(), Pose3d::from_translation(Vector3d(3.0, 0.0, 0.0)),
                   1e-12);
  EXPECT_POSE_NEAR(pg.get(b, a, 1.0).value(), Pose3d::from_translation(Vector3d(-3.0, 0.0, 0.0)),
                   1e-12);
  EXPECT_TRUE(pg.set(a, b, 2.0, Pose3d::from_translation(Vector3d(4.0, 0.0, 0.0))));
  EXPECT_FALSE(pg.set(a, b, 1.5, Pose3d::identity()));
}

TEST(PoseTree, TemporalDirect) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const Pose3d aTb0 = Pose3d::from_translation({0.7, -1.3, 2.0});
  const Pose3d aTb1 = Pose3d::from_translation({-0.7, -1.6, 2.4});
  EXPECT_TRUE(pg.set(a, b, 0.0, aTb0));
  EXPECT_TRUE(pg.set(a, b, 1.0, aTb1));
  EXPECT_POSE_NEAR(pg.get(a, b, 0.0).value(), aTb0, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 1.0).value(), aTb1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 12.6).value(), aTb1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.5).value(), Pose3d::from_translation({0.0, -1.45, 2.2}), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.3).value(), interpolate(0.3, aTb0, aTb1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.6).value(), interpolate(0.6, aTb0, aTb1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.9).value(), interpolate(0.9, aTb0, aTb1), 1e-12);
}

TEST(PoseTree, TemporalIndirect) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  const Pose3d aTb0 = kPoseA0;
  const Pose3d aTb1 = kPoseA1;
  const Pose3d bTc0 = kPoseB0;
  const Pose3d bTc1 = kPoseB1;
  EXPECT_TRUE(pg.set(a, b, 0.0, aTb0));
  EXPECT_TRUE(pg.set(a, b, 1.0, aTb1));
  EXPECT_TRUE(pg.set(b, c, 0.0, bTc0));
  EXPECT_TRUE(pg.set(b, c, 1.0, bTc1));
  EXPECT_POSE_NEAR(pg.get(a, b, 0.0).value(), aTb0, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 1.0).value(), aTb1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 12.6).value(), aTb1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 0.0).value(), bTc0, 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 1.0).value(), bTc1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 12.6).value(), bTc1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.3).value(), interpolate(0.3, aTb0, aTb1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.5).value(), interpolate(0.5, aTb0, aTb1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, b, 0.9).value(), interpolate(0.9, aTb0, aTb1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 0.3).value(), interpolate(0.3, bTc0, bTc1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 0.5).value(), interpolate(0.5, bTc0, bTc1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 0.9).value(), interpolate(0.9, bTc0, bTc1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, c, 0.3).value(),
                   interpolate(0.3, aTb0, aTb1) * interpolate(0.3, bTc0, bTc1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, c, 0.5).value(),
                   interpolate(0.5, aTb0, aTb1) * interpolate(0.5, bTc0, bTc1), 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, c, 0.9).value(),
                   interpolate(0.9, aTb0, aTb1) * interpolate(0.9, bTc0, bTc1), 1e-12);
}

TEST(PoseTree, get_latest) {
  const char* a = "a";
  const char* b = "b";
  const char* c = "c";
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  ASSERT_TRUE(pg.create_frame(a));
  ASSERT_TRUE(pg.create_frame(b));
  ASSERT_TRUE(pg.create_frame(c));
  const Pose3d aTb0 = kPoseA0;
  const Pose3d aTb1 = kPoseA1;
  const Pose3d bTc0 = kPoseB0;
  const Pose3d bTc1 = kPoseB1;
  ASSERT_TRUE(pg.set(a, b, 0.0, aTb0));
  ASSERT_TRUE(pg.set(a, b, 2.0, aTb1));
  ASSERT_TRUE(pg.set(b, c, 0.0, bTc0));
  ASSERT_TRUE(pg.set(b, c, 2.5, bTc1));

  EXPECT_NEAR(pg.get_latest(a, b)->second, 2.0, 1e-12);
  EXPECT_POSE_NEAR(pg.get_latest(a, b)->first, aTb1, 1e-12);
  EXPECT_POSE_NEAR(pg.get_latest(b, a)->first, aTb1.inverse(), 1e-12);

  EXPECT_NEAR(pg.get_latest(b, c)->second, 2.5, 1e-12);
  EXPECT_POSE_NEAR(pg.get_latest(b, c)->first, bTc1, 1e-12);
  EXPECT_POSE_NEAR(pg.get_latest(c, b)->first, bTc1.inverse(), 1e-12);

  EXPECT_FALSE(pg.get_latest(a, c));
  EXPECT_FALSE(pg.get_latest(a, a));
  EXPECT_FALSE(pg.get_latest(a, "e"));
  EXPECT_FALSE(pg.get_latest("e", b));
  PoseSequence poses;
  for (int i = 10; i <= 2048; i++) {
    ASSERT_TRUE(pg.set(a, b, static_cast<double>(i), poses.next()));
    ASSERT_NEAR(pg.get_latest(a, b)->second, static_cast<double>(i), 1e-12);
  }
}

TEST(PoseTree, get_latest_edge) {
  const char* a = "a";
  const char* b = "b";
  const char* c = "c";
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  ASSERT_TRUE(pg.create_frame(a));
  ASSERT_TRUE(pg.create_frame(b));
  ASSERT_TRUE(pg.create_frame(c));
  const Pose3d aTb0 = kPoseA0;
  const Pose3d aTb1 = kPoseA1;
  const Pose3d bTc0 = kPoseB0;
  const Pose3d bTc1 = kPoseB1;
  ASSERT_TRUE(pg.set(a, b, 0.0, aTb0));
  ASSERT_TRUE(pg.set(a, b, 2.5, aTb1));
  ASSERT_TRUE(pg.set(b, c, 0.0, bTc0));
  auto version = pg.get_pose_tree_version();
  ASSERT_TRUE(pg.set(b, c, 2.0, bTc1));

  EXPECT_POSE_NEAR(pg.get(a, b).value(), aTb1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(b, a).value(), aTb1.inverse(), 1e-12);

  EXPECT_POSE_NEAR(pg.get(b, c).value(), bTc1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(c, b).value(), bTc1.inverse(), 1e-12);

  EXPECT_POSE_NEAR(pg.get(a, c).value(), aTb1 * bTc1, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, c, version).value(), aTb1 * bTc0, 1e-12);
  EXPECT_POSE_NEAR(pg.get(a, a).value(), Pose3d::identity(), 1e-12);
  EXPECT_FALSE(pg.get(a, "e"));
  EXPECT_FALSE(pg.get("e", b));
}

TEST(PoseTree, disconnect_edge) {
  {
    PoseTree pg;
    ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
    const PoseTree::frame_t a = pg.create_frame().value();
    const PoseTree::frame_t b = pg.create_frame().value();
    const PoseTree::frame_t c = pg.create_frame().value();
    EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::identity()));
    EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::identity()));
    EXPECT_FALSE(pg.set(a, c, 0.0, Pose3d::identity()));
    EXPECT_FALSE(pg.disconnect_edge(a, b, 0.0));
    EXPECT_TRUE(pg.disconnect_edge(a, b, 1.0));
    EXPECT_FALSE(pg.disconnect_edge(a, b, 1.0));
    EXPECT_TRUE(pg.get(a, c, 0.0));
    EXPECT_TRUE(pg.get(c, a, 0.0));
    EXPECT_FALSE(pg.get(a, c, 1.0));
    EXPECT_FALSE(pg.get(c, a, 1.0));
    EXPECT_FALSE(pg.set(a, c, 0.99, Pose3d::identity()));
    EXPECT_TRUE(pg.set(a, c, 1.0, Pose3d::identity()));
  }
  {
    PoseTree pg;
    ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
    const PoseTree::frame_t a = pg.create_frame().value();
    const PoseTree::frame_t b = pg.create_frame().value();
    const PoseTree::frame_t c = pg.create_frame().value();
    EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::identity()));
    EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::identity()));
    EXPECT_FALSE(pg.disconnect_edge(a, c, 1.0));
    EXPECT_FALSE(pg.set(a, c, 0.0, Pose3d::identity()));
    EXPECT_TRUE(pg.disconnect_edge(c, b, 1.0));
    EXPECT_TRUE(pg.get(a, c, 0.0));
    EXPECT_TRUE(pg.get(c, a, 0.0));
    EXPECT_FALSE(pg.get(a, c, 1.0));
    EXPECT_FALSE(pg.get(c, a, 1.0));
    EXPECT_FALSE(pg.set(a, c, 0.99, Pose3d::identity()));
    EXPECT_TRUE(pg.set(a, c, 1.0, Pose3d::identity()));
  }
}

TEST(PoseTree, disconnect_frame) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::identity()));
  EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::identity()));
  EXPECT_FALSE(pg.set(a, c, 0.0, Pose3d::identity()));
  EXPECT_FALSE(pg.disconnect_frame(b, 0.0));
  EXPECT_TRUE(pg.disconnect_frame(b, 1.0));
  EXPECT_TRUE(pg.get(a, c, 0.0));
  EXPECT_FALSE(pg.get(a, c, 1.0));
  EXPECT_TRUE(pg.get(c, a, 0.0));
  EXPECT_FALSE(pg.get(c, a, 1.0));

  EXPECT_TRUE(pg.get(a, b, 0.0));
  EXPECT_FALSE(pg.get(a, b, 1.0));
  EXPECT_TRUE(pg.get(b, a, 0.0));
  EXPECT_FALSE(pg.get(b, a, 1.0));
  EXPECT_TRUE(pg.set(a, c, 1.01, Pose3d::identity()));
  EXPECT_TRUE(pg.set(a, b, 1.01, Pose3d::identity()));
}

TEST(PoseTree, delete_frame) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame("b").value();
  const PoseTree::frame_t c = pg.create_frame().value();
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::identity()));
  EXPECT_TRUE(pg.set(b, c, 1.0, Pose3d::identity()));
  EXPECT_FALSE(pg.set(a, c, 1.0, Pose3d::identity()));
  EXPECT_TRUE(pg.get(a, c, 1.0));
  EXPECT_TRUE(pg.delete_frame(b));
  EXPECT_FALSE(pg.find_frame("b"));
  EXPECT_EQ(pg.set(a, b, 1.0, Pose3d::identity()).error(), PoseTree::Error::kFrameNotFound);
  EXPECT_FALSE(pg.get(a, c, 1.0));
  EXPECT_TRUE(pg.set(a, c, 1.0, Pose3d::identity()));
}

TEST(PoseTree, delete_edge) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::identity()));
  EXPECT_TRUE(pg.set(b, c, 1.0, Pose3d::identity()));
  // For now a and c are connected and would create a loop, after we delete a-b
  // a-c are not connected and we can connect them
  EXPECT_FALSE(pg.set(a, c, 1.0, Pose3d::identity()));
  EXPECT_TRUE(pg.get(a, c, 1.0));
  EXPECT_TRUE(pg.delete_edge(a, b));
  EXPECT_TRUE(pg.get(b, c, 1.0));
  EXPECT_FALSE(pg.get(a, b, 1.0));
  EXPECT_FALSE(pg.get(a, c, 1.0));
  EXPECT_TRUE(pg.set(a, c, 1.0, Pose3d::identity()));
  // Deleting an edge reset the history, we can update in the past.
  EXPECT_FALSE(pg.set(b, c, 0.0, Pose3d::identity()));
  EXPECT_TRUE(pg.delete_edge(c, b));
  EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::identity()));
}

TEST(PoseTree, disconnect_edge_and_queries) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  const PoseTree::frame_t d = pg.create_frame().value();
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::from_translation(Vector3d(0.0, 1.0, 0.0))));
  EXPECT_TRUE(pg.set(c, d, 0.0, Pose3d::from_translation(Vector3d(0.0, 0.0, 1.0))));
  EXPECT_POSE_NEAR(pg.get(a, d, 1.0).value(), Pose3d::from_translation(Vector3d(1.0, 1.0, 1.0)),
                   1e-12);
  EXPECT_TRUE(pg.disconnect_edge(b, c, 1.0));
  EXPECT_FALSE(pg.get(a, d, 1.0));
  EXPECT_TRUE(pg.set(b, c, 2.0, Pose3d::from_translation(Vector3d(0.0, 2.0, 0.0))));
  EXPECT_TRUE(pg.disconnect_edge(b, c, 3.0));
  EXPECT_TRUE(pg.set(b, c, 4.0, Pose3d::from_translation(Vector3d(0.0, 4.0, 0.0))));
  EXPECT_TRUE(pg.set(b, c, 5.0, Pose3d::from_translation(Vector3d(0.0, 5.0, 0.0))));
  EXPECT_TRUE(pg.disconnect_edge(b, c, 6.0));
  EXPECT_TRUE(pg.set(b, c, 7.0, Pose3d::from_translation(Vector3d(0.0, 7.0, 0.0))));
  EXPECT_TRUE(pg.disconnect_edge(b, c, 8.0));
  EXPECT_TRUE(pg.set(b, c, 8.01, Pose3d::from_translation(Vector3d(0.0, 8.0, 0.0))));

  EXPECT_POSE_NEAR(pg.get(a, d, 2.5).value(), Pose3d::from_translation(Vector3d(1.0, 2.0, 1.0)),
                   1e-12);
  EXPECT_FALSE(pg.get(a, d, 3.5));
  EXPECT_POSE_NEAR(pg.get(a, d, 4.5).value(), Pose3d::from_translation(Vector3d(1.0, 4.5, 1.0)),
                   1e-12);
  EXPECT_POSE_NEAR(pg.get(a, d, 5.5).value(), Pose3d::from_translation(Vector3d(1.0, 5.0, 1.0)),
                   1e-12);
  EXPECT_FALSE(pg.get(a, d, 6.5));
  EXPECT_POSE_NEAR(pg.get(a, d, 7.99).value(), Pose3d::from_translation(Vector3d(1.0, 7.0, 1.0)),
                   1e-12);
  EXPECT_POSE_NEAR(pg.get(a, d, 8.01).value(), Pose3d::from_translation(Vector3d(1.0, 8.0, 1.0)),
                   1e-12);
}

TEST(PoseTree, AutoGeneratedName) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const auto maybe_frame = pg.create_frame();
  ASSERT_TRUE(maybe_frame);
  const auto maybe_name = pg.get_frame_name(maybe_frame.value());
  ASSERT_TRUE(maybe_name);
  const std::string expected_name = "_frame_" + std::to_string(maybe_frame.value());
  EXPECT_EQ(maybe_name.value(), expected_name);

  // Disallowed name
  EXPECT_FALSE(pg.find_or_create_frame("_lorem"));
}

TEST(PoseTree, DefaultAccess) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  ASSERT_TRUE(pg.create_edges(a, b, 16, PoseTreeEdgeHistory::AccessMethod::kNearest));
  ASSERT_TRUE(pg.create_edges(b, c, PoseTreeEdgeHistory::AccessMethod::kPrevious));
  EXPECT_TRUE(pg.set(a, b, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(b, c, 1.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0))));
  EXPECT_POSE_NEAR(pg.get(a, b, 0.6).value(), Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0)),
                   1e-12);
  EXPECT_POSE_NEAR(pg.get(b, c, 0.6).value(), Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0)),
                   1e-12);
}

TEST(PoseTree, get_content) {
  PoseTree pg;
  ASSERT_TRUE(pg.init(16, 256, 1024, 32, 32, 4, 4));
  const PoseTree::frame_t a = pg.create_frame().value();
  const PoseTree::frame_t b = pg.create_frame().value();
  const PoseTree::frame_t c = pg.create_frame().value();
  ASSERT_TRUE(pg.create_edges(a, b, 16, PoseTreeEdgeHistory::AccessMethod::kNearest));
  ASSERT_TRUE(pg.create_edges(b, c, PoseTreeEdgeHistory::AccessMethod::kPrevious));
  EXPECT_TRUE(pg.set(a, b, 1.0, Pose3d::from_translation(Vector3d(2.0, 0.0, 0.0))));
  EXPECT_TRUE(pg.set(b, c, 0.0, Pose3d::from_translation(Vector3d(1.0, 0.0, 0.0))));

  std::vector<PoseTree::frame_t> frames;
  std::vector<std::string_view> frame_names;
  std::vector<std::pair<PoseTree::frame_t, PoseTree::frame_t>> edges;
  std::vector<std::pair<std::string_view, std::string_view>> edge_names;

  ASSERT_FALSE(pg.get_frame_uids(frames));
  ASSERT_FALSE(pg.get_frame_names(frame_names));
  ASSERT_FALSE(pg.get_edge_uids(edges));
  ASSERT_FALSE(pg.get_edge_names(edge_names));

  frames.resize(3);
  frame_names.resize(3);
  edges.resize(2);
  edge_names.resize(2);

  ASSERT_TRUE(pg.get_frame_uids(frames));
  ASSERT_TRUE(pg.get_frame_names(frame_names));
  ASSERT_TRUE(pg.get_edge_uids(edges));
  ASSERT_TRUE(pg.get_edge_names(edge_names));

  EXPECT_EQ(frames.size(), 3);
  EXPECT_EQ(frame_names.size(), 3);
  EXPECT_EQ(edges.size(), 2);
  EXPECT_EQ(edge_names.size(), 2);
}

}  // namespace isaac::os::pose_tree
