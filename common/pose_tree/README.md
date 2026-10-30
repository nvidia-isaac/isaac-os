# Pose tree

Stores the transform between named coordinate frames over time, and answers where one frame was
relative to another at a given instant.

Frames are named, related in pairs, and written to as often as a sensor or an estimator produces a
new measurement. A reader asks for a pair at a time, and the tree walks whatever chain of edges
connects them, interpolating each to the instant asked for. The two frames need not be related
directly: a camera placed in a rig once at start-up, and that rig tracked against the world every
frame, is enough to answer where the camera was at any point in between.

Everything here is `isaac::os::pose_tree`.

## Using it

```cpp
#include "common/pose_tree/pose_tree.hpp"

isaac::os::pose_tree::PoseTree tree;
if (!tree.init()) { /* the tree could not reserve its storage */ }
```

`init()` takes the capacities up front -- how many frames, how many edges, how long a history --
and allocates once. It is the only allocating call; nothing below grows the tree, and the defaults
hold roughly a thousand frames and a million pose samples.

Name the frames, then relate them:

```cpp
const auto odom = tree.find_or_create_frame("odom");
const auto rig = tree.find_or_create_frame("rig");
if (!odom || !rig) { /* the name is too long, or the tree is full */ }
if (!tree.create_edges(*odom, *rig)) { /* already related, or out of edges */ }
```

Frames have to exist before an edge can relate them: the overloads taking names look them up rather
than creating them, so `create_edges("odom", "rig")` fails if either name is new. Declaring the edge
before anything is written to it is what lets a reader ask early and get an empty edge rather than a
missing one.

Write and read against time, in seconds:

```cpp
tree.set(*odom, *rig, at_seconds, pose);
const auto pose = tree.get(*odom, *rig, at_seconds);
```

`get` interpolates between the samples either side of the instant asked for. Overloads select how --
nearest sample, previous sample, linear, or spherical for the rotation -- and one takes a version
rather than a time, so a batch of queries can be pinned to one consistent state of the tree while a
writer keeps going. Every call returns an `expected`: an unrelated pair, a time before the history
starts, and an unknown frame are ordinary answers rather than exceptions.

Reads and writes are safe from several threads. A writer minting frames from more than one thread
calls `set_multithreading_info` first, which gives each thread a disjoint run of frame ids so two
threads cannot mint the same one.

`add_create_frame_callback` and `add_set_edge_callback` register a function to run whenever a frame
is created or an edge is written. **A callback must not register or deregister callbacks.** The
registry is held for reading while a callback runs and registering needs it for writing, so a
callback that does either waits on a lock it is itself holding and the thread stops there. The tree
does not detect this: the alternative is to invoke callbacks with the registry unlocked, which
would let one thread deregister a callback while another is part-way through running it, and free
what that callback was about to touch. Reading the tree from a callback is fine, since the tree's
own lock is released first; writing to it re-enters the notification you are already inside.

## Handing the tree to an operator

The application creates one tree and passes a `std::shared_ptr` to it as a construction argument to
each operator that needs it. The graph copies that argument into every operator it builds, so each
owns a share of the same tree and the tree outlives all of them, however many times an operator is
rebuilt.

Keeping the tree out of an operator's parameters is what lets those parameters still be bound to a
configuration file. Binding checks a digest of the whole parameter value, and shared ownership has
none, so an operator that carried the tree there could not be configured from a file at all.

One caveat, because it varies by SDK build. `GraphOwnedCapture`, the concept constraining
`Graph::op<>()`, is written to reject a top-level smart pointer, and some builds do reject one there
while others accept it. The rejection is of top-level pointers only, so where it applies the answer
is a struct holding the handle:

```cpp
struct PoseTreeHandle {
  std::shared_ptr<isaac::os::pose_tree::PoseTree> tree;
};
```

Nested in a value like that it passes, and the graph still copies one share per operator.

## What is here

| Path | Contents |
| --- | --- |
| `pose_tree.hpp` | The tree |
| `pose_tree_history.hpp` | One edge's poses over time |
| `math/` | The linear algebra layer the two are written against, and what a caller reads a pose as |
| `detail/` | The hash map, the index map, and the allocator the history buffers come from |
| `tests/` | Unit tests covering the tree, the edge history, and what a failed call leaves behind |

Everything is reached as `common/pose_tree/...`, from the repository root, so an include says
who owns the header and cannot be mistaken for, or shadow, one of the SDK's.

Only the single-process tree is here. There is no distributed client or server, and nothing that
publishes it as a graph service: SDK 5 has no service registry, so an application constructs one
tree and hands it to the operators that need it.
