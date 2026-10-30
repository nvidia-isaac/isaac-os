/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>

#include <holoscan/core/graph.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include "operators/ucxx_ops/ucxx_connect_op.h"
#include "operators/ucxx_ops/ucxx_endpoint.h"
#include "operators/ucxx_ops/ucxx_type_aliases.h"
#include "operators/unitree_sim_ucx/synthetic_clock_advance_op.h"
#include "operators/unitree_sim_ucx/unitree_sim_ucx_ops_adapter.h"

namespace isaac::os::ops {

struct UnitreeSimUcxSubgraph {
  holoscan::OperatorHandle<UnitreeSimStateOp> state;
  holoscan::OperatorHandle<ImageUcxxReceiverOp> image;
  holoscan::OperatorHandle<JointCommandUcxxSenderOp> sender;
};

/// Add the simulator transport and paired-state boundary once per graph.
/// The caller places state and sender on its control partition, places the image receiver
/// with its camera adapter (or drains it), and binds image_rx/out_message to the CUDA device.
/// The supplied clock must also be the graph's default clock.
inline UnitreeSimUcxSubgraph add_unitree_sim_ucx_subgraph(
    holoscan::Graph& graph, const std::string& hostname, int port,
    const std::shared_ptr<holoscan::SyntheticClock>& clock, UcxxConnectOptions connection = {}) {
  UcxxEndpoint endpoint{hostname, port, false};
  auto connector = graph.op<UcxxConnectOp>("ucx_connect", endpoint, connection);
  auto joint_state_rx = graph.op<JointStateUcxxReceiverOp>(
      "joint_state_rx", endpoint, 0xe2e1a077ULL, 65536U, 0U, TimestampSource::kMessage);
  auto imu_rx = graph.op<ImuUcxxReceiverOp>("imu_rx", endpoint, 0x59efb15fULL, 65536U, 0U,
                                            TimestampSource::kMessage);
  // Clock messages must be delivered before advancing the clock to their payload time.
  auto time_rx = graph.op<TimeUcxxReceiverOp>("time_rx", endpoint, 0xb1ebcedfULL, 65536U, 0U,
                                              TimestampSource::kGraphClock);
  auto image_rx = graph.op<ImageUcxxReceiverOp>("image_rx", endpoint, 0x1018c86bULL, 65536U,
                                                1920U * 1080U * 4U, TimestampSource::kMessage);
  auto command_tx = graph.op<JointCommandUcxxSenderOp>("command_tx", endpoint, 0x3ad708d3ULL);
  auto advance =
      graph.op<SyntheticClockAdvanceOp>("advance_simulation_clock", SyntheticClockState{clock});
  auto state = graph.op<UnitreeSimStateOp>("joint_state_subscriber");
  const holoscan::ConnectionOptions sensor_queue{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                                 .queue_depth = 2U};
  graph.add_flow(joint_state_rx->out_message, state->joint_state_input, sensor_queue);
  graph.add_flow(imu_rx->out_message, state->imu_input, sensor_queue);
  graph.add_flow(time_rx->out_message, advance->input);
  graph.partition("connection").add(connector);
  graph.partition("joint_state_transport").add(joint_state_rx);
  graph.partition("imu_transport").add(imu_rx);
  graph.partition("time_transport").add(time_rx).add(advance);
  return {.state = state, .image = image_rx, .sender = command_tx};
}

}  // namespace isaac::os::ops
