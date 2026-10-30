/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "common/robot_control_benchmark/arm_kinematics.hpp"
#include "common/robot_control_benchmark/gpu_ik_workload.hpp"
#include "common/robot_control_benchmark/gpu_motion_ik.hpp"

namespace isaac::os::benchmark {
namespace {

using gpu_ik::kJoints;
using gpu_ik::kSeeds;
using gpu_ik::SolveInput;

/// Throws an actionable runtime error when a CUDA operation fails.
void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

// Sixteen independent blocks let the candidate work span SMs. Each thread owns one
// serial DLS solve; the second kernel selects across all blocks on the same stream.
constexpr int kThreads = 64;
constexpr int kBlocks = kSeeds / kThreads;
static_assert(kSeeds % kThreads == 0);

/// Executes all 1,024 single-precision solves and writes coalesced candidate arrays.
__global__ void ik_kernel(const SolveInput* input, float* candidates, float* scores) {
  const int seed = blockIdx.x * blockDim.x + threadIdx.x;
  float q[kJoints];
  scores[seed] = gpu_ik::solve_candidate(*input, seed, q);
  for (int j = 0; j < kJoints; ++j) {
    candidates[j * kSeeds + seed] = q[j];
  }
}

/// Selects the lowest score, resolving ties by seed index, with a bounded hold fallback.
__global__ void select_kernel(const SolveInput* input, const float* candidates, const float* scores,
                              float* q_out) {
  __shared__ float best_scores[kThreads];
  __shared__ int best_seeds[kThreads];
  const int lane = threadIdx.x;
  float best_score = INFINITY;
  int best_seed = kSeeds;
  for (int seed = lane; seed < kSeeds; seed += kThreads) {
    const float score = scores[seed];
    if (score < best_score) {
      best_score = score;
      best_seed = seed;
    }
  }
  best_scores[lane] = best_score;
  best_seeds[lane] = best_seed;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (lane < stride) {
      const float other_score = best_scores[lane + stride];
      const int other_seed = best_seeds[lane + stride];
      if (other_score < best_scores[lane] ||
          (other_score == best_scores[lane] && other_seed < best_seeds[lane])) {
        best_scores[lane] = other_score;
        best_seeds[lane] = other_seed;
      }
    }
    __syncthreads();
  }
  if (lane == 0) {
    float hold[kJoints];
    for (int j = 0; j < kJoints; ++j) {
      hold[j] = detail::kin_clamp(input->measured_q[j], input->lower[j], input->upper[j]);
    }
    const bool use_candidate =
        best_seeds[0] < kSeeds && best_scores[0] < gpu_ik::command_score(*input, hold);
    for (int j = 0; j < kJoints; ++j) {
      q_out[j] = use_candidate ? candidates[j * kSeeds + best_seeds[0]] : hold[j];
    }
  }
}

}  // namespace

/// CUDA resources and staging buffers hidden from host-only callers.
struct GpuDlsIkBackend::Impl {
  struct DeviceWorkspace {
    // Preserve coalesced candidate writes and aligned candidate/score array starts.
    float candidates[kSeeds * kJoints];
    float scores[kSeeds];
    SolveInput input;
    float q_out[kJoints];
  };

  // Pinned staging keeps per-tick copies asynchronous and holds the narrowed inputs.
  struct HostWorkspace {
    SolveInput input;
    float q_out[kJoints];
  };

  struct PreparedGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    unsigned long long stream_id = 0;

    ~PreparedGraph() {
      if (executable != nullptr) {
        static_cast<void>(cudaGraphExecDestroy(executable));
      }
      if (graph != nullptr) {
        static_cast<void>(cudaGraphDestroy(graph));
      }
    }
  };

  cudaStream_t stream = nullptr;
  cudaEvent_t completion_event = nullptr;
  DeviceWorkspace* device = nullptr;
  HostWorkspace* host = nullptr;
  std::unique_ptr<PreparedGraph> prepared;

  /// Releases successfully allocated resources, including after partial initialization.
  ~Impl() {
    prepared.reset();
    if (completion_event != nullptr) {
      static_cast<void>(cudaEventDestroy(completion_event));
    }
    if (host != nullptr) {
      static_cast<void>(cudaFreeHost(host));
    }
    if (device != nullptr) {
      static_cast<void>(cudaFree(device));
    }
    if (stream != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream));
    }
  }

  /// Records copies and kernels against fixed workspaces on the caller's actual stream.
  void record_work(cudaStream_t solve_stream) {
    check_cuda(cudaMemcpyAsync(&device->input, &host->input, sizeof(SolveInput),
                               cudaMemcpyHostToDevice, solve_stream),
               "Unable to upload the GPU IK inputs");

    ik_kernel<<<kBlocks, kThreads, 0, solve_stream>>>(&device->input, device->candidates,
                                                      device->scores);
    check_cuda(cudaGetLastError(), "The GPU IK solve kernel failed to launch");

    select_kernel<<<1, kThreads, 0, solve_stream>>>(&device->input, device->candidates,
                                                    device->scores, device->q_out);
    check_cuda(cudaGetLastError(), "The GPU IK selection kernel failed to launch.");

    check_cuda(cudaMemcpyAsync(host->q_out, device->q_out, kJoints * sizeof(float),
                               cudaMemcpyDeviceToHost, solve_stream),
               "Unable to download the GPU IK result");
  }

  /// Prepares once per stream, preserving the capture stream's Green Context placement.
  void prepare_graph(cudaStream_t solve_stream) {
    unsigned long long stream_id = 0;
    check_cuda(cudaStreamGetId(solve_stream, &stream_id), "Unable to identify the GPU IK stream.");
    if (prepared && prepared->stream_id == stream_id) {
      return;
    }
    auto next = std::make_unique<PreparedGraph>();
    next->stream_id = stream_id;
    check_cuda(cudaStreamBeginCapture(solve_stream, cudaStreamCaptureModeThreadLocal),
               "Unable to begin GPU IK graph preparation.");
    try {
      record_work(solve_stream);
    } catch (...) {
      // End even an invalidated capture so a borrowed stream remains usable on failure.
      static_cast<void>(cudaStreamEndCapture(solve_stream, &next->graph));
      throw;
    }
    check_cuda(cudaStreamEndCapture(solve_stream, &next->graph),
               "Unable to finish GPU IK graph preparation.");
    check_cuda(cudaGraphInstantiate(&next->executable, next->graph, 0),
               "Unable to instantiate the GPU IK graph.");
    prepared = std::move(next);
  }
};

/// Allocates and warms all resources used by the measured solve path.
GpuDlsIkBackend::GpuDlsIkBackend() : impl_(std::make_unique<Impl>()) {
  check_cuda(cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking),
             "Unable to create the GPU IK stream");
  check_cuda(cudaEventCreateWithFlags(&impl_->completion_event, cudaEventDisableTiming),
             "Unable to create the GPU IK completion event.");
  check_cuda(cudaMalloc(&impl_->device, sizeof(Impl::DeviceWorkspace)),
             "Unable to allocate the GPU IK device workspace.");
  check_cuda(cudaMallocHost(&impl_->host, sizeof(Impl::HostWorkspace)),
             "Unable to allocate the GPU IK pinned host workspace.");

  // Warm up so the first measured tick does not pay module load or lazy context
  // initialization. This is the arm's home configuration, which is valid and finite.
  MotionInput warm{};
  warm.measured_q = ArmModel::home();
  warm.target_p[0] = 0.45;
  warm.target_p[2] = 0.5;
  warm.target_R = {1, 0, 0, 0, -1, 0, 0, 0, -1};
  compute(warm);
}

GpuDlsIkBackend::~GpuDlsIkBackend() = default;

/// Executes one synchronous IK solve and copies its result to host memory.
void GpuDlsIkBackend::compute(MotionInput& in, CudaStreamHandle solve_stream) {
  const cudaStream_t stream = solve_stream == nullptr ? impl_->stream : solve_stream;
  // The runtime's Green Context stream first becomes available during warmup. Recapture
  // when it changes; replaying a graph captured on the owned stream would lose SM isolation.
  impl_->prepare_graph(stream);
  // Copy fresh CPU inputs to fixed pinned addresses immediately before replay. The graph
  // uploads this staging region and downloads into the pinned output region on every tick.
  impl_->host->input =
      gpu_ik::make_solve_input(in.measured_q.data(), in.target_p, in.target_R.data());
  check_cuda(cudaGraphLaunch(impl_->prepared->executable, stream),
             "Unable to launch the GPU IK graph.");
  check_cuda(cudaEventRecord(impl_->completion_event, stream),
             "Unable to record GPU IK completion.");
  // Keep the control thread active until the selected command reaches host memory.
  // This lowers latency at the cost of increased cpu utilization for the purpose of accurately
  // measuring GPU latency. In practice: this event could be waited on with a deadline based on the
  // control loop deadline.
  cudaError_t completion;
  while ((completion = cudaEventQuery(impl_->completion_event)) == cudaErrorNotReady) {
  }
  check_cuda(completion, "Unable to complete the GPU IK solve.");

  for (int j = 0; j < kJoints; ++j) {
    in.q_warm_start[j] = static_cast<double>(impl_->host->q_out[j]);
  }
}

}  // namespace isaac::os::benchmark
