/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_GR00T_WBC_SHUTDOWN_SIGNALS_H
#define ISAAC_OS_GR00T_WBC_SHUTDOWN_SIGNALS_H

#include <signal.h>

#include <atomic>

namespace holoscan {
class RunSession;
}

namespace isaac::os {

// Own SIGINT/SIGTERM handling for one application, including threads created by a profiler
// before main(). Construct and destroy on the main thread, outside the graph/session lifetime.
// The first occurrence requests cooperative stop; repeating the same signal terminates with
// its default disposition. Previous dispositions and the main thread's mask are restored.
class ShutdownSignals final {
 public:
  ShutdownSignals();
  ~ShutdownSignals();

  ShutdownSignals(const ShutdownSignals&) = delete;
  ShutdownSignals& operator=(const ShutdownSignals&) = delete;

  [[nodiscard]] bool stop_requested() const noexcept;
  void wait(holoscan::RunSession& session) const;

 private:
  static void request_shutdown(int) noexcept;
  inline static std::atomic_flag signal_owner_ = ATOMIC_FLAG_INIT;
  inline static std::atomic<bool> shutdown_requested_{false};
  static_assert(std::atomic<bool>::is_always_lock_free);

  struct sigaction previous_interrupt_{};
  struct sigaction previous_terminate_{};
  sigset_t previous_mask_{};
};

}  // namespace isaac::os

#endif  // ISAAC_OS_GR00T_WBC_SHUTDOWN_SIGNALS_H
