/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "applications/gr00t_n1_7_wbc_g1/shutdown_signals.h"

#include <pthread.h>

#include <cerrno>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <system_error>
#include <thread>

#include <holoscan/core/run_session.hpp>

namespace isaac::os {
namespace {

void restore_action(int signal, const struct sigaction& action) noexcept {
  if (sigaction(signal, &action, nullptr) != 0) {
    std::terminate();
  }
}

}  // namespace

// Signal handlers may only touch lock-free state.
void ShutdownSignals::request_shutdown(int) noexcept { shutdown_requested_.store(true); }

ShutdownSignals::ShutdownSignals() {
  if (signal_owner_.test_and_set()) {
    throw std::logic_error("Shutdown signal handling is already active.");
  }
  shutdown_requested_.store(false);

  struct sigaction action{};
  action.sa_handler = &request_shutdown;
  sigemptyset(&action.sa_mask);
  sigaddset(&action.sa_mask, SIGINT);
  sigaddset(&action.sa_mask, SIGTERM);
  action.sa_flags = SA_RESTART | SA_RESETHAND;

  if (sigaction(SIGINT, &action, &previous_interrupt_) != 0) {
    const int error = errno;
    signal_owner_.clear();
    throw std::system_error(error, std::generic_category(), "Unable to handle interrupt signals");
  }
  if (sigaction(SIGTERM, &action, &previous_terminate_) != 0) {
    const int error = errno;
    restore_action(SIGINT, previous_interrupt_);
    signal_owner_.clear();
    throw std::system_error(error, std::generic_category(), "Unable to handle termination signals");
  }

  // Receive shutdown signals even if the launcher blocked them.
  const int error = pthread_sigmask(SIG_UNBLOCK, &action.sa_mask, &previous_mask_);
  if (error != 0) {
    restore_action(SIGTERM, previous_terminate_);
    restore_action(SIGINT, previous_interrupt_);
    signal_owner_.clear();
    throw std::system_error(error, std::generic_category(), "Unable to receive shutdown signals");
  }
}

ShutdownSignals::~ShutdownSignals() {
  restore_action(SIGTERM, previous_terminate_);
  restore_action(SIGINT, previous_interrupt_);
  if (pthread_sigmask(SIG_SETMASK, &previous_mask_, nullptr) != 0) {
    std::terminate();
  }
  signal_owner_.clear();
}

bool ShutdownSignals::stop_requested() const noexcept { return shutdown_requested_.load(); }

void ShutdownSignals::wait(holoscan::RunSession& session) const {
  while (session.is_running()) {
    if (stop_requested()) {
      session.request_stop();
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  }
  session.wait();
}

}  // namespace isaac::os
