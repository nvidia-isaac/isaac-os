/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Ctrl+C mid-run must stop the reference app's graph in order and exit 0, not crash or wait out the
// run's duration. This runs the app, waits until it says it is running, sends SIGINT and checks how
// it exits.

#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <string>
#include <thread>

#include "gtest/gtest.h"

namespace isaac::os::ros2_bridge_reference {
namespace {

using Clock = std::chrono::steady_clock;

// The app prints this once its signal handlers are installed.
constexpr char kReadyLine[] = "Publishing circular_pose";
// The app is told to run far longer than the test, so a signal it ignores shows up as a missed
// shutdown deadline.
constexpr char kLongRun[] = "--duration-s=60";
constexpr auto kStartupLimit = std::chrono::seconds(30);
// Time the app runs after it is ready, so its round trip has delivered something: the app fails a
// run that delivered nothing.
constexpr auto kRunningTime = std::chrono::seconds(2);
constexpr auto kShutdownLimit = std::chrono::seconds(20);
constexpr auto kPollInterval = std::chrono::milliseconds(50);

// One run of the app with its output captured. Kills and reaps it if the test ends first.
class AppProcess {
 public:
  AppProcess() = default;
  AppProcess(const AppProcess&) = delete;
  AppProcess& operator=(const AppProcess&) = delete;

  ~AppProcess() {
    if (pid_ > 0 && !reaped_) {
      kill(pid_, SIGKILL);
      waitpid(pid_, nullptr, 0);
    }
    if (output_fd_ >= 0) close(output_fd_);
  }

  // Starts the app with stdout and stderr going to one pipe. Returns false if that fails.
  bool Start(const char* app_path) {
    std::array<int, 2> fds{};
    if (pipe(fds.data()) != 0) return false;
    pid_ = fork();
    if (pid_ < 0) {
      close(fds[0]);
      close(fds[1]);
      return false;
    }
    if (pid_ == 0) {
      dup2(fds[1], STDOUT_FILENO);
      dup2(fds[1], STDERR_FILENO);
      close(fds[0]);
      close(fds[1]);
      execl(app_path, app_path, kLongRun, nullptr);
      _exit(127);
    }
    close(fds[1]);
    output_fd_ = fds[0];
    return true;
  }

  // Reads output until `text` has appeared. False if the output ends or `limit` passes first.
  bool WaitForOutput(const char* text, Clock::duration limit) {
    const auto deadline = Clock::now() + limit;
    while (output_.find(text) == std::string::npos) {
      if (eof_ || Clock::now() >= deadline) return false;
      Read(std::min<Clock::duration>(kPollInterval, deadline - Clock::now()));
    }
    return true;
  }

  // Keeps reading output for `duration`, or until the output ends.
  void Drain(Clock::duration duration) {
    const auto deadline = Clock::now() + duration;
    while (!eof_ && Clock::now() < deadline) {
      Read(std::min<Clock::duration>(kPollInterval, deadline - Clock::now()));
    }
  }

  bool Signal(int signal_number) const { return kill(pid_, signal_number) == 0; }

  // Waits for the app to exit, reading its output meanwhile. False if `limit` passes first.
  bool WaitForExit(Clock::duration limit, int* status) {
    const auto deadline = Clock::now() + limit;
    while (true) {
      if (waitpid(pid_, status, WNOHANG) == pid_) {
        reaped_ = true;
        // Collect what the app wrote last.
        while (!eof_) Read(kPollInterval);
        return true;
      }
      if (Clock::now() >= deadline) return false;
      if (eof_) {
        std::this_thread::sleep_for(kPollInterval);
      } else {
        Read(kPollInterval);
      }
    }
  }

  const std::string& output() const { return output_; }

 private:
  // Appends whatever output arrives within `timeout`. Marks the end of the output.
  void Read(Clock::duration timeout) {
    pollfd poll_fd{output_fd_, POLLIN, 0};
    const auto timeout_ms =
        static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(timeout).count());
    const int ready = poll(&poll_fd, 1, std::max(timeout_ms, 0));
    if (ready < 0 && errno == EINTR) return;
    if (ready <= 0) {
      eof_ = eof_ || ready < 0;
      return;
    }
    std::array<char, 4096> buffer;
    const ssize_t count = read(output_fd_, buffer.data(), buffer.size());
    if (count > 0) {
      output_.append(buffer.data(), static_cast<std::size_t>(count));
    } else if (count == 0 || errno != EINTR) {
      eof_ = true;
    }
  }

  pid_t pid_ = -1;
  int output_fd_ = -1;
  bool reaped_ = false;
  bool eof_ = false;
  std::string output_;
};

TEST(Ros2BridgeReferenceSigint, StopsInOrderAndExitsCleanly) {
  AppProcess app;
  ASSERT_TRUE(app.Start(ISAAC_ROS2_BRIDGE_REFERENCE_BIN));
  ASSERT_TRUE(app.WaitForOutput(kReadyLine, kStartupLimit))
      << "The app did not reach its running state.\n"
      << app.output();
  app.Drain(kRunningTime);

  ASSERT_TRUE(app.Signal(SIGINT));
  int status = 0;
  ASSERT_TRUE(app.WaitForExit(kShutdownLimit, &status))
      << "The app did not exit within " << kShutdownLimit.count() << " s of SIGINT.\n"
      << app.output();

  ASSERT_TRUE(WIFEXITED(status)) << "The app was killed by signal "
                                 << (WIFSIGNALED(status) ? WTERMSIG(status) : 0) << ".\n"
                                 << app.output();
  EXPECT_EQ(WEXITSTATUS(status), 0) << app.output();
  EXPECT_NE(app.output().find("Stop requested"), std::string::npos)
      << "The app did not stop through its signal handler.\n"
      << app.output();
}

}  // namespace
}  // namespace isaac::os::ros2_bridge_reference
