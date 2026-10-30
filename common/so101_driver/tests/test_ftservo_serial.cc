/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <thread>

#include "ftservo/SCSerial.h"
#include "gtest/gtest.h"

namespace {
class Serial : public SCSerial {
 public:
  explicit Serial(int descriptor) {
    fd = descriptor;
    IOTimeOut = 40;
  }
  using SCSerial::readSCS;
};

class SerialTest : public ::testing::Test {
 protected:
  void SetUp() override { ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets_.data()), 0); }
  void TearDown() override {
    close(sockets_[0]);
    if (sockets_[1] >= 0) close(sockets_[1]);
  }
  int reader() const { return sockets_[0]; }
  int writer() const { return sockets_[1]; }
  void disconnect() {
    close(sockets_[1]);
    sockets_[1] = -1;
  }

 private:
  std::array<int, 2> sockets_{-1, -1};
};

TEST_F(SerialTest, DisconnectReturnsForBothOverloads) {
  Serial serial(reader());
  disconnect();
  std::array<unsigned char, 4> data{};
  EXPECT_EQ(serial.readSCS(data.data(), 4), 0);
  EXPECT_EQ(serial.readSCS(data.data(), 4, 40), 0);
}

TEST_F(SerialTest, PartialReadSurvivesDisconnect) {
  Serial serial(reader());
  ASSERT_EQ(write(writer(), "ab", 2), 2);
  disconnect();
  std::array<unsigned char, 4> data{};
  EXPECT_EQ(serial.readSCS(data.data(), 4, 40), 2);
  EXPECT_EQ(data[0], 'a');
  EXPECT_EQ(data[1], 'b');
}

TEST_F(SerialTest, TimeoutAndRecovery) {
  Serial serial(reader());
  std::array<unsigned char, 4> data{};
  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(serial.readSCS(data.data(), 4), 0);
  EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(30));
  ASSERT_EQ(write(writer(), "abcd", 4), 4);
  EXPECT_EQ(serial.readSCS(data.data(), 4), 4);
}

TEST_F(SerialTest, PartialReadsUseOneDeadline) {
  Serial serial(reader());
  ASSERT_EQ(write(writer(), "a", 1), 1);
  std::jthread write_thread([descriptor = writer()] {
    for (int i = 0; i < 9; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (write(descriptor, "a", 1) != 1) return;
    }
  });
  std::array<unsigned char, 16> data{};
  const auto start = std::chrono::steady_clock::now();
  const int count = serial.readSCS(data.data(), 16, 40);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  write_thread.join();
  EXPECT_GT(count, 0);
  EXPECT_LT(count, 10);
  EXPECT_LT(elapsed, std::chrono::milliseconds(90));
}

TEST_F(SerialTest, ReadErrorReturnsWithoutChangingBuffer) {
  // Directories are selectable on Linux, but read() fails with EISDIR.
  const int directory = open(".", O_RDONLY | O_DIRECTORY);
  ASSERT_GE(directory, 0);
  Serial serial(directory);
  std::array<unsigned char, 4> data{1, 2, 3, 4};
  EXPECT_EQ(serial.readSCS(data.data(), 4), 0);
  EXPECT_EQ(data[0], 1);
  close(directory);
}

TEST_F(SerialTest, SelectErrorReturnsWithoutChangingBuffer) {
  const int descriptor = dup(reader());
  ASSERT_GE(descriptor, 0);
  close(descriptor);
  Serial serial(descriptor);
  std::array<unsigned char, 4> data{1, 2, 3, 4};
  EXPECT_EQ(serial.readSCS(data.data(), 4), 0);
  EXPECT_EQ(data[0], 1);
}

TEST_F(SerialTest, InvalidDescriptorReturnsWithoutReading) {
  Serial serial(-1);
  std::array<unsigned char, 4> data{};
  EXPECT_EQ(serial.readSCS(data.data(), 4), 0);
  // A descriptor beyond the fd_set capacity must also be rejected.
  Serial oversized(FD_SETSIZE);
  EXPECT_EQ(oversized.readSCS(data.data(), 4), 0);
}
}  // namespace
