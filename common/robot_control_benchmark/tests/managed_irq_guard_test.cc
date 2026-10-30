/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/robot_control_benchmark/managed_irq_guard.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace isaac::os::benchmark {
namespace {

class ManagedIrqGuardTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto template_path =
        std::filesystem::temp_directory_path() /  // NOSONAR: mkdtemp atomically creates a private
                                                  // mode-0700 directory; S5443 is a false positive.
        "robot-control-irq-test.XXXXXX";
    std::string pattern = template_path.string();
    const char* directory = mkdtemp(pattern.data());
    ASSERT_NE(directory, nullptr);
    root_ = directory;
    table();
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  void write(const std::string& content) const {
    std::ofstream output(root() / "interrupts");
    output << content;
    ASSERT_TRUE(output.good());
  }

  void table(const std::string& counts = "10 20 0 0 0 0 30 40",
             const std::string& other = "50 60 0 0 0 0 70 80") const {
    write(" CPU0 CPU8 CPU9 CPU10 CPU11 CPU12 CPU13 CPU14\n42: " + counts +
          " PCI-MSI nvme0q1\n43: 0 0 0 0 0 0 0 0 PCI-MSI nvme1q12\n44: " + other +
          " GIC arch_timer\nIPI0: " + other + " Rescheduling interrupts\n");
  }

  static std::string warnings(const ManagedIrqGuard& guard) {
    testing::internal::CaptureStderr();
    guard.check();
    return testing::internal::GetCapturedStderr();
  }

  const std::filesystem::path& root() const { return root_; }

 private:
  std::filesystem::path root_;
};

TEST_F(ManagedIrqGuardTest, UnchangedCountersProduceNoOutput) {
  const ManagedIrqGuard guard({8, 9, 10, 11, 12, 13}, {"nvme"}, root());
  testing::internal::CaptureStdout();
  EXPECT_TRUE(warnings(guard).empty());
  EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
}

TEST_F(ManagedIrqGuardTest, HousekeepingNvmeAndOtherInterruptSourcesAreIgnored) {
  const ManagedIrqGuard guard({8, 9, 10, 11, 12, 13}, {"nvme"}, root());
  table("100 20 0 0 0 0 30 400", "500 600 0 0 0 0 700 800");
  EXPECT_TRUE(warnings(guard).empty());
}

TEST_F(ManagedIrqGuardTest, ActivityWarnsOnStderrWithoutCsvOutput) {
  const ManagedIrqGuard guard({8, 9, 10, 11, 12, 13}, {"nvme"}, root());
  table("100 21 2 3 4 5 36 400");
  testing::internal::CaptureStdout();
  EXPECT_EQ(warnings(guard),
            "Warning: Observed 21 interrupts on CPUs 8 9 10 11 12 13 during the run. Source "
            "substrings: 'nvme'. "
            "Interrupt handling can delay benchmark work and increase the measured latency. "
            "Managed IRQs are routed by the kernel and may still target isolated CPUs despite "
            "CPU isolation settings: isolcpus=managed_irq provides best-effort isolation and "
            "cannot redirect an IRQ if its affinity mask contains no online non-isolated CPU.\n");
  EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
}

TEST_F(ManagedIrqGuardTest, WatchesOnlyRequestedCpusIncludingOutsideTheThorRange) {
  const ManagedIrqGuard guard({14, 0, 14}, {"nvme"}, root());
  table("12 200 10 10 10 10 300 43");
  EXPECT_EQ(
      warnings(guard),
      "Warning: Observed 5 interrupts on CPUs 0 14 during the run. Source substrings: 'nvme'. "
      "Interrupt handling can delay benchmark work and increase the measured latency. "
      "Managed IRQs are routed by the kernel and may still target isolated CPUs despite "
      "CPU isolation settings: isolcpus=managed_irq provides best-effort isolation and "
      "cannot redirect an IRQ if its affinity mask contains no online non-isolated CPU.\n");
}

TEST_F(ManagedIrqGuardTest, UnselectedCpuColumnsAreNotRequired) {
  write(" CPU2 CPU5\n42: 10 20 PCI-MSI nvme0q1\n");
  const ManagedIrqGuard guard({5}, {"nvme"}, root());
  write(" CPU2 CPU5\n42: 100 23 PCI-MSI nvme0q1\n");
  EXPECT_EQ(warnings(guard),
            "Warning: Observed 3 interrupts on CPUs 5 during the run. Source substrings: 'nvme'. "
            "Interrupt handling can delay benchmark work and increase the measured latency. "
            "Managed IRQs are routed by the kernel and may still target isolated CPUs despite "
            "CPU isolation settings: isolcpus=managed_irq provides best-effort isolation and "
            "cannot redirect an IRQ if its affinity mask contains no online non-isolated CPU.\n");
}

TEST_F(ManagedIrqGuardTest, MatchesMultipleSubstringsAnywhereInSourceNamesOncePerRow) {
  write(
      " CPU2 CPU5\n"
      "42: 10 20 PCI-MSI pci-nvme0q1\n"
      "43: 30 40 PCI-MSI eth0-TxRx-0\n"
      "44: 50 60 nvme-controller arch_timer\n");
  const ManagedIrqGuard guard({5}, {"nvme", "TxRx", "nvme0"}, root());
  write(
      " CPU2 CPU5\n"
      "42: 100 22 PCI-MSI pci-nvme0q1\n"
      "43: 300 43 PCI-MSI eth0-TxRx-0\n"
      "44: 500 600 nvme-controller arch_timer\n");
  EXPECT_EQ(warnings(guard),
            "Warning: Observed 5 interrupts on CPUs 5 during the run. "
            "Source substrings: 'nvme' 'TxRx' 'nvme0'. "
            "Interrupt handling can delay benchmark work and increase the measured latency. "
            "Managed IRQs are routed by the kernel and may still target isolated CPUs despite "
            "CPU isolation settings: isolcpus=managed_irq provides best-effort isolation and "
            "cannot redirect an IRQ if its affinity mask contains no online non-isolated CPU.\n");
}

TEST_F(ManagedIrqGuardTest, CustomSourceSelectionDoesNotImplicitlyIncludeNvme) {
  const ManagedIrqGuard guard({8}, {"arch_timer"}, root());
  table("10 200 0 0 0 0 30 40");
  EXPECT_TRUE(warnings(guard).empty());
  table("10 200 0 0 0 0 30 40", "50 63 0 0 0 0 70 80");
  EXPECT_EQ(warnings(guard),
            "Warning: Observed 3 interrupts on CPUs 8 during the run. "
            "Source substrings: 'arch_timer'. "
            "Interrupt handling can delay benchmark work and increase the measured latency. "
            "Managed IRQs are routed by the kernel and may still target isolated CPUs despite "
            "CPU isolation settings: isolcpus=managed_irq provides best-effort isolation and "
            "cannot redirect an IRQ if its affinity mask contains no online non-isolated CPU.\n");
}

TEST_F(ManagedIrqGuardTest, EmptySourceSelectionOrSubstringWarns) {
  EXPECT_NE(warnings(ManagedIrqGuard({8}, {}, root())).find("nonempty interrupt source"),
            std::string::npos);
  EXPECT_NE(warnings(ManagedIrqGuard({8}, {"nvme", ""}, root())).find("nonempty interrupt source"),
            std::string::npos);
}

TEST_F(ManagedIrqGuardTest, EmptyOrMissingCpuSelectionWarns) {
  EXPECT_NE(warnings(ManagedIrqGuard({}, {"nvme"}, root())).find("Warning: Unable to check"),
            std::string::npos);
  EXPECT_NE(warnings(ManagedIrqGuard({2}, {"nvme"}, root())).find("CPU 2 exactly once"),
            std::string::npos);
  const ManagedIrqGuard guard({14}, {"nvme"}, root());
  write(" CPU0 CPU8\n42: 10 20 PCI-MSI nvme0q1\n");
  EXPECT_NE(warnings(guard).find("CPU 14 exactly once"), std::string::npos);
}

TEST_F(ManagedIrqGuardTest, MatchesCpuLabelsRatherThanColumnPositions) {
  write(" CPU13 CPU0 CPU12 CPU11 CPU10 CPU9 CPU8\n42: 1 10 2 3 4 5 6 PCI-MSI nvme0q1\n");
  const ManagedIrqGuard guard({8, 9, 10, 11, 12, 13}, {"nvme"}, root());
  write(" CPU0 CPU8 CPU9 CPU10 CPU11 CPU12 CPU13\n42: 100 6 5 4 3 2 2 PCI-MSI nvme0q1\n");
  EXPECT_NE(warnings(guard).find("Observed 1 interrupts"), std::string::npos);
}

TEST_F(ManagedIrqGuardTest, IrqIdentitiesAndVectorCountsNeedNotMatch) {
  const ManagedIrqGuard guard({8, 9, 10, 11, 12, 13}, {"nvme"}, root());
  write(" CPU0 CPU8 CPU9 CPU10 CPU11 CPU12 CPU13\n99: 0 50 0 0 0 0 0 PCI-MSI nvme2q3   \n");
  EXPECT_TRUE(warnings(guard).empty());
}

TEST_F(ManagedIrqGuardTest, CounterDecreaseWarnsThatTheCheckCouldNotComplete) {
  const ManagedIrqGuard guard({8, 9, 10, 11, 12, 13}, {"nvme"}, root());
  table("10 19 0 0 0 0 30 40");
  EXPECT_EQ(warnings(guard),
            "Warning: Unable to check interrupt activity. The matching interrupt total "
            "decreased; the run cannot be verified.\n");
}

TEST_F(ManagedIrqGuardTest, MissingFinalEvidenceWarns) {
  const ManagedIrqGuard guard({8, 9, 10, 11, 12, 13}, {"nvme"}, root());
  std::filesystem::remove(root() / "interrupts");
  EXPECT_NE(warnings(guard).find("Unable to read IRQ evidence"), std::string::npos);
}

TEST_F(ManagedIrqGuardTest, MissingNvmeRowsWarn) {
  const ManagedIrqGuard guard({8, 9, 10, 11, 12, 13}, {"nvme"}, root());
  write(" CPU0 CPU8 CPU9 CPU10 CPU11 CPU12 CPU13\n44: 0 0 0 0 0 0 0 GIC arch_timer\n");
  EXPECT_NE(warnings(guard).find("no matching interrupt counters"), std::string::npos);
}

TEST_F(ManagedIrqGuardTest, MalformedCountersAndOverflowWarn) {
  for (const std::string counts : {"10 x 0 0 0 0 30 40", "10 -1 0 0 0 0 30 40",
                                   "0 18446744073709551615 1 0 0 0 0 0", "10 20"}) {
    table(counts);
    EXPECT_NE(warnings(ManagedIrqGuard({8, 9, 10, 11, 12, 13}, {"nvme"}, root()))
                  .find("Warning: Unable to check"),
              std::string::npos)
        << counts;
  }
  table("0 18446744073709551615 0 0 0 0 0 0");
  std::ofstream(root() / "interrupts", std::ios::app) << "45: 0 1 0 0 0 0 0 0 PCI-MSI nvme2q1\n";
  EXPECT_NE(warnings(ManagedIrqGuard({8}, {"nvme"}, root())).find("overflowed"), std::string::npos);
}

TEST_F(ManagedIrqGuardTest, MissingDuplicateAndInvalidCpuColumnsWarn) {
  for (const std::string header : {"", "CPU0 CPU8", "CPU8 CPU9 CPU10 CPU11 CPU12 CPU13 CPU13",
                                   "CPU8 CPU9 CPU10 CPU11 CPU12 CPU13 CPUx"}) {
    write(header + "\n42: 0 0 0 0 0 0 0 PCI-MSI nvme0q1\n");
    EXPECT_NE(warnings(ManagedIrqGuard({8, 9, 10, 11, 12, 13}, {"nvme"}, root()))
                  .find("Warning: Unable to check"),
              std::string::npos)
        << header;
  }
}

TEST_F(ManagedIrqGuardTest, InitialFailureIsRetainedUntilCheckEvenIfProcfsRecovers) {
  std::filesystem::remove(root() / "interrupts");
  const ManagedIrqGuard guard({8, 9, 10, 11, 12, 13}, {"nvme"}, root());
  table();
  EXPECT_NE(warnings(guard).find("Unable to read IRQ evidence"), std::string::npos);
}

}  // namespace
}  // namespace isaac::os::benchmark
