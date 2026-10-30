/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

namespace isaac::os::mcap_logger {

class ScopedTempDirectory final {
 public:
  ScopedTempDirectory() {
    const auto template_path =
        std::filesystem::temp_directory_path() /  // NOSONAR: S5443 is a false positive because
                                                  // mkdtemp atomically creates a mode-0700 private
                                                  // directory.
        "isaac_mcap_logger_test_XXXXXX";
    std::string mutable_template = template_path.string();
    const char* created = ::mkdtemp(mutable_template.data());
    const int mkdtemp_errno = errno;
    if (created == nullptr) {
      throw std::system_error(
          mkdtemp_errno, std::generic_category(),
          "Unable to create a private MCAP test directory from " + template_path.string());
    }
    path_ = created;
  }

  ~ScopedTempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  ScopedTempDirectory(const ScopedTempDirectory&) = delete;
  ScopedTempDirectory& operator=(const ScopedTempDirectory&) = delete;

  const std::filesystem::path& path() const { return path_; }
  std::filesystem::path path(std::string_view filename) const { return path_ / filename; }

 private:
  std::filesystem::path path_;
};

}  // namespace isaac::os::mcap_logger
