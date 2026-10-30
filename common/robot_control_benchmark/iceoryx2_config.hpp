/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iox2/iceoryx2.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace isaac::os::benchmark {
namespace detail {

class AnonymousConfigFile {
 public:
  AnonymousConfigFile() {
    descriptor_ = ::memfd_create("robot-control-iceoryx2", MFD_CLOEXEC);
    if (descriptor_ < 0) {
      throw std::system_error(errno, std::generic_category(),
                              "Unable to create an anonymous iceoryx2 configuration");
    }
    path_ = "/proc/self/fd/" + std::to_string(descriptor_);
  }

  ~AnonymousConfigFile() {
    if (descriptor_ >= 0) {
      ::close(descriptor_);
    }
  }

  AnonymousConfigFile(const AnonymousConfigFile&) = delete;
  AnonymousConfigFile& operator=(const AnonymousConfigFile&) = delete;

  void write_all(std::string_view contents) {
    std::size_t written = 0;
    while (written < contents.size()) {
      const ssize_t result =
          ::write(descriptor_, contents.data() + written, contents.size() - written);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result <= 0) {
        throw std::system_error(errno, std::generic_category(),
                                "Unable to write the temporary iceoryx2 configuration");
      }
      written += static_cast<std::size_t>(result);
    }
  }

  void rewind() {
    if (::lseek(descriptor_, 0, SEEK_SET) < 0) {
      throw std::system_error(errno, std::generic_category(),
                              "Unable to rewind the anonymous iceoryx2 configuration");
    }
  }

  [[nodiscard]] const std::string& path() const { return path_; }

 private:
  int descriptor_ = -1;
  std::string path_;
};

}  // namespace detail

inline void configure_iceoryx2(const std::string& prefix) {
  const bool valid = !prefix.empty() && prefix.size() <= 255U &&
                     std::all_of(prefix.begin(), prefix.end(), [](unsigned char character) {
                       return std::isalnum(character) != 0 || character == '-' ||
                              character == '.' || character == ':' || character == '_';
                     });
  if (!valid) {
    throw std::runtime_error(
        "The iceoryx2 prefix must contain only letters, digits, '-', '.', ':', or '_'.");
  }

  detail::AnonymousConfigFile config_file;
  config_file.write_all("[global]\nprefix = \"" + prefix + "\"\n");
  config_file.rewind();

  iox2_config_ptr config = nullptr;
  const int result = iox2_config_setup_global_config_from_file(&config, config_file.path().c_str());
  if (result != IOX2_OK) {
    const auto error = static_cast<iox2_config_creation_error_e>(result);
    throw std::runtime_error(
        std::string("Unable to configure the iceoryx2 shared-memory namespace: ") +
        iox2_config_creation_error_string(error) + ".");
  }
  std::fprintf(stderr, "iceoryx2 shared-memory prefix: %s\n", prefix.c_str());
}

}  // namespace isaac::os::benchmark
