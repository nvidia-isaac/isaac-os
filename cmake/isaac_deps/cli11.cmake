# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CLI11 dependency setup.

set(ISAAC_CLI11_VERSION 2.6.2)

if(TARGET CLI11::CLI11)
  return()
endif()

include(isaac_cpm_find_package)

isaac_cpm_find_package(
  NAME CLI11
  VERSION ${ISAAC_CLI11_VERSION}
  GITHUB_REPOSITORY CLIUtils/CLI11
  GIT_TAG v${ISAAC_CLI11_VERSION}
)
