# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script - Ensure a resource is present and valid (build-time entry point).
#
# Required -D variables:
# - URL: Resource source URL.
# - CACHE_DIR: Directory for metadata, stamps, archive artifacts, and content/ subdir.
#
# Optional -D variables:
# - LINK_DIR: Directory at which to create a symlink pointing to the resolved view.
# - EXPECTED_HASH: Identity hash in ALGO=hex format.
# - STRIP_PREFIX: Sub-path under content/ to expose as the resolved view (archive mode).
# - PATCH_FILE: Patch to apply after extraction (archive mode).
# - ARCHIVE_PATTERNS: Semicolon-separated archive patterns for filtered extraction.
# - KEEP_ARCHIVE: If TRUE, keep the downloaded archive after extraction.
# - LOCK_TIMEOUT: Seconds to wait for lock acquisition (empty = wait indefinitely).
# - RESOURCE_LABEL: Friendly resource name for logs/errors.

include("${CMAKE_CURRENT_LIST_DIR}/isaac_resource_helpers.cmake")

set(_keep_archive_flag "")
if(KEEP_ARCHIVE)
  set(_keep_archive_flag "KEEP_ARCHIVE")
endif()

isaac_resource_ensure(
  RESOURCE_LABEL "${RESOURCE_LABEL}"
  URL "${URL}"
  CACHE_DIR "${CACHE_DIR}"
  LINK_DIR "${LINK_DIR}"
  EXPECTED_HASH "${EXPECTED_HASH}"
  STRIP_PREFIX "${STRIP_PREFIX}"
  PATCH_FILE "${PATCH_FILE}"
  ARCHIVE_PATTERNS ${ARCHIVE_PATTERNS}
  LOCK_TIMEOUT "${LOCK_TIMEOUT}"
  ${_keep_archive_flag}
)
