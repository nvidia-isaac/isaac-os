# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

################################################################################
# Helper functions
################################################################################

# isaac_resource_stamp_matches: Check whether a stamp file matches expected content.
# ARGS (positional)
#  - stamp_file: Stamp path to inspect.
#  - expected_content: Expected stamp content; empty means stamp presence is sufficient.
#  - out_var: Output variable set to TRUE when stamp is valid.
function(isaac_resource_stamp_matches stamp_file expected_content out_var)
  set(_matches FALSE)
  if(EXISTS "${stamp_file}")
    if(DEFINED expected_content AND NOT "${expected_content}" STREQUAL "")
      file(READ "${stamp_file}" _stamp_content)
      string(STRIP "${_stamp_content}" _stamp_content)
      string(STRIP "${expected_content}" _expected_content)
      if("${_stamp_content}" STREQUAL "${_expected_content}")
        set(_matches TRUE)
      endif()
    else()
      set(_matches TRUE)
    endif()
  endif()
  set(${out_var} ${_matches} PARENT_SCOPE)
endfunction()

# isaac_resource_directory_has_content: Check whether a directory exists and is non-empty.
# ARGS (positional)
#  - dir_path: Directory path to inspect.
#  - out_var: Output variable set to TRUE when directory has content.
function(isaac_resource_directory_has_content dir_path out_var)
  set(_has_content FALSE)
  if(IS_DIRECTORY "${dir_path}")
    file(GLOB _dir_contents "${dir_path}/*")
    if(_dir_contents)
      set(_has_content TRUE)
    endif()
  endif()
  set(${out_var} ${_has_content} PARENT_SCOPE)
endfunction()

# isaac_resource_is_archive_url: Detect whether a URL/path points to an archive.
# ARGS (positional)
#  - url: Resource URL or filename.
#  - out_var: Output variable set to TRUE for supported archive extensions.
function(isaac_resource_is_archive_url url out_var)
  set(_is_archive FALSE)
  # Keep this suffix list aligned with CMake archive/compression support.
  # We intentionally detect archive containers (including tar+compression),
  # not raw single-file compression streams.
  if(url MATCHES "\\.(7z|tar\\.gz|tgz|tar\\.bz2|tbz2|tar\\.xz|txz|tar\\.zst|tzst|tar|zip)$")
    set(_is_archive TRUE)
  endif()
  set(${out_var} ${_is_archive} PARENT_SCOPE)
endfunction()

# isaac_resource_canonicalize_patterns: Map ARCHIVE_PATTERNS to a recipe token.
# Returns "*" when no patterns are supplied (full extraction). For non-empty
# lists, sorts and newline-joins the patterns, then returns the first 8 hex
# chars of the SHA256. Newline is used as delimiter instead of semicolon
# because patterns may legally contain ';' but not newlines.
# ARGS (positional)
#  - out_var: Output variable set in caller's scope.
#  - ARGN: Zero or more pattern strings.
function(isaac_resource_canonicalize_patterns out_var)
  set(_patterns ${ARGN})
  if(NOT _patterns)
    set(${out_var} "*" PARENT_SCOPE)
    return()
  endif()
  list(SORT _patterns)
  string(JOIN "\n" _joined ${_patterns})
  string(SHA256 _hex "${_joined}")
  string(SUBSTRING "${_hex}" 0 8 _short)
  set(${out_var} "${_short}" PARENT_SCOPE)
endfunction()

# isaac_resource_recipes_contains: Check whether the recipes stamp covers a recipe.
# Covers = the recipe token or `*` appears on any line of stamp_file.
# ARGS (positional)
#  - stamp_file: Path to the recipes stamp (may not exist).
#  - recipe: Recipe token to test for (e.g. "*" or an 8-hex string).
#  - out_var: Output variable set to TRUE when the stamp covers the recipe.
function(isaac_resource_recipes_contains stamp_file recipe out_var)
  set(_covered FALSE)
  if(EXISTS "${stamp_file}")
    file(STRINGS "${stamp_file}" _lines)
    foreach(_line IN LISTS _lines)
      string(STRIP "${_line}" _line)
      if("${_line}" STREQUAL "*" OR "${_line}" STREQUAL "${recipe}")
        set(_covered TRUE)
        break()
      endif()
    endforeach()
  endif()
  set(${out_var} ${_covered} PARENT_SCOPE)
endfunction()

# isaac_resource_recipes_append: Append a recipe token to the recipes stamp.
# Idempotent: if the token (or `*`) is already present, the stamp is left alone.
# ARGS (positional)
#  - stamp_file: Path to the recipes stamp (will be created if missing).
#  - recipe: Recipe token to append.
function(isaac_resource_recipes_append stamp_file recipe)
  isaac_resource_recipes_contains("${stamp_file}" "${recipe}" _already)
  if(_already)
    return()
  endif()
  get_filename_component(_parent "${stamp_file}" DIRECTORY)
  if(NOT IS_DIRECTORY "${_parent}")
    file(MAKE_DIRECTORY "${_parent}")
  endif()
  file(APPEND "${stamp_file}" "${recipe}\n")
endfunction()

# isaac_resource_parse_expected_hash: Parse an expected hash.
# Sets <prefix>_ALGORITHM and <prefix>_HEX.
function(isaac_resource_parse_expected_hash out_prefix expected_hash)
  set(_algorithm "SHA256")
  if(expected_hash MATCHES "^([A-Za-z0-9]+)=(.+)$")
    set(_algorithm "${CMAKE_MATCH_1}")
    set(_hex "${CMAKE_MATCH_2}")
  else()
    set(_hex "${expected_hash}")
  endif()
  string(TOUPPER "${_algorithm}" _algorithm)
  string(TOLOWER "${_hex}" _hex)

  set(${out_prefix}_ALGORITHM "${_algorithm}" PARENT_SCOPE)
  set(${out_prefix}_HEX "${_hex}" PARENT_SCOPE)
endfunction()

# isaac_resource_resolve_cache_paths: Derive a resource's versioned cache paths.
# ARGS (positional)
#  - cache_dir: Root cache directory for one resource.
#  - expected_hash: Normalized resource identity hash (ALGO=HEX or HEX).
#  - strip_prefix: Optional content subdirectory exposed to consumers.
#  - out_real_cache: Output variable for CACHE_DIR/<sha8>.
#  - out_content_dir: Output variable for CACHE_DIR/<sha8>/content.
#  - out_view_dir: Output variable for content/<strip_prefix>, or content when empty.
function(isaac_resource_resolve_cache_paths
    cache_dir expected_hash strip_prefix out_real_cache out_content_dir out_view_dir)
  isaac_resource_parse_expected_hash(_hash "${expected_hash}")
  string(SUBSTRING "${_hash_HEX}" 0 8 _sha8)

  set(_real_cache "${cache_dir}/${_sha8}")
  set(_content_dir "${_real_cache}/content")
  set(_view_dir "${_content_dir}")
  if(NOT "${strip_prefix}" STREQUAL "")
    set(_view_dir "${_content_dir}/${strip_prefix}")
  endif()

  set(${out_real_cache} "${_real_cache}" PARENT_SCOPE)
  set(${out_content_dir} "${_content_dir}" PARENT_SCOPE)
  set(${out_view_dir} "${_view_dir}" PARENT_SCOPE)
endfunction()

# Internal: describe a cache-backed content view without ensuring it.
function(_isaac_resource_describe)
  cmake_parse_arguments(DESCRIBE ""
    "CACHE_DIR;TARGET_NAME;URL;EXPECTED_HASH;STRIP_PREFIX;PATCH;OUTPUT_CONTENT_DIR;OUTPUT_CONTENT_ID;OUTPUT_DESTINATION;OUTPUT_IS_ARCHIVE"
    "ARCHIVE_PATTERNS"
    ${ARGN}
  )
  if(DESCRIBE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "_isaac_resource_describe: unknown arguments: ${DESCRIBE_UNPARSED_ARGUMENTS}")
  endif()
  foreach(_required CACHE_DIR TARGET_NAME URL EXPECTED_HASH)
    if(NOT DEFINED DESCRIBE_${_required}
        OR "${DESCRIBE_${_required}}" STREQUAL "")
      message(FATAL_ERROR
        "_isaac_resource_describe(${DESCRIBE_TARGET_NAME}): ${_required} is required.")
    endif()
  endforeach()
  if("${DESCRIBE_OUTPUT_CONTENT_DIR}" STREQUAL ""
      AND "${DESCRIBE_OUTPUT_CONTENT_ID}" STREQUAL ""
      AND "${DESCRIBE_OUTPUT_DESTINATION}" STREQUAL ""
      AND "${DESCRIBE_OUTPUT_IS_ARCHIVE}" STREQUAL "")
    message(FATAL_ERROR
      "_isaac_resource_describe(${DESCRIBE_TARGET_NAME}): requires an OUTPUT argument.")
  endif()

  string(STRIP "${DESCRIBE_EXPECTED_HASH}" _expected_hash)
  isaac_resource_parse_expected_hash(_hash "${_expected_hash}")
  set(_canonical_hash "${_hash_ALGORITHM}=${_hash_HEX}")
  isaac_resource_resolve_cache_paths(
    "${DESCRIBE_CACHE_DIR}" "${_expected_hash}" "${DESCRIBE_STRIP_PREFIX}"
    _unused_real_cache _unused_content_dir _view_dir)
  get_filename_component(_filename "${DESCRIBE_URL}" NAME)
  isaac_resource_is_archive_url("${DESCRIBE_URL}" _is_archive)
  if(_is_archive)
    set(_destination "${DESCRIBE_TARGET_NAME}")
    set(_patterns ${DESCRIBE_ARCHIVE_PATTERNS})
    if(_patterns)
      list(SORT _patterns)
      string(JOIN "\n" _patterns_text ${_patterns})
      string(SHA256 _patterns_id "${_patterns_text}")
    else()
      set(_patterns_id "*")
    endif()

    if(NOT "${DESCRIBE_PATCH}" STREQUAL "")
      if(NOT EXISTS "${DESCRIBE_PATCH}")
        message(FATAL_ERROR
          "_isaac_resource_describe(${DESCRIBE_TARGET_NAME}): PATCH does not exist: "
          "${DESCRIBE_PATCH}")
      endif()
      file(SHA256 "${DESCRIBE_PATCH}" _patch_id)
    else()
      set(_patch_id "none")
    endif()

    string(CONCAT _identity_input
      "kind=archive\nhash=${_canonical_hash}\npatterns=${_patterns_id}\n"
      "strip=${DESCRIBE_STRIP_PREFIX}\npatch=${_patch_id}")
  else()
    set(_destination "${_filename}")
    set(_identity_input
      "kind=file\nhash=${_canonical_hash}\nfilename=${_filename}")
  endif()
  string(SHA256 _content_id "${_identity_input}")

  if(NOT "${DESCRIBE_OUTPUT_CONTENT_DIR}" STREQUAL "")
    set(${DESCRIBE_OUTPUT_CONTENT_DIR} "${_view_dir}" PARENT_SCOPE)
  endif()
  if(NOT "${DESCRIBE_OUTPUT_CONTENT_ID}" STREQUAL "")
    set(${DESCRIBE_OUTPUT_CONTENT_ID} "${_content_id}" PARENT_SCOPE)
  endif()
  if(NOT "${DESCRIBE_OUTPUT_DESTINATION}" STREQUAL "")
    set(${DESCRIBE_OUTPUT_DESTINATION} "${_destination}" PARENT_SCOPE)
  endif()
  if(NOT "${DESCRIBE_OUTPUT_IS_ARCHIVE}" STREQUAL "")
    set(${DESCRIBE_OUTPUT_IS_ARCHIVE} "${_is_archive}" PARENT_SCOPE)
  endif()
endfunction()

# isaac_resource_file_state_is_valid: Check file output and verification stamp validity.
# ARGS (positional)
#  - output_file: Expected file output path.
#  - verified_file: Verification stamp path.
#  - expected_hash: Expected hash stored in verification stamp.
#  - out_var: Output variable set to TRUE when file state is valid.
function(isaac_resource_file_state_is_valid output_file verified_file expected_hash out_var)
  set(_valid FALSE)
  if(EXISTS "${output_file}")
    isaac_resource_stamp_matches("${verified_file}" "${expected_hash}" _stamp_valid)
    if(_stamp_valid)
      set(_valid TRUE)
    endif()
  endif()
  set(${out_var} ${_valid} PARENT_SCOPE)
endfunction()

################################################################################
# Operation functions (download, verify, extract)
################################################################################

# isaac_resource_download: Download a file if it is missing.
# ARGS (named)
#  - URL: Source URL to download.
#  - DEST_FILE: Absolute destination file path.
#  - EXPECTED_HASH: Identity used to invalidate partial downloads when the resource changes.
function(isaac_resource_download)
  cmake_parse_arguments(ARG "" "URL;DEST_FILE;EXPECTED_HASH" "" ${ARGN})
  if(NOT DEFINED ARG_URL OR "${ARG_URL}" STREQUAL "")
    message(FATAL_ERROR "URL is not defined.")
  endif()
  if(NOT DEFINED ARG_DEST_FILE OR "${ARG_DEST_FILE}" STREQUAL "")
    message(FATAL_ERROR "DEST_FILE is not defined.")
  endif()

  if(EXISTS "${ARG_DEST_FILE}")
    message(STATUS "Resource already exists: ${ARG_DEST_FILE}. Skipping download.")
    return()
  endif()

  get_filename_component(_dest_dir "${ARG_DEST_FILE}" DIRECTORY)
  if(NOT IS_DIRECTORY "${_dest_dir}")
    file(MAKE_DIRECTORY "${_dest_dir}")
  endif()

  # Preserve partial HTTP downloads. Other schemes retain CMake transport support.
  if(ARG_URL MATCHES "^https?://")
    find_program(_isaac_resource_python NAMES python3 REQUIRED)
    execute_process(
      COMMAND "${_isaac_resource_python}"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/isaac_resource_download.py"
        "${ARG_URL}" "${ARG_DEST_FILE}" "${ARG_EXPECTED_HASH}"
      RESULT_VARIABLE _download_result
    )
    if(NOT "${_download_result}" STREQUAL "0")
      message(FATAL_ERROR
        "Unable to download ${ARG_DEST_FILE}. Partial bytes are retained; rerun the same command to continue.")
    endif()
    return()
  endif()

  set(_max_attempts 3)
  set(_retry_delay_sec 5)
  set(_dl_success FALSE)

  foreach(_attempt RANGE 1 ${_max_attempts})
    if(_attempt GREATER 1)
      message(STATUS "Retry attempt ${_attempt} of ${_max_attempts} after ${_retry_delay_sec} seconds...")
      execute_process(COMMAND ${CMAKE_COMMAND} -E sleep ${_retry_delay_sec})
    else()
      message(STATUS "Downloading ${ARG_URL} -> ${ARG_DEST_FILE}.")
    endif()

    file(DOWNLOAD
      "${ARG_URL}"
      "${ARG_DEST_FILE}"
      SHOW_PROGRESS
      TLS_VERIFY ON
      STATUS _dl_status
    )

    list(GET _dl_status 0 _dl_code)
    list(GET _dl_status 1 _dl_msg)
    if(_dl_code EQUAL 0)
      set(_dl_success TRUE)
      break()
    else()
      message(WARNING "Download attempt ${_attempt} failed: ${_dl_msg}.")
      if(EXISTS "${ARG_DEST_FILE}")
        file(REMOVE "${ARG_DEST_FILE}")
      endif()
    endif()
  endforeach()

  if(NOT _dl_success)
    message(FATAL_ERROR "Download failed after ${_max_attempts} attempts: ${_dl_msg}.")
  endif()

  message(STATUS "Downloaded: ${ARG_DEST_FILE}.")
endfunction()

# isaac_resource_verify: Verify content, then optionally publish it and write a stamp.
# ARGS (named)
#  - DEST_FILE: File path to verify.
#  - EXPECTED_HASH: Expected hash string (ALGO=HEX or HEX).
#  - OUTPUT_VALID: Optional result variable; return FALSE on mismatch instead of failing.
#  - PUBLISH_FILE: Optional final path to rename the verified file to.
#  - VERIFIED_FILE: Optional verification stamp path to write after publication.
function(isaac_resource_verify)
  cmake_parse_arguments(ARG "" "DEST_FILE;EXPECTED_HASH;OUTPUT_VALID;PUBLISH_FILE;VERIFIED_FILE" "" ${ARGN})
  if(NOT DEFINED ARG_DEST_FILE OR "${ARG_DEST_FILE}" STREQUAL "")
    message(FATAL_ERROR "DEST_FILE is not defined.")
  endif()
  if(NOT EXISTS "${ARG_DEST_FILE}")
    message(FATAL_ERROR "File does not exist: ${ARG_DEST_FILE}.")
  endif()
  if(NOT DEFINED ARG_EXPECTED_HASH OR "${ARG_EXPECTED_HASH}" STREQUAL "")
    message(FATAL_ERROR "EXPECTED_HASH is required for verification.")
  endif()

  isaac_resource_parse_expected_hash(_hash "${ARG_EXPECTED_HASH}")

  message(STATUS "Verifying resource: ${ARG_DEST_FILE}.")
  file(${_hash_ALGORITHM} "${ARG_DEST_FILE}" _actual_hash)
  if(NOT "${ARG_OUTPUT_VALID}" STREQUAL "")
    set(${ARG_OUTPUT_VALID} FALSE PARENT_SCOPE)
  endif()
  if(NOT "${_actual_hash}" STREQUAL "${_hash_HEX}")
    if(NOT "${ARG_OUTPUT_VALID}" STREQUAL "")
      message(WARNING "Checksum mismatch for ${ARG_DEST_FILE}; discarding the candidate.")
      return()
    endif()
    message(FATAL_ERROR "Checksum mismatch:\n  - ${_actual_hash} (${ARG_DEST_FILE})\n  - ${_hash_HEX} (Expected).")
  endif()

  if(NOT "${ARG_OUTPUT_VALID}" STREQUAL "")
    set(${ARG_OUTPUT_VALID} TRUE PARENT_SCOPE)
  endif()
  if(NOT "${ARG_PUBLISH_FILE}" STREQUAL "")
    cmake_path(ABSOLUTE_PATH ARG_PUBLISH_FILE OUTPUT_VARIABLE _publish_file)
    get_filename_component(_publish_dir "${_publish_file}" DIRECTORY)
    file(MAKE_DIRECTORY "${_publish_dir}")
    file(RENAME "${ARG_DEST_FILE}" "${_publish_file}")
  endif()
  if("${ARG_VERIFIED_FILE}" STREQUAL "")
    return()
  endif()

  get_filename_component(_verified_dir "${ARG_VERIFIED_FILE}" DIRECTORY)
  if(NOT IS_DIRECTORY "${_verified_dir}")
    file(MAKE_DIRECTORY "${_verified_dir}")
  endif()
  string(STRIP "${ARG_EXPECTED_HASH}" _expected_hash_stripped)
  file(WRITE "${ARG_VERIFIED_FILE}" "${_expected_hash_stripped}\n")
endfunction()

# isaac_resource_extract: Extract archive content additively into EXTRACT_DIR.
# Files already present at colliding paths are overwritten with identical content
# from the archive; files not matching the patterns (or outside them) are left
# alone. EXTRACT_DIR is created if missing. Caller is responsible for recording
# the extraction (e.g. in a recipes stamp).
# ARGS (named)
#  - ARCHIVE_FILE: Local archive path.
#  - EXTRACT_DIR: Destination extraction directory.
#  - PATCH_FILE: Optional patch to apply after extraction.
#  - ARCHIVE_PATTERNS: Optional archive patterns for filtered extraction.
function(isaac_resource_extract)
  cmake_parse_arguments(
    ARG
    ""
    "ARCHIVE_FILE;EXTRACT_DIR;PATCH_FILE"
    "ARCHIVE_PATTERNS"
    ${ARGN}
  )
  if(NOT DEFINED ARG_ARCHIVE_FILE OR "${ARG_ARCHIVE_FILE}" STREQUAL "")
    message(FATAL_ERROR "ARCHIVE_FILE is not defined.")
  endif()
  if(NOT DEFINED ARG_EXTRACT_DIR OR "${ARG_EXTRACT_DIR}" STREQUAL "")
    message(FATAL_ERROR "EXTRACT_DIR is not defined.")
  endif()
  if(NOT EXISTS "${ARG_ARCHIVE_FILE}")
    message(FATAL_ERROR "Archive file does not exist: ${ARG_ARCHIVE_FILE}")
  endif()

  get_filename_component(_archive_name "${ARG_ARCHIVE_FILE}" NAME)
  message(STATUS "Extracting ${_archive_name} -> ${ARG_EXTRACT_DIR}")

  if(NOT IS_DIRECTORY "${ARG_EXTRACT_DIR}")
    file(MAKE_DIRECTORY "${ARG_EXTRACT_DIR}")
  endif()

  if(ARG_ARCHIVE_PATTERNS)
    file(ARCHIVE_EXTRACT
      INPUT "${ARG_ARCHIVE_FILE}"
      DESTINATION "${ARG_EXTRACT_DIR}"
      PATTERNS ${ARG_ARCHIVE_PATTERNS}
      VERBOSE
    )
  else()
    file(ARCHIVE_EXTRACT
      INPUT "${ARG_ARCHIVE_FILE}"
      DESTINATION "${ARG_EXTRACT_DIR}"
      VERBOSE
    )
  endif()

  if(DEFINED ARG_PATCH_FILE AND NOT "${ARG_PATCH_FILE}" STREQUAL "")
    if(NOT EXISTS "${ARG_PATCH_FILE}")
      message(FATAL_ERROR "Patch file does not exist: ${ARG_PATCH_FILE}")
    endif()
    message(STATUS "Applying patch: ${ARG_PATCH_FILE}")
    execute_process(
      COMMAND patch -p0 -i "${ARG_PATCH_FILE}"
      WORKING_DIRECTORY "${ARG_EXTRACT_DIR}"
      RESULT_VARIABLE _patch_result
      ERROR_VARIABLE _patch_error
      OUTPUT_VARIABLE _patch_output
    )
    if(NOT _patch_result EQUAL 0)
      message(FATAL_ERROR "Failed to apply patch ${ARG_PATCH_FILE}\nError output: ${_patch_error}\n${_patch_output}")
    else()
      message(STATUS "Patch applied successfully.")
    endif()
  endif()

  message(STATUS "Extracted: ${ARG_EXTRACT_DIR}")
endfunction()

# isaac_resource_acquire_lock: Acquire an advisory per-version lock.
# IMPORTANT: This must be a macro (not a function). CMake macros expand inline and share
# the calling function's scope, so GUARD FUNCTION releases the lock when the CALLING
# function returns - not at the end of this macro. Using a function here would release
# the lock immediately on return, before any work is done.
# Advisory locks (fcntl F_SETLKW) are released by the OS on process exit or crash;
# stale locks are not possible. NFS advisory locks are unreliable (local filesystem only).
# ARGS (positional)
#  - lock_path: path of the .lock file (parent dir must be creatable)
#  - lock_timeout: timeout in seconds to wait for lock acquisition, or empty for indefinite
macro(isaac_resource_acquire_lock lock_path lock_timeout)
  get_filename_component(_ral_parent "${lock_path}" DIRECTORY)
  if(NOT IS_DIRECTORY "${_ral_parent}")
    file(MAKE_DIRECTORY "${_ral_parent}")
  endif()
  message(DEBUG "Acquiring resource lock: ${lock_path}")
  set(_ral_args GUARD FUNCTION)
  if(NOT "${lock_timeout}" STREQUAL "")
    list(APPEND _ral_args TIMEOUT "${lock_timeout}")
  endif()
  file(LOCK "${lock_path}" ${_ral_args})
  unset(_ral_args)
  unset(_ral_parent)
endmacro()

# isaac_resource_ensure_link: Create or update a relative symlink.
# Handles all edge cases idempotently: missing (create), correct (no-op),
# stale wrong-target (re-link), existing real object (replace with symlink).
# ARGS (positional)
#  - link: Path at which to create the symlink.
#  - link_target: Path the symlink should point to.
function(isaac_resource_ensure_link link link_target)
  if(NOT DEFINED link OR "${link}" STREQUAL "")
    message(FATAL_ERROR "isaac_resource_ensure_link: link is required.")
  endif()
  if(NOT DEFINED link_target OR "${link_target}" STREQUAL "")
    message(FATAL_ERROR "isaac_resource_ensure_link: link_target is required.")
  endif()

  get_filename_component(_parent "${link}" DIRECTORY)
  if(NOT IS_DIRECTORY "${_parent}")
    file(MAKE_DIRECTORY "${_parent}")
  endif()

  # Use a relative target so the symlink works regardless of mount point (e.g. in Docker).
  file(RELATIVE_PATH _rel_target "${_parent}" "${link_target}")

  if(IS_SYMLINK "${link}")
    file(READ_SYMLINK "${link}" _current_target)
    if("${_current_target}" STREQUAL "${_rel_target}")
      return()  # Already correct, no-op.
    endif()
    message(STATUS "Updating stale symlink: ${link}")
    file(REMOVE "${link}")
  elseif(IS_DIRECTORY "${link}")
    message(STATUS "Replacing real directory with symlink: ${link}")
    file(REMOVE_RECURSE "${link}")
  elseif(EXISTS "${link}")
    file(REMOVE "${link}")
  endif()

  file(CREATE_LINK "${_rel_target}" "${link}" SYMBOLIC)
endfunction()

# isaac_resource_ensure_downloaded: Ensure file presence.
# ARGS (named)
#  - RESOURCE_LABEL: Friendly resource label for logs and errors.
#  - URL: Source URL to download.
#  - DEST_FILE: Absolute destination file path.
#  - EXPECTED_HASH: Identity used to invalidate partial downloads when the resource changes.
function(isaac_resource_ensure_downloaded)
  cmake_parse_arguments(ARG "" "RESOURCE_LABEL;URL;DEST_FILE;EXPECTED_HASH" "" ${ARGN})
  if(NOT DEFINED ARG_RESOURCE_LABEL OR "${ARG_RESOURCE_LABEL}" STREQUAL "")
    message(FATAL_ERROR "RESOURCE_LABEL is not defined.")
  endif()
  isaac_resource_download(
    URL "${ARG_URL}"
    DEST_FILE "${ARG_DEST_FILE}"
    EXPECTED_HASH "${ARG_EXPECTED_HASH}"
  )
endfunction()

# isaac_resource_ensure_verified: Ensure a file is downloaded and hash-verified.
# ARGS (named)
#  - RESOURCE_LABEL: Friendly resource label for logs and errors.
#  - URL: Source URL to download when refresh is needed.
#  - DEST_FILE: File path to verify.
#  - EXPECTED_HASH: Expected identity hash (ALGO=HEX).
#  - VERIFIED_FILE: Verification stamp path.
function(isaac_resource_ensure_verified)
  cmake_parse_arguments(ARG "" "RESOURCE_LABEL;URL;DEST_FILE;EXPECTED_HASH;VERIFIED_FILE" "" ${ARGN})
  isaac_resource_stamp_matches("${ARG_VERIFIED_FILE}" "${ARG_EXPECTED_HASH}" _stamp_valid)

  # Trust stamp content when output exists.
  if(EXISTS "${ARG_DEST_FILE}" AND _stamp_valid)
    message(DEBUG "Resource ${ARG_RESOURCE_LABEL} is already verified: ${ARG_DEST_FILE}.")
    return()
  endif()

  # Catch stale state and clean up.
  if(EXISTS "${ARG_DEST_FILE}" AND NOT _stamp_valid)
    message(STATUS "Resource ${ARG_RESOURCE_LABEL} stamp mismatch; refreshing ${ARG_DEST_FILE}.")
    file(REMOVE "${ARG_DEST_FILE}")
    file(REMOVE "${ARG_VERIFIED_FILE}")
  elseif(NOT EXISTS "${ARG_DEST_FILE}" AND EXISTS "${ARG_VERIFIED_FILE}")
    message(STATUS "Resource ${ARG_RESOURCE_LABEL} has stale verification stamp: ${ARG_VERIFIED_FILE}.")
    file(REMOVE "${ARG_VERIFIED_FILE}")
  endif()

  # Transport produces an unpublished candidate. Verification, publication, and
  # stamping remain in this parent process under the resource lock.
  get_filename_component(_candidate_dir "${ARG_VERIFIED_FILE}" DIRECTORY)
  get_filename_component(_filename "${ARG_DEST_FILE}" NAME)
  cmake_path(APPEND _candidate_dir "${_filename}.tmp" OUTPUT_VARIABLE _candidate)
  foreach(_attempt RANGE 1 3)
    isaac_resource_ensure_downloaded(
      RESOURCE_LABEL "${ARG_RESOURCE_LABEL}"
      URL "${ARG_URL}"
      DEST_FILE "${_candidate}"
      EXPECTED_HASH "${ARG_EXPECTED_HASH}"
    )
    isaac_resource_verify(
      DEST_FILE "${_candidate}"
      EXPECTED_HASH "${ARG_EXPECTED_HASH}"
      OUTPUT_VALID _valid
      PUBLISH_FILE "${ARG_DEST_FILE}"
      VERIFIED_FILE "${ARG_VERIFIED_FILE}"
    )
    if(_valid)
      return()
    endif()
    file(REMOVE "${_candidate}")
  endforeach()
  message(FATAL_ERROR "Checksum mismatch after 3 attempts for ${ARG_RESOURCE_LABEL}.")
endfunction()

# isaac_resource_ensure: Unified entry point for ensuring a resource is present and valid.
# Auto-detects archive vs file mode from the URL extension. All paths are explicit -
# callers are responsible for constructing CACHE_DIR.
#
# Cache layout:
#   CACHE_DIR/<sha8>/                    (sha8 = first 8 hex of EXPECTED_HASH)
#     .lock                              per-sha8 advisory lock
#     <archive>                          archive file (kept only with KEEP_ARCHIVE)
#     <archive>.hash.verified            per-sha8 hash verification stamp
#     content/                         single canonical extraction tree, grown
#                                          additively as new ARCHIVE_PATTERNS
#                                          recipes are requested
#     content.recipes                  one recipe token per line (see below)
#
# Recipe tokens (in content.recipes):
#   `*`              full extraction was performed; covers any patterns request.
#   <8-hex>          canonical SHA-8 of a sorted, newline-joined patterns list.
# A new caller whose recipe (`*` or 8-hex) is already listed is a cache hit;
# otherwise we extract additively with that caller's patterns and append its
# token. `*` in the stamp means every later request is a cache hit.
#
# Content is extracted with the archive's literal layout (no implicit strip-prefix
# or single-root flattening at extraction time).
#
# STRIP_PREFIX is a *view*, not a cache key: when set, OUTPUT_CONTENT_DIR and the
# LINK_DIR symlink resolves to `content/<STRIP_PREFIX>/`. Callers with different
# STRIP_PREFIX values share one physical extraction. Fatal if STRIP_PREFIX names
# a path that does not exist after extraction.
#
# Concurrency: acquires a per-version advisory lock (CACHE_DIR/<sha8>/.lock) before
# any work. Uses isaac_resource_acquire_lock (a macro) so the lock scope matches
# this function's lifetime.
#
# ARGS (named)
#  - RESOURCE_LABEL: Friendly resource label for logs and errors.
#  - URL: Source URL to download.
#  - CACHE_DIR: Root cache dir for this resource (<sha8> subdir appended internally).
#  - EXPECTED_HASH: Required identity hash (ALGO=HEX).
#  - LINK_DIR: Optional directory at which to link the resolved view.
#  - STRIP_PREFIX: Optional sub-path under content/ to expose as the resolved view.
#  - PATCH_FILE: Optional patch to apply after extraction (archive mode).
#  - ARCHIVE_PATTERNS: Optional archive patterns for filtered extraction (archive mode).
#  - KEEP_ARCHIVE: Optional flag, keep archive after extraction (default: delete).
#  - LOCK_TIMEOUT: Optional timeout in seconds for lock acquisition (empty = indefinite).
#  - OUTPUT_CONTENT_DIR: Optional variable name to set in caller's scope with the
#    resolved view path (content/<STRIP_PREFIX>/, or content/ when STRIP_PREFIX
#    is unset).
function(isaac_resource_ensure)
  cmake_parse_arguments(ARG "KEEP_ARCHIVE"
    "RESOURCE_LABEL;URL;CACHE_DIR;LINK_DIR;EXPECTED_HASH;STRIP_PREFIX;PATCH_FILE;LOCK_TIMEOUT;OUTPUT_CONTENT_DIR"
    "ARCHIVE_PATTERNS" ${ARGN})
  if(NOT DEFINED ARG_URL OR "${ARG_URL}" STREQUAL "")
    message(FATAL_ERROR "isaac_resource_ensure: URL is required.")
  endif()
  if(NOT DEFINED ARG_CACHE_DIR OR "${ARG_CACHE_DIR}" STREQUAL "")
    message(FATAL_ERROR "isaac_resource_ensure: CACHE_DIR is required.")
  endif()
  if(NOT DEFINED ARG_EXPECTED_HASH OR "${ARG_EXPECTED_HASH}" STREQUAL "")
    message(FATAL_ERROR "isaac_resource_ensure: EXPECTED_HASH is required.")
  endif()

  isaac_resource_is_archive_url("${ARG_URL}" _is_archive)
  string(STRIP "${ARG_EXPECTED_HASH}" _expected_hash)
  get_filename_component(_url_filename "${ARG_URL}" NAME)

  isaac_resource_resolve_cache_paths(
    "${ARG_CACHE_DIR}" "${_expected_hash}" "${ARG_STRIP_PREFIX}"
    _real_cache _content_dir _view_dir)

  # Per-version advisory lock.
  isaac_resource_acquire_lock("${_real_cache}/.lock" "${ARG_LOCK_TIMEOUT}")

  set(_verified_stamp "${_real_cache}/${_url_filename}.hash.verified")
  set(_recipes_stamp "${_real_cache}/content.recipes")

  if(_is_archive)
    set(_archive_file "${_real_cache}/${_url_filename}")
    isaac_resource_canonicalize_patterns(_recipe ${ARG_ARCHIVE_PATTERNS})
    isaac_resource_recipes_contains("${_recipes_stamp}" "${_recipe}" _recipe_covered)
    isaac_resource_directory_has_content("${_view_dir}" _view_has_content)

    if(_recipe_covered AND _view_has_content)
      message(DEBUG
        "Resource ${ARG_RESOURCE_LABEL}: recipe '${_recipe}' already extracted at ${_content_dir}.")
    else()
      # Need to (re-)extract. Ensure the archive is downloaded + verified.
      isaac_resource_ensure_verified(
        RESOURCE_LABEL "${ARG_RESOURCE_LABEL}"
        URL "${ARG_URL}"
        DEST_FILE "${_archive_file}"
        EXPECTED_HASH "${_expected_hash}"
        VERIFIED_FILE "${_verified_stamp}"
      )
      isaac_resource_extract(
        ARCHIVE_FILE "${_archive_file}"
        EXTRACT_DIR "${_content_dir}"
        PATCH_FILE "${ARG_PATCH_FILE}"
        ARCHIVE_PATTERNS ${ARG_ARCHIVE_PATTERNS}
      )
      isaac_resource_recipes_append("${_recipes_stamp}" "${_recipe}")

      # Delete archive after extraction unless KEEP_ARCHIVE is set.
      if(NOT ARG_KEEP_ARCHIVE AND EXISTS "${_archive_file}")
        file(REMOVE "${_archive_file}")
        message(STATUS "Removed archive: ${_archive_file}")
      endif()
    endif()
  else()
    # Non-archive (file) mode: one file lives directly under content/.
    set(_output_file "${_content_dir}/${_url_filename}")
    isaac_resource_ensure_verified(
      RESOURCE_LABEL "${ARG_RESOURCE_LABEL}"
      URL "${ARG_URL}"
      DEST_FILE "${_output_file}"
      EXPECTED_HASH "${_expected_hash}"
      VERIFIED_FILE "${_verified_stamp}"
    )
  endif()

  # After extraction: validate the STRIP_PREFIX view exists.
  if(DEFINED ARG_STRIP_PREFIX AND NOT "${ARG_STRIP_PREFIX}" STREQUAL ""
      AND NOT EXISTS "${_view_dir}")
    message(FATAL_ERROR
      "isaac_resource_ensure: STRIP_PREFIX path does not exist after extraction: ${_view_dir}")
  endif()

  # Expose resolved view to caller if requested.
  if(DEFINED ARG_OUTPUT_CONTENT_DIR AND NOT "${ARG_OUTPUT_CONTENT_DIR}" STREQUAL "")
    set("${ARG_OUTPUT_CONTENT_DIR}" "${_view_dir}" PARENT_SCOPE)
  endif()

  # Create or update LINK_DIR symlink pointing to the resolved view.
  if(DEFINED ARG_LINK_DIR AND NOT "${ARG_LINK_DIR}" STREQUAL "")
    isaac_resource_ensure_link("${ARG_LINK_DIR}" "${_view_dir}")
  endif()
endfunction()
