# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# isaac_add_resource() - Ensure a resource is present; optionally at configure or build time.
#
# Usage:
#   isaac_add_resource(<name>
#     URL <url>
#     EXPECTED_HASH <ALGO=hex>
#     [LINK_DIR <path>]
#     [STRIP_PREFIX <prefix>]
#     [PATCH <patch_file>]
#     [ARCHIVE_PATTERNS <pattern>...]
#     [KEEP_ARCHIVE]
#     [CONFIG_TIME]
#   )
#
#   isaac_add_resource_bundle(<name>
#     [LINK_DIR <path>]
#     RESOURCE <member-name>
#       URL <url>
#       EXPECTED_HASH <ALGO=hex>
#       [STRIP_PREFIX <prefix>]
#       [PATCH <patch_file>]
#       [ARCHIVE_PATTERNS <pattern>...]
#       [KEEP_ARCHIVE]
#     [RESOURCE <member-name> ...]
#   )
#
# Arguments:
# - name: Resource identifier. Used as the cmake target name (build-time mode) and as
#         the cache subdirectory name under ISAAC_RESOURCE_CACHE.
# - URL: Resource location to download.
# - EXPECTED_HASH: Required identity hash in ALGO=hex format (for example SHA256=...).
# - LINK_DIR: Optional directory symlink pointing to the resolved view (content dir,
#             or the STRIP_PREFIX sub-path within it when STRIP_PREFIX is set).
#             Default (build-time): ${PROJECT_SOURCE_DIR}/data/<name>
#             Default (config-time): none (content accessible via <name>_CONTENT_DIR)
# - STRIP_PREFIX: Archive-only sub-path under content/ to expose as the resolved view.
#                 Implemented as a consumer-side path view, not an extraction rename:
#                 different STRIP_PREFIX values on the same hash share one cached
#                 extraction. Fatal if the path does not exist after extraction.
# - PATCH: Archive-only patch file applied after extraction.
# - ARCHIVE_PATTERNS: Archive-only list of patterns for filtered extraction.
# - KEEP_ARCHIVE: If set, keep the downloaded archive after extraction (default: delete).
# - CONFIG_TIME: Run immediately at cmake configure time instead of creating a build target.
#
# Bundle behavior:
# - RESOURCE is a reserved delimiter and cannot be a member name or argument value.
# - Bundle members accept ordinary resource arguments except LINK_DIR and CONFIG_TIME.
# - File members are linked by URL basename at the bundle root. Archive members are linked as
#   directories named after their member targets and expose their resolved STRIP_PREFIX views.
# - Members retain independent ordinary caches. The bundle cache contains member symlinks and
#   generated sibling artifacts at <cache>/<bundle>/<bundle-sha8>/content.
# - The bundle identity is order-independent and changes when any visible member view changes.
#   Generated files at the bundle root are reused only for an unchanged composition; generated
#   files inside an archive mount live in that archive member's independently reusable cache.
# - LINK_DIR defaults to ${PROJECT_SOURCE_DIR}/data/<bundle-name>. Like ordinary resource links, it
#   is managed output and replaces an existing real object. Real objects at member destinations
#   are never replaced.
#
# Project-level variables (set via -D or environment before including this file):
# - ISAAC_RESOURCE_CACHE: Root cache dir. Default: ${PROJECT_SOURCE_DIR}/.cache/resources
#   Cache layout: ${ISAAC_RESOURCE_CACHE}/<name>/<sha8>/content/ (one canonical
#   tree per sha8; recipes recorded in <sha8>/content.recipes).
# - ISAAC_RESOURCE_LOCK_TIMEOUT: Seconds to wait for lock acquisition. Default: 5 minutes.
#
# Outputs:
# - Configuration-time mode (CONFIG_TIME): sets <name>_CONTENT_DIR to the resolved content path in caller's scope.
# - Build-time mode: creates a custom target <name> that ensures the resource is present.
# - LINK_DIR: when provided, creates a symlink LINK_DIR -> content dir.

include(isaac_set)
include(isaac_resource_helpers)

isaac_set(ISAAC_RESOURCE_CACHE
  TYPE PATH
  DEFAULT "${PROJECT_SOURCE_DIR}/.cache/resources"
  DESCRIPTION "Isaac resource cache dir"
)

isaac_set(ISAAC_RESOURCE_LOCK_TIMEOUT
  DEFAULT "300"
  DESCRIPTION "Resource lock timeout in seconds (empty = indefinite)"
)

# Capture the current path at include time for use in the function below.
# Note: not using PROJECT_SOURCE_DIR so that external projects can include these files.
set(_ISAAC_ADD_RESOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "")

function(isaac_add_resource target_name)
  # Parse and validate arguments.
  set(options CONFIG_TIME KEEP_ARCHIVE)
  set(oneValueArgs URL EXPECTED_HASH LINK_DIR STRIP_PREFIX PATCH)
  set(multiValueArgs ARCHIVE_PATTERNS)
  cmake_parse_arguments(RESOURCE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT RESOURCE_URL)
    message(FATAL_ERROR "isaac_add_resource(${target_name}): URL is required.")
  endif()

  set(_cache_dir "${ISAAC_RESOURCE_CACHE}/${target_name}")
  set(_link_dir "${RESOURCE_LINK_DIR}")

  # Path construction.
  # Bundle callers inherit this function-scoped flag to suppress member links.
  if("${_link_dir}" STREQUAL "" AND NOT _ISAAC_ADD_RESOURCE_NO_DEFAULT_LINK
      AND NOT RESOURCE_CONFIG_TIME)
    # Build-time default: symlink under data/.
    set(_link_dir "${PROJECT_SOURCE_DIR}/data/${target_name}")
  endif()

  if(RESOURCE_CONFIG_TIME)
    # Run immediately at configure time.
    set(_keep_archive "")
    if(RESOURCE_KEEP_ARCHIVE)
      set(_keep_archive KEEP_ARCHIVE)
    endif()
    isaac_resource_ensure(
      RESOURCE_LABEL "${target_name}"
      URL "${RESOURCE_URL}"
      CACHE_DIR "${_cache_dir}"
      LINK_DIR "${_link_dir}"
      EXPECTED_HASH "${RESOURCE_EXPECTED_HASH}"
      STRIP_PREFIX "${RESOURCE_STRIP_PREFIX}"
      PATCH_FILE "${RESOURCE_PATCH}"
      ARCHIVE_PATTERNS ${RESOURCE_ARCHIVE_PATTERNS}
      LOCK_TIMEOUT "${ISAAC_RESOURCE_LOCK_TIMEOUT}"
      ${_keep_archive}
      OUTPUT_CONTENT_DIR _content_dir
    )
    # Expose content dir in caller's scope.
    set(${target_name}_CONTENT_DIR "${_content_dir}" PARENT_SCOPE)
  else()
    # Create a build-time custom target.
    # Note: we do not rely on OUTPUT / DEPENDS since we use content-based stamp checks
    # which is stricter than timestamp-based dependencies.
    set(_ensure_args
      -DURL=${RESOURCE_URL}
      -DRESOURCE_LABEL=${target_name}
      -DCACHE_DIR=${_cache_dir}
    )
    if(RESOURCE_KEEP_ARCHIVE)
      list(APPEND _ensure_args -DKEEP_ARCHIVE=TRUE)
    endif()
    if(NOT "${_link_dir}" STREQUAL "")
      list(APPEND _ensure_args -DLINK_DIR=${_link_dir})
    endif()
    if(RESOURCE_EXPECTED_HASH)
      list(APPEND _ensure_args -DEXPECTED_HASH=${RESOURCE_EXPECTED_HASH})
    endif()
    if(RESOURCE_STRIP_PREFIX)
      list(APPEND _ensure_args -DSTRIP_PREFIX=${RESOURCE_STRIP_PREFIX})
    endif()
    if(RESOURCE_PATCH)
      list(APPEND _ensure_args -DPATCH_FILE=${RESOURCE_PATCH})
    endif()
    if(RESOURCE_ARCHIVE_PATTERNS)
      # Join with escaped semicolons so the list survives command-line passthrough
      # as a single -D value. CMake's VERBATIM mode splits on raw semicolons.
      list(JOIN RESOURCE_ARCHIVE_PATTERNS "\;" _patterns_str)
      list(APPEND _ensure_args "-DARCHIVE_PATTERNS=${_patterns_str}")
    endif()
    if(DEFINED ISAAC_RESOURCE_LOCK_TIMEOUT)
      list(APPEND _ensure_args "-DLOCK_TIMEOUT=${ISAAC_RESOURCE_LOCK_TIMEOUT}")
    endif()

    add_custom_target(${target_name}
      COMMAND ${CMAKE_COMMAND}
              ${_ensure_args}
              -P "${_ISAAC_ADD_RESOURCE_DIR}/isaac_resource_ensure_script.cmake"
      # Stream live progress through Ninja's console pool instead of buffering.
      # Tradeoff: resource targets run one at a time, including verification
      # and extraction, delaying the next download. This can increase total
      # setup time, especially when one download cannot use all available bandwidth.
      USES_TERMINAL
      VERBATIM
    )
    if(NOT TARGET isaac_download_resources)
      add_custom_target(isaac_download_resources)
    endif()
    add_dependencies(isaac_download_resources ${target_name})
    _isaac_resource_describe(
      CACHE_DIR "${_cache_dir}"
      TARGET_NAME "${target_name}"
      URL "${RESOURCE_URL}"
      EXPECTED_HASH "${RESOURCE_EXPECTED_HASH}"
      STRIP_PREFIX "${RESOURCE_STRIP_PREFIX}"
      PATCH "${RESOURCE_PATCH}"
      ARCHIVE_PATTERNS ${RESOURCE_ARCHIVE_PATTERNS}
      OUTPUT_CONTENT_DIR _content_dir
      OUTPUT_CONTENT_ID _content_id
      OUTPUT_DESTINATION _destination
      OUTPUT_IS_ARCHIVE _is_archive
    )
    set_target_properties(${target_name} PROPERTIES
      ISAAC_RESOURCE_CONTENT_DIR "${_content_dir}"
      ISAAC_RESOURCE_CONTENT_ID "${_content_id}"
      ISAAC_RESOURCE_DESTINATION "${_destination}"
      ISAAC_RESOURCE_IS_ARCHIVE "${_is_archive}"
    )
  endif()
endfunction()

# isaac_add_resource_bundle() - Group ordinary resources under one content view.
# RESOURCE is a reserved delimiter and cannot be used as a literal member value.
# Optional member values must be omitted rather than passed as empty strings.
# LINK_DIR is managed like an ordinary resource link and replaces an existing real object.
function(isaac_add_resource_bundle bundle_name)
  if(TARGET "${bundle_name}")
    message(FATAL_ERROR
      "isaac_add_resource_bundle(${bundle_name}): target already exists.")
  endif()

  set(_tokens ${ARGN})
  list(LENGTH _tokens _token_count)
  set(_index 0)
  set(_bundle_link_dir "")
  set(_bundle_link_dir_set FALSE)
  set(_member_names "")
  set(_member_targets "")
  set(_destinations "")
  set(_identity_entries "")
  set(_seen_resource FALSE)
  set(_member_options CONFIG_TIME KEEP_ARCHIVE)
  set(_member_one_value_args EXPECTED_HASH LINK_DIR PATCH STRIP_PREFIX URL)
  set(_member_multi_value_args ARCHIVE_PATTERNS)
  set(_member_keywords
    ${_member_options}
    ${_member_one_value_args}
    ${_member_multi_value_args}
  )

  # Child function scopes inherit this flag; it is intentionally not public API.
  set(_ISAAC_ADD_RESOURCE_NO_DEFAULT_LINK TRUE)
  while(_index LESS _token_count)
    list(GET _tokens ${_index} _token)
    if(NOT _seen_resource)
      if(_token STREQUAL "LINK_DIR")
        if(_bundle_link_dir_set)
          message(FATAL_ERROR
            "isaac_add_resource_bundle(${bundle_name}): LINK_DIR may be specified only once.")
        endif()
        math(EXPR _index "${_index} + 1")
        if(NOT _index LESS _token_count)
          message(FATAL_ERROR
            "isaac_add_resource_bundle(${bundle_name}): LINK_DIR requires a directory.")
        endif()
        list(GET _tokens ${_index} _bundle_link_dir)
        if(_bundle_link_dir STREQUAL "RESOURCE")
          message(FATAL_ERROR
            "isaac_add_resource_bundle(${bundle_name}): RESOURCE is a reserved delimiter and cannot be used as LINK_DIR.")
        endif()
        set(_bundle_link_dir_set TRUE)
        math(EXPR _index "${_index} + 1")
        continue()
      elseif(NOT _token STREQUAL "RESOURCE")
        if(_token STREQUAL "CONFIG_TIME")
          message(FATAL_ERROR
            "isaac_add_resource_bundle(${bundle_name}): cannot use CONFIG_TIME.")
        endif()
        message(FATAL_ERROR
          "isaac_add_resource_bundle(${bundle_name}): unknown bundle argument ${_token}.")
      endif()
      set(_seen_resource TRUE)
    endif()

    # Every iteration after bundle-level arguments begins at a RESOURCE delimiter.
    math(EXPR _index "${_index} + 1")
    if(NOT _index LESS _token_count)
      message(FATAL_ERROR
        "isaac_add_resource_bundle(${bundle_name}): RESOURCE is a reserved delimiter and requires a member name.")
    endif()
    list(GET _tokens ${_index} _member_name)
    if(_member_name STREQUAL "RESOURCE")
      message(FATAL_ERROR
        "isaac_add_resource_bundle(${bundle_name}): RESOURCE is a reserved delimiter and cannot be a member name.")
    endif()
    if(_member_name IN_LIST _member_keywords)
      message(FATAL_ERROR
        "isaac_add_resource_bundle(${bundle_name}): RESOURCE requires a member name before ${_member_name}.")
    endif()
    math(EXPR _index "${_index} + 1")

    set(_member_args "")
    while(_index LESS _token_count)
      list(GET _tokens ${_index} _member_token)
      if(_member_token STREQUAL "RESOURCE")
        break()
      endif()
      list(APPEND _member_args "${_member_token}")
      math(EXPR _index "${_index} + 1")
    endwhile()

    list(FIND _member_args LINK_DIR _member_link_index)
    if(NOT _member_link_index EQUAL -1)
      message(FATAL_ERROR
        "isaac_add_resource_bundle(${bundle_name}): member ${_member_name} cannot set LINK_DIR.")
    endif()
    list(FIND _member_args CONFIG_TIME _member_config_index)
    if(NOT _member_config_index EQUAL -1)
      message(FATAL_ERROR
        "isaac_add_resource_bundle(${bundle_name}): member ${_member_name} cannot use CONFIG_TIME.")
    endif()

    cmake_parse_arguments(MEMBER "${_member_options}"
      "${_member_one_value_args}"
      "${_member_multi_value_args}"
      ${_member_args}
    )
    if(MEMBER_UNPARSED_ARGUMENTS)
      message(FATAL_ERROR
        "isaac_add_resource_bundle(${bundle_name}): member ${_member_name} has unknown arguments: ${MEMBER_UNPARSED_ARGUMENTS}.")
    endif()
    if(NOT DEFINED MEMBER_URL OR "${MEMBER_URL}" STREQUAL "")
      message(FATAL_ERROR
        "isaac_add_resource_bundle(${bundle_name}): member ${_member_name} requires URL.")
    endif()
    if(NOT DEFINED MEMBER_EXPECTED_HASH OR "${MEMBER_EXPECTED_HASH}" STREQUAL "")
      message(FATAL_ERROR
        "isaac_add_resource_bundle(${bundle_name}): member ${_member_name} requires EXPECTED_HASH.")
    endif()
    if(TARGET "${_member_name}" OR "${_member_name}" STREQUAL "${bundle_name}"
        OR _member_name IN_LIST _member_names)
      message(FATAL_ERROR
        "isaac_add_resource_bundle(${bundle_name}): duplicate member target ${_member_name}.")
    endif()

    isaac_add_resource("${_member_name}" ${_member_args})
    get_target_property(
      _member_content_dir "${_member_name}" ISAAC_RESOURCE_CONTENT_DIR)
    get_target_property(
      _member_content_id "${_member_name}" ISAAC_RESOURCE_CONTENT_ID)
    get_target_property(
      _member_destination "${_member_name}" ISAAC_RESOURCE_DESTINATION)
    get_target_property(
      _member_is_archive "${_member_name}" ISAAC_RESOURCE_IS_ARCHIVE)
    if(_member_destination IN_LIST _destinations)
      message(FATAL_ERROR
        "isaac_add_resource_bundle(${bundle_name}): duplicate destination ${_member_destination}.")
    endif()

    list(APPEND _member_names "${_member_name}")
    list(APPEND _destinations "${_member_destination}")
    if(_member_is_archive)
      list(APPEND _identity_entries
        "archive|${_member_destination}|${_member_content_id}")
      list(APPEND _member_targets "${_member_content_dir}")
    else()
      list(APPEND _identity_entries
        "file|${_member_destination}|${_member_content_id}")
      list(APPEND _member_targets
        "${_member_content_dir}/${_member_destination}")
    endif()
  endwhile()
  unset(_ISAAC_ADD_RESOURCE_NO_DEFAULT_LINK)

  if(NOT _seen_resource)
    message(FATAL_ERROR
      "isaac_add_resource_bundle(${bundle_name}): requires RESOURCE member clauses.")
  endif()
  if(NOT _bundle_link_dir_set)
    set(_bundle_link_dir "${PROJECT_SOURCE_DIR}/data/${bundle_name}")
  endif()

  list(SORT _identity_entries)
  string(JOIN "\n" _identity_members ${_identity_entries})
  string(SHA256 _bundle_content_id
    "isaac-resource-bundle-v1\n${_identity_members}")
  string(SUBSTRING "${_bundle_content_id}" 0 8 _bundle_sha8)
  set(_bundle_cache_dir
    "${ISAAC_RESOURCE_CACHE}/${bundle_name}/${_bundle_sha8}")
  set(_bundle_content_dir "${_bundle_cache_dir}/content")

  set(_manifest_dir "${CMAKE_CURRENT_BINARY_DIR}/isaac_resource_bundles")
  set(_manifest_file "${_manifest_dir}/${bundle_name}-${_bundle_sha8}.cmake")
  file(MAKE_DIRECTORY "${_manifest_dir}")
  file(WRITE "${_manifest_file}" "set(ISAAC_RESOURCE_BUNDLE_DESTINATIONS\n")
  foreach(_destination IN LISTS _destinations)
    file(APPEND "${_manifest_file}" "  [==[${_destination}]==]\n")
  endforeach()
  file(APPEND "${_manifest_file}" ")\nset(ISAAC_RESOURCE_BUNDLE_TARGETS\n")
  foreach(_target IN LISTS _member_targets)
    file(APPEND "${_manifest_file}" "  [==[${_target}]==]\n")
  endforeach()
  file(APPEND "${_manifest_file}" ")\n")

  add_custom_target("${bundle_name}"
    COMMAND ${CMAKE_COMMAND}
      -DBUNDLE_LABEL=${bundle_name}
      -DCACHE_DIR=${_bundle_cache_dir}
      -DISAAC_ADD_RESOURCE_DIR=${_ISAAC_ADD_RESOURCE_DIR}
      -DLINK_DIR=${_bundle_link_dir}
      -DLOCK_TIMEOUT=${ISAAC_RESOURCE_LOCK_TIMEOUT}
      -DMANIFEST_FILE=${_manifest_file}
      -P "${_ISAAC_ADD_RESOURCE_DIR}/isaac_resource_bundle_ensure_script.cmake"
    VERBATIM
  )
  add_dependencies(isaac_download_resources "${bundle_name}")
  foreach(_member_name IN LISTS _member_names)
    add_dependencies("${bundle_name}" "${_member_name}")
  endforeach()
  set_target_properties("${bundle_name}" PROPERTIES
    ISAAC_RESOURCE_BUNDLE_CONTENT_DIR "${_bundle_content_dir}"
    ISAAC_RESOURCE_BUNDLE_CONTENT_ID "${_bundle_content_id}"
    ISAAC_RESOURCE_BUNDLE_LINK_DIR "${_bundle_link_dir}"
    ISAAC_RESOURCE_BUNDLE_MEMBERS "${_member_names}"
  )
endfunction()
