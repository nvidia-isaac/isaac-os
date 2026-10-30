# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

include(isaac_resource_helpers)

function(isaac_resource_bundle_ensure)
  cmake_parse_arguments(ARG ""
    "BUNDLE_LABEL;CACHE_DIR;LINK_DIR;LOCK_TIMEOUT;MANIFEST_FILE"
    ""
    ${ARGN}
  )
  foreach(_required BUNDLE_LABEL CACHE_DIR LINK_DIR MANIFEST_FILE)
    if(NOT DEFINED ARG_${_required} OR "${ARG_${_required}}" STREQUAL "")
      message(FATAL_ERROR
        "isaac_resource_bundle_ensure: ${_required} is required.")
    endif()
  endforeach()
  if(NOT EXISTS "${ARG_MANIFEST_FILE}")
    message(FATAL_ERROR
      "isaac_resource_bundle_ensure: manifest does not exist: ${ARG_MANIFEST_FILE}")
  endif()

  include("${ARG_MANIFEST_FILE}")
  list(LENGTH ISAAC_RESOURCE_BUNDLE_DESTINATIONS _destination_count)
  list(LENGTH ISAAC_RESOURCE_BUNDLE_TARGETS _target_count)
  if(NOT _destination_count EQUAL _target_count)
    message(FATAL_ERROR
      "isaac_resource_bundle_ensure(${ARG_BUNDLE_LABEL}): invalid manifest list lengths.")
  endif()

  isaac_resource_acquire_lock("${ARG_CACHE_DIR}/.lock" "${ARG_LOCK_TIMEOUT}")
  set(_content_dir "${ARG_CACHE_DIR}/content")
  file(MAKE_DIRECTORY "${_content_dir}")

  if(_destination_count GREATER 0)
    math(EXPR _last_index "${_destination_count} - 1")
    foreach(_index RANGE ${_last_index})
      list(GET ISAAC_RESOURCE_BUNDLE_DESTINATIONS ${_index} _destination)
      list(GET ISAAC_RESOURCE_BUNDLE_TARGETS ${_index} _target)
      set(_link "${_content_dir}/${_destination}")
      if(NOT IS_SYMLINK "${_link}" AND EXISTS "${_link}")
        message(FATAL_ERROR
          "isaac_resource_bundle_ensure(${ARG_BUNDLE_LABEL}): refusing to replace real object at declared destination: ${_link}")
      endif()
      isaac_resource_ensure_link("${_link}" "${_target}")
    endforeach()
  endif()

  isaac_resource_ensure_link("${ARG_LINK_DIR}" "${_content_dir}")
endfunction()
