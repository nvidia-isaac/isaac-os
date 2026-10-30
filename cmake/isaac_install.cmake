# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

set(ISAAC_INSTALL_LIBEXEC_DIR "${CMAKE_INSTALL_LIBEXECDIR}/isaac_os")
set(ISAAC_INSTALL_SHARE_DIR "${CMAKE_INSTALL_DATADIR}/isaac_os")

# Resolve a project-local source file. Real paths prevent symlinks from
# escaping the project source tree.
function(_isaac_source_file_absolute_path source_file output)
  file(REAL_PATH "${PROJECT_SOURCE_DIR}" _project_source_dir)
  file(
    REAL_PATH "${source_file}" _absolute_path
    BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  if(NOT EXISTS "${_absolute_path}")
    message(FATAL_ERROR "isaac_install source file does not exist: ${source_file}")
  endif()

  cmake_path(
    IS_PREFIX _project_source_dir "${_absolute_path}"
    NORMALIZE _inside_project_source
  )
  if(NOT _inside_project_source)
    message(FATAL_ERROR
      "isaac_install source file is outside PROJECT_SOURCE_DIR: ${source_file}"
    )
  endif()

  set("${output}" "${_absolute_path}" PARENT_SCOPE)
endfunction()

# Map a source file to its project-relative location in the installed share
# tree.
function(_isaac_source_file_install_paths
    source_file
    absolute_path_output
    destination_output
    installed_path_output
)
  _isaac_source_file_absolute_path("${source_file}" _absolute_path)
  file(REAL_PATH "${PROJECT_SOURCE_DIR}" _project_source_dir)
  file(
    RELATIVE_PATH _relative_path
    "${_project_source_dir}"
    "${_absolute_path}"
  )
  get_filename_component(_relative_parent "${_relative_path}" DIRECTORY)
  set(_destination "${ISAAC_INSTALL_SHARE_DIR}")
  if(_relative_parent)
    string(APPEND _destination "/${_relative_parent}")
  endif()

  set("${absolute_path_output}" "${_absolute_path}" PARENT_SCOPE)
  set("${destination_output}" "${_destination}" PARENT_SCOPE)
  set(
    "${installed_path_output}"
    "${ISAAC_INSTALL_SHARE_DIR}/${_relative_path}"
    PARENT_SCOPE
  )
endfunction()

# Install component-owned runtime targets, programs, and source files using
# conventional, relocatable destinations.
function(isaac_install)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "COMPONENT" "FILES;PROGRAMS;TARGETS")

  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "isaac_install received unknown arguments: ${ARG_UNPARSED_ARGUMENTS}"
    )
  endif()
  if(ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "isaac_install requires values for: ${ARG_KEYWORDS_MISSING_VALUES}"
    )
  endif()
  if(NOT ARG_COMPONENT)
    message(FATAL_ERROR "isaac_install requires COMPONENT")
  endif()
  if(NOT ARG_FILES AND NOT ARG_PROGRAMS AND NOT ARG_TARGETS)
    message(FATAL_ERROR "isaac_install requires FILES, PROGRAMS, or TARGETS")
  endif()

  file(
    RELATIVE_PATH _relative_libdir
    "/${CMAKE_INSTALL_BINDIR}"
    "/${CMAKE_INSTALL_LIBDIR}"
  )
  if(_relative_libdir STREQUAL ".")
    set(_executable_rpath "$ORIGIN")
  else()
    set(_executable_rpath "$ORIGIN/${_relative_libdir}")
  endif()

  set(_built_targets)
  set(_imported_targets)
  foreach(_target IN LISTS ARG_TARGETS)
    if(NOT TARGET "${_target}")
      message(FATAL_ERROR "isaac_install target does not exist: ${_target}")
    endif()

    get_target_property(_aliased_target "${_target}" ALIASED_TARGET)
    if(_aliased_target)
      message(FATAL_ERROR
        "isaac_install requires a concrete target, not alias ${_target}"
      )
    endif()

    get_target_property(_target_type "${_target}" TYPE)
    if(NOT _target_type MATCHES "^(EXECUTABLE|MODULE_LIBRARY|SHARED_LIBRARY)$")
      message(FATAL_ERROR
        "isaac_install does not support ${_target_type} target ${_target}"
      )
    endif()

    get_target_property(_existing_component "${_target}" ISAAC_INSTALL_COMPONENT)
    if(_existing_component)
      message(FATAL_ERROR
        "isaac_install target ${_target} is already registered "
        "for component ${_existing_component}"
      )
    endif()

    get_target_property(_imported "${_target}" IMPORTED)
    if(_imported)
      list(APPEND _imported_targets "${_target}")
    else()
      list(APPEND _built_targets "${_target}")
      if(_target_type STREQUAL "EXECUTABLE")
        set_property(TARGET "${_target}" PROPERTY INSTALL_RPATH "${_executable_rpath}")
      else()
        set_property(TARGET "${_target}" PROPERTY INSTALL_RPATH "$ORIGIN")
      endif()
    endif()
    set_property(TARGET "${_target}" PROPERTY ISAAC_INSTALL_COMPONENT "${ARG_COMPONENT}")
  endforeach()

  if(_built_targets)
    install(TARGETS ${_built_targets}
      RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT "${ARG_COMPONENT}"
      LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT "${ARG_COMPONENT}"
    )
  endif()

  if(_imported_targets)
    install(IMPORTED_RUNTIME_ARTIFACTS ${_imported_targets}
      RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT "${ARG_COMPONENT}"
      LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT "${ARG_COMPONENT}"
    )
  endif()

  foreach(_program IN LISTS ARG_PROGRAMS)
    _isaac_source_file_absolute_path("${_program}" _absolute_path)
    install(PROGRAMS "${_absolute_path}"
      DESTINATION "${ISAAC_INSTALL_LIBEXEC_DIR}"
      COMPONENT "${ARG_COMPONENT}"
    )
  endforeach()

  foreach(_source_file IN LISTS ARG_FILES)
    _isaac_source_file_install_paths(
      "${_source_file}"
      _absolute_path
      _destination
      _installed_path
    )
    install(FILES "${_absolute_path}"
      DESTINATION "${_destination}"
      COMPONENT "${ARG_COMPONENT}"
    )
  endforeach()
endfunction()
