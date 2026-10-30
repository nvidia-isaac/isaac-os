# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holohub_config_helper)

# Isaac-specific wrapper functions for building applications, operators, and common libraries.

# Internal macro to parse dependencies and enable common libraries.
macro(_isaac_prepare_dependencies NAME PREFIX)
  cmake_parse_arguments(ARG "" "" "DEPENDS" ${ARGN})
  set(HOLOHUB_DEPENDS "")

  if(ARG_DEPENDS)
    cmake_parse_arguments(DEPS "" "" "COMMON;OPERATORS;EXTENSIONS" ${ARG_DEPENDS})

    # Pass through OPERATORS and EXTENSIONS to HoloHub.
    if(DEPS_OPERATORS)
      list(APPEND HOLOHUB_DEPENDS "OPERATORS" ${DEPS_OPERATORS})
    endif()
    if(DEPS_EXTENSIONS)
      list(APPEND HOLOHUB_DEPENDS "EXTENSIONS" ${DEPS_EXTENSIONS})
    endif()

    # Enable COMMON libraries if the parent target is enabled.
    if(DEPS_COMMON)
      set(target_var "${PREFIX}_${NAME}")
      set(target_enabled FALSE)
      if(DEFINED ${target_var})
        set(target_enabled ${${target_var}})
      endif()
      if(target_enabled)
        foreach(lib IN LISTS DEPS_COMMON)
          set("LIB_${lib}" ON CACHE BOOL "Build the ${lib} common library" FORCE)
        endforeach()
      endif()
    endif()
  endif()
endmacro()

# =====================================================
# Helper function to build Isaac applications with common library dependencies.
# =====================================================
# Extends add_holohub_application to support COMMON library dependencies.
#
# Parameters:
#   NAME: The name of the application to build
#
# Keyword Arguments:
#   DEPENDS: Dependency specification with sub-arguments:
#     COMMON: List of common libraries that this application depends on
#     OPERATORS: List of operators that this application depends on (supports OPTIONAL)
#     EXTENSIONS: List of GXF extensions that this application depends on
#
# Example:
#   add_isaac_application(my_app DEPENDS
#     COMMON mcap foxglove
#     OPERATORS my_op OPTIONAL optional_op
#   )
function(add_isaac_application NAME)
  _isaac_prepare_dependencies(${NAME} "APP" ${ARGN})
  if(HOLOHUB_DEPENDS)
    add_holohub_application(${NAME} DEPENDS ${HOLOHUB_DEPENDS})
  else()
    add_holohub_application(${NAME})
  endif()
endfunction()

# Launch a headless camera app against the test-only UCXX image source and verify the image is
# consumed by MessageDropSink.
function(add_headless_camera_test TARGET TEST_NAME CONFIG CAMERA_KEY)
  add_test(
    NAME ${TEST_NAME}
    COMMAND $<TARGET_FILE:test_unitree_sim_ucx_headless_camera>
      $<TARGET_FILE:${TARGET}>
      ${CONFIG}
      ${CAMERA_KEY}
  )
  set_tests_properties(${TEST_NAME} PROPERTIES
    LABELS "cpp;applications;gpu;ucx"
    TIMEOUT 60
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
endfunction()

# =====================================================
# Helper function to build Isaac operators.
# =====================================================
# Wrapper for add_holohub_operator for consistent naming.
#
# Parameters:
#   NAME: The name of the operator to build
#
# Keyword Arguments:
#   DEPENDS: Dependency specification with sub-arguments:
#     COMMON: List of common libraries that this operator depends on
#     OPERATORS: List of operators that this operator depends on
#     EXTENSIONS: List of GXF extensions that this operator depends on
#
# Example:
#   add_isaac_operator(my_op DEPENDS COMMON mcap OPERATORS other_op)
function(add_isaac_operator NAME)
  _isaac_prepare_dependencies(${NAME} "OP" ${ARGN})
  if(HOLOHUB_DEPENDS)
    add_holohub_operator(${NAME} DEPENDS ${HOLOHUB_DEPENDS})
  else()
    add_holohub_operator(${NAME})
  endif()
endfunction()

# =====================================================
# Helper function to build common libraries selectively.
# =====================================================
# Builds a common library if its LIB_${NAME} option is enabled.
# Applications and operators set LIB_* options via their COMMON dependencies.
#
# Parameters:
#   NAME: The name of the common library to build
#
# Keyword Arguments:
#   DEPENDS: List of other common libraries that this library depends on.
#     When LIB_${NAME} is ON, LIB_${dep} is forced ON for each dependency.
#     Libraries with DEPENDS must be listed BEFORE their dependencies in
#     CMakeLists.txt so the flags are set before the dependency is processed.
#
# Creates:
#   LIB_${NAME}: CMake option to enable/disable this library (defaults to BUILD_ALL)
#
# Example:
#   add_isaac_common(mcap DEPENDS foxglove)
#   add_isaac_common(foxglove)
function(add_isaac_common NAME)
  cmake_parse_arguments(ARG "" "" "DEPENDS" ${ARGN})
  set(libname "LIB_${NAME}")

  option(${libname} "Build the ${NAME} common library" ${BUILD_ALL})

  # Enable dependencies when this library is enabled.
  if(${libname} AND ARG_DEPENDS)
    foreach(dep IN LISTS ARG_DEPENDS)
      if(NOT LIB_${dep})
        message(STATUS "Enabling ${dep} because ${NAME} depends on it")
        set("LIB_${dep}" ON CACHE BOOL "Build the ${dep} common library" FORCE)
      endif()
    endforeach()
  endif()

  if(${libname})
    message(STATUS "Building common library: ${NAME}")
    add_subdirectory(${NAME})
  else()
    message(STATUS "Skipping common library: ${NAME}")
  endif()
endfunction()
