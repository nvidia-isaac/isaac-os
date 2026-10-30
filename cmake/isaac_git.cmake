# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(isaac_git_url_to_https output_variable git_url)
  if(git_url MATCHES "^https://")
    set(${output_variable} "${git_url}" PARENT_SCOPE)
    return()
  endif()

  if(git_url MATCHES "^ssh://([^@/]+@)?([^/:]+)(:[0-9]+)?/(.+)$")
    set(${output_variable} "https://${CMAKE_MATCH_2}/${CMAKE_MATCH_4}" PARENT_SCOPE)
    return()
  endif()

  if(git_url MATCHES "^([^@/:]+)@([^:]+):(.+)$")
    set(${output_variable} "https://${CMAKE_MATCH_2}/${CMAKE_MATCH_3}" PARENT_SCOPE)
    return()
  endif()

  message(FATAL_ERROR "Expected an HTTPS or SSH Git URL, got '${git_url}'.")
endfunction()

# isaac_git_basic_auth_push(
#   <state-variable> <repository-variable> <credential-environment-variable>)
# isaac_git_basic_auth_pop(<state-variable>)
#
# When the credential environment variable is non-empty, temporarily convert
# the repository from SSH to HTTPS and configure its Basic authorization
# header. With no credential, leave the repository and Git configuration alone.
macro(isaac_git_basic_auth_push
    state_variable repository_variable credential_environment_variable)
  if(DEFINED ${state_variable}_ACTIVE)
    message(FATAL_ERROR "Git Basic auth state '${state_variable}' is already active.")
  endif()

  set(${state_variable}_ACTIVE TRUE)
  set(${state_variable}_AUTHENTICATED FALSE)
  set(${state_variable}_REPOSITORY "${${repository_variable}}")
  set(${state_variable}_REPOSITORY_VARIABLE "${repository_variable}")
  set(_isaac_git_basic_auth_credential "$ENV{${credential_environment_variable}}")

  if(NOT _isaac_git_basic_auth_credential STREQUAL "")
    if(NOT _isaac_git_basic_auth_credential MATCHES "^[A-Za-z0-9+/]+(=|==)?$")
      message(FATAL_ERROR
        "The encoded Git credential in ${credential_environment_variable} is not valid base64.")
    endif()

    isaac_git_url_to_https(${repository_variable} "${${repository_variable}}")
    isaac_git_config_push(
      ${state_variable}_CONFIG
      "http.${${repository_variable}}.extraHeader=Authorization: Basic ${_isaac_git_basic_auth_credential}"
    )
    set(${state_variable}_AUTHENTICATED TRUE)
  endif()

  unset(_isaac_git_basic_auth_credential)
endmacro()

macro(isaac_git_basic_auth_pop state_variable)
  if(NOT DEFINED ${state_variable}_ACTIVE)
    message(FATAL_ERROR "Git Basic auth state '${state_variable}' is not active.")
  endif()

  if(${state_variable}_AUTHENTICATED)
    isaac_git_config_pop(${state_variable}_CONFIG)
    set(${${state_variable}_REPOSITORY_VARIABLE} "${${state_variable}_REPOSITORY}")
  endif()

  unset(${state_variable}_ACTIVE)
  unset(${state_variable}_AUTHENTICATED)
  unset(${state_variable}_REPOSITORY)
  unset(${state_variable}_REPOSITORY_VARIABLE)
endmacro()

# isaac_git_config_push(<state-variable> <key=value>...)
# isaac_git_config_pop(<state-variable>)
#
# Temporarily append correctly quoted entries to GIT_CONFIG_PARAMETERS. Use a
# unique state variable for each active push; nested pushes are supported.
macro(isaac_git_config_push state_variable)
  if(DEFINED ${state_variable}_ACTIVE)
    message(FATAL_ERROR "Git config state '${state_variable}' is already active.")
  endif()
  if("${ARGN}" STREQUAL "")
    message(FATAL_ERROR "At least one Git config parameter is required.")
  endif()

  if(DEFINED ENV{GIT_CONFIG_PARAMETERS})
    set(${state_variable}_WAS_DEFINED TRUE)
    set(${state_variable}_VALUE "$ENV{GIT_CONFIG_PARAMETERS}")
  else()
    set(${state_variable}_WAS_DEFINED FALSE)
    set(${state_variable}_VALUE "")
  endif()

  set(_isaac_git_config_parameters "${${state_variable}_VALUE}")
  foreach(_isaac_git_config_parameter IN ITEMS ${ARGN})
    if(_isaac_git_config_parameter MATCHES "['\r\n]")
      message(FATAL_ERROR "Git config parameters cannot contain quotes or newlines.")
    endif()
    string(APPEND _isaac_git_config_parameters " '${_isaac_git_config_parameter}'")
  endforeach()
  string(STRIP "${_isaac_git_config_parameters}" _isaac_git_config_parameters)
  set(ENV{GIT_CONFIG_PARAMETERS} "${_isaac_git_config_parameters}")
  set(${state_variable}_ACTIVE TRUE)

  unset(_isaac_git_config_parameter)
  unset(_isaac_git_config_parameters)
endmacro()

macro(isaac_git_config_pop state_variable)
  if(NOT DEFINED ${state_variable}_ACTIVE)
    message(FATAL_ERROR "Git config state '${state_variable}' is not active.")
  endif()

  if(${state_variable}_WAS_DEFINED)
    set(ENV{GIT_CONFIG_PARAMETERS} "${${state_variable}_VALUE}")
  else()
    unset(ENV{GIT_CONFIG_PARAMETERS})
  endif()

  unset(${state_variable}_ACTIVE)
  unset(${state_variable}_VALUE)
  unset(${state_variable}_WAS_DEFINED)
endmacro()
