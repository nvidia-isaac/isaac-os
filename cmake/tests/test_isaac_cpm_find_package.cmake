# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/..")

function(CPMAddPackage)
endfunction()

macro(CPMFindPackage)
  set(_observed_cpm_args ${ARGN})
endmacro()

include("${CMAKE_CURRENT_LIST_DIR}/../isaac_cpm_find_package.cmake")

function(isaac_get_imported_targets_in_scope output_variable)
  set(${output_variable} "" PARENT_SCOPE)
endfunction()

function(isaac_promote_new_imported_targets)
endfunction()

isaac_cpm_find_package(NAME test_package)
list(FIND _observed_cpm_args GIT_SHALLOW _git_shallow_index)
if(NOT _git_shallow_index EQUAL -1)
  message(FATAL_ERROR "Default GIT_SHALLOW prevented CPM from handling commit hashes")
endif()

list(FIND _observed_cpm_args GIT_CONFIG _git_config_index)
if(_git_config_index EQUAL -1)
  message(FATAL_ERROR "Default GIT_CONFIG was not passed to CPM")
endif()
math(EXPR _git_config_value_index "${_git_config_index} + 1")
list(GET _observed_cpm_args ${_git_config_value_index} _git_config)
if(NOT _git_config STREQUAL
    "core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=15")
  message(FATAL_ERROR "Unexpected default GIT_CONFIG: '${_git_config}'")
endif()

isaac_cpm_find_package(NAME test_package GIT_CONFIG "core.sshCommand=custom-ssh-command")
list(FIND _observed_cpm_args "core.sshCommand=custom-ssh-command" _custom_git_config_index)
if(_custom_git_config_index EQUAL -1)
  message(FATAL_ERROR "Caller GIT_CONFIG did not override the default")
endif()
