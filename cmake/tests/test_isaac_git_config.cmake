# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

include("${CMAKE_CURRENT_LIST_DIR}/../isaac_git.cmake")

isaac_git_url_to_https(
  _https_url
  "ssh://git@gitlab.example.com:12051/group/repository.git"
)
if(NOT _https_url STREQUAL "https://gitlab.example.com/group/repository.git")
  message(FATAL_ERROR "Unexpected HTTPS Git URL: '${_https_url}'")
endif()

isaac_git_url_to_https(
  _https_url
  "ssh://gitlab.example.com/group/repository.git"
)
if(NOT _https_url STREQUAL "https://gitlab.example.com/group/repository.git")
  message(FATAL_ERROR "Unexpected HTTPS Git URL without SSH user: '${_https_url}'")
endif()

isaac_git_url_to_https(
  _https_url
  "git@gitlab.example.com:group/repository.git"
)
if(NOT _https_url STREQUAL "https://gitlab.example.com/group/repository.git")
  message(FATAL_ERROR "Unexpected HTTPS Git URL from SCP syntax: '${_https_url}'")
endif()

isaac_git_url_to_https(
  _https_url
  "https://gitlab.example.com/group/repository.git"
)
if(NOT _https_url STREQUAL "https://gitlab.example.com/group/repository.git")
  message(FATAL_ERROR "Existing HTTPS Git URL was modified: '${_https_url}'")
endif()

set(ENV{GIT_CONFIG_PARAMETERS} "'existing.key=existing value'")
set(_expected_git_config
  "'existing.key=existing value' 'credential.helper=' 'http.https://example.com/repo.git.extraHeader=Authorization: Basic abc='")

isaac_git_config_push(
  _test_git_config
  "credential.helper="
  "http.https://example.com/repo.git.extraHeader=Authorization: Basic abc="
)

if(NOT "$ENV{GIT_CONFIG_PARAMETERS}" STREQUAL "${_expected_git_config}")
  message(FATAL_ERROR
    "Expected Git config '${_expected_git_config}', got '$ENV{GIT_CONFIG_PARAMETERS}'")
endif()
isaac_git_config_pop(_test_git_config)
if(NOT "$ENV{GIT_CONFIG_PARAMETERS}" STREQUAL "'existing.key=existing value'")
  message(FATAL_ERROR "Existing Git config was not restored")
endif()

unset(ENV{GIT_CONFIG_PARAMETERS})
set(_expected_git_config "'credential.helper='")
isaac_git_config_push(_test_git_config "credential.helper=")

if(NOT "$ENV{GIT_CONFIG_PARAMETERS}" STREQUAL "${_expected_git_config}")
  message(FATAL_ERROR
    "Expected Git config '${_expected_git_config}', got '$ENV{GIT_CONFIG_PARAMETERS}'")
endif()
isaac_git_config_pop(_test_git_config)
if(DEFINED ENV{GIT_CONFIG_PARAMETERS})
  message(FATAL_ERROR "Originally undefined Git config was not unset")
endif()

set(_ssh_repository "ssh://git@gitlab.example.com:12051/group/repository.git")
set(_repository "${_ssh_repository}")
set(ENV{TEST_GIT_BASIC_AUTH} "b2F1dGgyOnRva2Vu")
isaac_git_basic_auth_push(_test_git_auth _repository TEST_GIT_BASIC_AUTH)

if(NOT _repository STREQUAL "https://gitlab.example.com/group/repository.git")
  message(FATAL_ERROR "Authenticated Git repository was not converted to HTTPS")
endif()
set(_expected_git_config
  "'http.https://gitlab.example.com/group/repository.git.extraHeader=Authorization: Basic b2F1dGgyOnRva2Vu'")
if(NOT "$ENV{GIT_CONFIG_PARAMETERS}" STREQUAL "${_expected_git_config}")
  message(FATAL_ERROR
    "Expected Git config '${_expected_git_config}', got '$ENV{GIT_CONFIG_PARAMETERS}'")
endif()

isaac_git_basic_auth_pop(_test_git_auth)
if(NOT _repository STREQUAL "${_ssh_repository}")
  message(FATAL_ERROR "Authenticated Git repository was not restored")
endif()
if(DEFINED ENV{GIT_CONFIG_PARAMETERS})
  message(FATAL_ERROR "Authenticated Git config was not unset")
endif()

unset(ENV{TEST_GIT_BASIC_AUTH})
isaac_git_basic_auth_push(_test_git_auth _repository TEST_GIT_BASIC_AUTH)
if(NOT _repository STREQUAL "${_ssh_repository}")
  message(FATAL_ERROR "Unauthenticated Git repository was modified")
endif()
isaac_git_basic_auth_pop(_test_git_auth)
