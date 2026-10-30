# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# isaac_set(<var> [DEFAULT <value>] [TYPE <type>] [DESCRIPTION <desc>])
#
# Resolve a project-level variable from the first available source:
#   1. CMake cache / -D command line (already defined)
#   2. Environment variable of the same name
#   3. DEFAULT value (if provided)
# If none of the above apply, the variable remains unset.
#
# When TYPE is given the variable is promoted to a cache entry (visible in
# cmake-gui / ccmake). DESCRIPTION is used as the cache entry docstring.
#
# Must be a macro so the variable is set in the caller's scope.
macro(isaac_set var)
  cmake_parse_arguments(_SFE "" "DEFAULT;TYPE;DESCRIPTION" "" ${ARGN})

  # Set the variable from the first available source.
  if(NOT DEFINED ${var})
    if(DEFINED ENV{${var}})
      set(${var} "$ENV{${var}}")
    elseif(DEFINED _SFE_DEFAULT)
      set(${var} "${_SFE_DEFAULT}")
    endif()
  endif()

  # Promote to a cache entry when DESCRIPTION or TYPE is provided (visible in cmake-gui).
  if(DEFINED _SFE_DESCRIPTION OR DEFINED _SFE_TYPE)
    # No TYPE: infer BOOL from boolean-valued DEFAULT; otherwise default to STRING.
    if(NOT DEFINED _SFE_TYPE)
      if(DEFINED _SFE_DEFAULT)
        string(TOUPPER "${_SFE_DEFAULT}" _SFE_DEFAULT_UPPER)
        if(_SFE_DEFAULT_UPPER MATCHES "^(ON|OFF|TRUE|FALSE|YES|NO)$")
          set(_SFE_TYPE BOOL)
        endif()
        unset(_SFE_DEFAULT_UPPER)
      endif()
      if(NOT DEFINED _SFE_TYPE)
        set(_SFE_TYPE STRING)
      endif()
    endif()

    # Always create cache entry so unset variables are still visible.
    if(NOT DEFINED ${var})
      set(${var} "" CACHE ${_SFE_TYPE} "${_SFE_DESCRIPTION}")
    else()
      set(${var} "${${var}}" CACHE ${_SFE_TYPE} "${_SFE_DESCRIPTION}")
    endif()
  endif()

  # Log the variable value.
  if(DEFINED ${var})
    message(STATUS "${var}: ${${var}}")
  else()
    message(STATUS "${var}: <not set>")
  endif()

  # Clear the parsed arguments to avoid polluting the caller's scope (we're a macro).
  unset(_SFE_DEFAULT)
  unset(_SFE_TYPE)
  unset(_SFE_DESCRIPTION)
  unset(_SFE_UNPARSED_ARGUMENTS)
endmacro()
