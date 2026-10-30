# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file contains helper functions for managing CMake targets in the Isaac project.

#[[
# Promotes a specific list of targets to be globally visible.
#
# This function is designed to be used for packages like PyTorch where the
# automatic discovery of imported targets does not work due to the package's
# complex internal scoping.
#
# @param ... A list of target names to promote.
#]]
function(isaac_promote_explicit_targets)
  foreach(target_name ${ARGN})
    if(TARGET ${target_name})
      get_target_property(is_imported ${target_name} IMPORTED)
      if(is_imported)
        #message(STATUS "Promoting explicit target '${target_name}' to global scope.")
        set_target_properties(${target_name} PROPERTIES IMPORTED_GLOBAL TRUE)
      endif()
    endif()
  endforeach()
endfunction()

#[[
# Gets the list of all IMPORTED targets in the current directory.
#
# This function is intended to be called before a find_package() call to capture
# the state of the build. Requires CMake 3.21 or newer.
#
# @param output_variable The name of the variable to store the list of targets in.
#]]
function(isaac_get_imported_targets_in_scope output_variable)
  get_property(targets DIRECTORY . PROPERTY IMPORTED_TARGETS)
  set(${output_variable} ${targets} PARENT_SCOPE)
endfunction()

#[[
# Finds all new imported targets since a previous point and promotes them to be globally visible.
#
# This function is designed to be called after a find_package() command to make the
# discovered targets available throughout the entire project. Requires CMake 3.21 or newer.
#
# @param targets_before_var The name of the variable holding the list of targets
#                           captured by isaac_get_imported_targets_in_scope() before the
#                           find_package() call.
#]]
function(isaac_promote_new_imported_targets targets_before_var)
  # Get the list of all imported targets now.
  get_property(targets_after DIRECTORY . PROPERTY IMPORTED_TARGETS)

  # The variable passed in is the *name* of the list, so we need to dereference it
  # to get the actual list content from the parent scope.
  set(targets_before ${${targets_before_var}})

  # Compare the 'after' list to the 'before' list to find new targets.
  if(targets_before)
    list(REMOVE_ITEM targets_after ${targets_before})
  endif()

  # Promote any new targets.
  isaac_promote_explicit_targets(${targets_after})
endfunction()
