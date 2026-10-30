# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# libevdev dependency setup (system package: libevdev-dev).

# Already exists?
if(TARGET libevdev::libevdev)
  return()
endif()

find_library(LIBEVDEV_LIBRARY NAMES evdev DOC "libevdev library" REQUIRED)
# Headers live in a versioned dir, e.g. /usr/include/libevdev-1.0/libevdev/libevdev.h
find_path(LIBEVDEV_INCLUDE_DIR
    NAMES libevdev/libevdev.h
    PATH_SUFFIXES libevdev-1.0
    DOC "libevdev headers"
    REQUIRED
)

add_library(libevdev::libevdev SHARED IMPORTED GLOBAL)
set_target_properties(libevdev::libevdev PROPERTIES
    IMPORTED_LOCATION ${LIBEVDEV_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${LIBEVDEV_INCLUDE_DIR}
)

message(STATUS "Found libevdev: ${LIBEVDEV_LIBRARY}")
