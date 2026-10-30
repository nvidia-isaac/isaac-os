# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(isaac_check_nvenc output)
  find_library(_isaac_nvidia_encode_library
    NAMES
      nvidia-encode
      libnvidia-encode.so.1
    PATH_SUFFIXES nvidia
    NO_CACHE
  )
  if(_isaac_nvidia_encode_library)
    set(${output} ON PARENT_SCOPE)
  else()
    set(${output} OFF PARENT_SCOPE)
  endif()
endfunction()
