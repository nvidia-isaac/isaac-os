# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(isaac_deps/holoscan)
include(isaac_deps/mcap)

add_library(mcap_replayer SHARED
  compressed_video_builder.cc
  compressed_video_builder.h
  decoder.h
  mcap_file.cc
  mcap_replayer.cc
  pacing.cc
  message.h
  replay_context.h
  replay_stats.cc
  typed_port.h
)
add_library(isaac::common::mcap_replayer ALIAS mcap_replayer)

target_include_directories(mcap_replayer PUBLIC ${CMAKE_SOURCE_DIR})
# mcap_file.h declares members of type mcap::McapReader and mcap::LinearMessageView, so any
# consumer that includes mcap_file.h needs mcap's headers too -- but not its static library.
# mcap statically bundles zstd with default (exported) symbol visibility, and links it PUBLIC
# itself, so linking the mcap *target* PUBLIC here would carry that archive one level further
# out on every consumer, past the --exclude-libs guard below (which only covers this library's
# own link step) and back into the exact collision it exists to prevent: a shared library built
# by a downstream consumer would re-export zstd symbols that can collide with a different zstd
# copy in the same process (e.g. the foxglove-sdk's own vendored zstd), and the dynamic linker's
# global symbol scope can then bind an internal zstd call in one library to the other's
# incompatible implementation -- memory corruption, not a link error. Propagating only mcap's
# include directories gets consumers the headers mcap_file.h needs without re-exporting the
# archive.
target_include_directories(mcap_replayer PUBLIC
  $<TARGET_PROPERTY:mcap,INTERFACE_INCLUDE_DIRECTORIES>)
# Same treatment, same reasoning, for the generated tensor envelope header: decoder.h includes it
# and the public mcap_replayer.h includes decoder.h unconditionally, so every consumer needs that
# header's include directory -- but none of them needs the logger's compiled code, or the public
# dependencies that come with linking it.
target_include_directories(mcap_replayer PUBLIC
  $<TARGET_PROPERTY:isaac::common::mcap_logger,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_options(mcap_replayer PRIVATE "-Wl,--exclude-libs,ALL")
target_link_libraries(mcap_replayer
  PUBLIC
    holoscan::core
    # decoder.h calls holoscan::schema_package<T>() in inline code, so every consumer of this
    # target compiles that call, and links against it, on its own behalf.
    holoscan::schema
    # The headers here are type-agnostic, but the tensor-schema types a consumer names (such as
    # holoscan::schema::ImageT) come with generated schema traits and port operations from this
    # target, which that consumer compiles against on its own behalf.
    holoscan::sensor_io
  PRIVATE
    # mcap_file.cc and replay_stats.cc call fmt::format(), and holoscan::core does not reliably
    # re-export fmt::fmt to its own consumers. PRIVATE because no header this target installs
    # expands an fmt call: the only headers here that log are compiled into this library alone,
    # so a consumer never needs fmt on its own behalf.
    fmt::fmt
    # Depend on the tensor envelope schema, not on the logger's own sink/writer code: every
    # HsdkMcapLogger/McapSink include must stay confined to tests/. PRIVATE, with only the
    # include directory propagated above: the generated envelope accessors are header-only
    # inline code, so nothing here calls a symbol this target compiles, and linking it PUBLIC
    # would put its own public dependencies on every consumer of this library for no symbol.
    isaac::common::mcap_logger
    mcap
)

if(ISAAC_ENABLE_NV_VIDEO_SDK)
  include(isaac_deps/cuda)
  # Private: the decoder and NPP appear only in compressed_video_builder.cc, behind the
  # builder's interface, so no consumer of this library needs them.
  target_compile_definitions(mcap_replayer PRIVATE ISAAC_ENABLE_NV_VIDEO_SDK)
  target_link_libraries(mcap_replayer
    PRIVATE
      CUDA::cuda_driver
      # nppiNV12ToYUV420 and nppiYCbCr420ToRGB live in the colour-conversion module,
      # nppiSwapChannels in the data exchange module.
      CUDA::nppicc
      CUDA::nppidei
      isaac::common::nv_video_decoder
  )
endif()

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
