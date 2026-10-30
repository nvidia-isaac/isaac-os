/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "mcap/reader.hpp"

namespace isaac::os::mcap_logger {

// What one chunk of a recording says about how it was compressed.
struct ChunkCompression {
  std::string algorithm;  // "" for none, otherwise "lz4" or "zstd".
  std::uint64_t compressed_bytes = 0;
  std::uint64_t uncompressed_bytes = 0;
};

// The compression each chunk of a finished recording declares, in file order. Read from the file
// alone, as any MCAP reader does: the reader is never told how the file was written.
inline std::vector<ChunkCompression> ReadChunkCompressions(const std::filesystem::path& path) {
  mcap::McapReader reader;
  const auto opened = reader.open(path.string());
  EXPECT_TRUE(opened.ok()) << opened.message;
  const auto summary = reader.readSummary(mcap::ReadSummaryMethod::NoFallbackScan);
  EXPECT_TRUE(summary.ok()) << summary.message;
  std::vector<ChunkCompression> chunks;
  for (const mcap::ChunkIndex& chunk : reader.chunkIndexes()) {
    chunks.push_back({chunk.compression, chunk.compressedSize, chunk.uncompressedSize});
  }
  return chunks;
}

}  // namespace isaac::os::mcap_logger
