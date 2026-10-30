/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/tensorrt_engine/tensorrt_engine.h"

#include <filesystem>
#include <fstream>
#include <string>

#include "gtest/gtest.h"

namespace isaac {

TEST(TensorRTEngineTest, ReturnsErrorForNonExistentOnnx) {
  auto result = TensorRTEngine::create("test", "/nonexistent/model.onnx");
  EXPECT_FALSE(result.has_value());
}

TEST(TensorRTEngineTest, ReturnsErrorForInvalidOnnx) {
  const auto path = std::filesystem::path(TEST_OUTPUT_DIR) / "tensorrt_engine_test_invalid.onnx";
  {
    std::ofstream f(path, std::ios::binary);
    f << "not a valid onnx file";
  }
  auto result = TensorRTEngine::create("test", path.string());
  EXPECT_FALSE(result.has_value());
  std::filesystem::remove(path);
}

TEST(TensorRTEngineTest, LoadsIdentityModel) {
  // Use a directory in the build tree so the built .engine cache never lands in TEST_DATA_DIR
  // (i.e. the source tree). The directory and its contents are removed in teardown.
  const auto tmp_dir =
      std::filesystem::path(TEST_OUTPUT_DIR) / "tensorrt_engine_test_identity_model";
  std::filesystem::remove_all(tmp_dir);
  ASSERT_TRUE(std::filesystem::create_directory(tmp_dir));

  const auto tmp_onnx = tmp_dir / "identity.onnx";
  ASSERT_TRUE(std::filesystem::copy_file(TEST_DATA_DIR "/identity.onnx", tmp_onnx));

  auto result = TensorRTEngine::create("test", tmp_onnx.string());
  ASSERT_TRUE(result.has_value());

  // Keep both engines alive to exercise creating multiple TensorRT runtimes in one process.
  auto second_result = TensorRTEngine::create("second test", tmp_onnx.string());
  ASSERT_TRUE(second_result.has_value());

  auto& engine = (*result)->engine();
  EXPECT_EQ(engine.getNbIOTensors(), 2);

  auto& ctx = (*result)->context();
  const auto in_shape = ctx.getTensorShape("input");
  ASSERT_EQ(in_shape.nbDims, 1);
  EXPECT_EQ(in_shape.d[0], 4);

  const auto out_shape = ctx.getTensorShape("output");
  ASSERT_EQ(out_shape.nbDims, 1);
  EXPECT_EQ(out_shape.d[0], 4);

  std::filesystem::remove_all(tmp_dir);
}

}  // namespace isaac
