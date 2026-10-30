/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nppi_color_conversion.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "holoscan/core/compile.hpp"
#include "holoscan/core/diagnostic.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/lifecycle.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/readiness_source.hpp"
#include "holoscan/core/run.hpp"
#include "holoscan/memory/tensor_storage.hpp"
#include "holoscan/sensor_io/image_metadata.hpp"
#include "holoscan/sensor_io/sensor_schema_package.hpp"
#include "holoscan/time/realtime_clock.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <flatbuffers/flatbuffers.h>

#include "common/image_schema/image_schema.h"
#include "common/mcap_logger/compressed_video_proto.h"
#include "common/mcap_logger/frame_encoder.h"
#include "common/mcap_logger/mcap_logger.h"
#include "common/mcap_logger/tests/post_with_retry.h"
#include "common/mcap_logger/tests/read_in_log_time_order.h"
#include "common/mcap_logger/tests/scoped_temp_directory.h"
#include "common/mcap_logger/video_encode_options.h"
#include "common/nv_video_decoder/nv_video_decoder.hpp"
#include "mcap/reader.hpp"

namespace isaac::os::mcap_logger {
namespace {

using isaac::os::mcap_logger::testing::post_with_retry;

constexpr int kMessageCount = 5;
// NVENC rejects a session below its minimum encode dimensions, so the tiny frames the raw
// tensor integration test uses would make every encode fail and leave both channels empty.
// 640x480 is comfortably above that floor.
constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 480;

// Which of FrameEncoder's staging paths a run exercises. The encoder picks its NPP entry point
// from two independent properties of the frame -- whether it can read it where it lies, and
// whether converting to ARGB changes the pixel width -- so the four combinations below reach it
// by four distinct routes, across three NPP entry points: a three-channel frame widens
// out-of-place through the same call whether it is read straight from the tensor or from a
// staged copy, a four-channel device frame swaps out-of-place, and a four-channel host frame
// swaps in place inside the staging buffer it was copied into. The two three-channel routes
// share an entry point but not the pointer they hand it, which is its own way to get a
// conversion wrong. A round trip that only ever feeds it host RGB8 leaves most of that code
// unchecked.
//
// The accessors below are exhaustive switches rather than comparisons against one value on
// purpose: written as `source != kHostRgb8` or `source == kDeviceRgba8 ? 4 : 3`, a newly added
// source would be silently misclassified instead of failing to compile.
enum class FrameSource { kHostRgb8, kHostRgba8, kDeviceRgb8, kDeviceRgba8 };

constexpr bool IsDeviceResident(FrameSource source) {
  switch (source) {
    case FrameSource::kDeviceRgb8:
    case FrameSource::kDeviceRgba8:
      return true;
    case FrameSource::kHostRgb8:
    case FrameSource::kHostRgba8:
      return false;
  }
  return false;
}

constexpr std::uint32_t ChannelsOf(FrameSource source) {
  switch (source) {
    case FrameSource::kHostRgba8:
    case FrameSource::kDeviceRgba8:
      return 4U;
    case FrameSource::kHostRgb8:
    case FrameSource::kDeviceRgb8:
      return 3U;
  }
  return 3U;
}

constexpr holoscan::schema::ImageEncoding EncodingOf(FrameSource source) {
  return ChannelsOf(source) == 4U ? holoscan::schema::ImageEncoding_RGBA8
                                  : holoscan::schema::ImageEncoding_RGB8;
}

constexpr std::size_t FrameBytesOf(FrameSource source) {
  return static_cast<std::size_t>(kWidth) * kHeight * ChannelsOf(source);
}

const char* NameOf(FrameSource source) {
  switch (source) {
    case FrameSource::kDeviceRgb8:
      return "DeviceRgb8";
    case FrameSource::kDeviceRgba8:
      return "DeviceRgba8";
    case FrameSource::kHostRgba8:
      return "HostRgba8";
    case FrameSource::kHostRgb8:
      break;
  }
  return "HostRgb8";
}

// A deliberately asymmetric pixel: the whole point of the colour-order check below is that an
// R/B swap must change the answer, and any colour with R == B (grey included) would survive one
// unnoticed. The values are also far enough apart to stay ordered through 4:2:0 subsampling.
// What the generator stamps on every frame beyond its pixels. foxglove.CompressedVideo has no field
// for any of it, so each has to come out of the sidecar exactly as it went in; the capture
// timestamp advances by one 30 fps frame interval per frame so that a sidecar matched to the wrong
// frame would not go unnoticed.
constexpr std::int64_t kFirstCaptureTimestampNs = 1'700'000'000'123'456'789;
constexpr std::int64_t kCaptureIntervalNs = 33'333'333;
constexpr std::int64_t kExposureTimeNs = 8'333'333;
constexpr float kGain = 2.5F;
constexpr std::uint8_t kSignificantBits = 8;
constexpr std::int32_t kRoiOffsetX = 12;
constexpr std::int32_t kRoiOffsetY = 34;
constexpr std::string_view kFrameId = "camera_optical_frame";

constexpr std::uint8_t kRed = 200;
constexpr std::uint8_t kGreen = 100;
constexpr std::uint8_t kBlue = 50;
// The alpha a four-channel source frame carries. The encoder discards it, so its value only has
// to be a legal opaque one.
constexpr std::uint8_t kOpaqueAlpha = 255;

// Interval between posted notifications -- same reasoning as
// test_mcap_logger_tensor_schema_integration.cc: TensorTopicPoller polls far faster than this,
// which keeps posting-then-checking deterministic rather than racing an unread sample's
// overwrite.
constexpr auto kNotificationPeriod = std::chrono::milliseconds(20);

// Publishes the running generator operator's NotificationSender out to the test, since
// holoscan::compile() reconstructs operators into its own immutable canonical Graph rather than
// reusing compose()'s objects.
class GeneratorTriggerProbe {
 public:
  static void publish(holoscan::NotificationSender sender) {
    const std::scoped_lock lock(mutex_);
    senders_.push_back(std::move(sender));
    cv_.notify_all();
  }

  // Bounded rather than an open wait: if arm() never publishes, an unbounded wait would hang
  // RunGraph() and with it the whole test binary, which a CI run reports as a timeout with no
  // indication of which test stopped. The deadline is far longer than arming has ever taken, so
  // reaching it means something is broken rather than slow.
  [[nodiscard]] static std::optional<holoscan::NotificationSender> wait_for_sender(
      std::chrono::seconds timeout = std::chrono::seconds(30)) {
    auto senders = wait_for_senders(1, timeout);
    if (!senders) {
      return std::nullopt;
    }
    return senders->front();
  }

  // A graph with more than one generator arms them independently, so a caller driving both has
  // to wait for both rather than for whichever happened to arm first.
  [[nodiscard]] static std::optional<std::vector<holoscan::NotificationSender>> wait_for_senders(
      std::size_t count, std::chrono::seconds timeout = std::chrono::seconds(30)) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [count] { return senders_.size() >= count; })) {
      return std::nullopt;
    }
    return senders_;
  }

  static void reset() {
    const std::scoped_lock lock(mutex_);
    senders_.clear();
  }

 private:
  static std::mutex mutex_;
  static std::condition_variable cv_;
  static std::vector<holoscan::NotificationSender> senders_;
};

std::mutex GeneratorTriggerProbe::mutex_;
std::condition_variable GeneratorTriggerProbe::cv_;
std::vector<holoscan::NotificationSender> GeneratorTriggerProbe::senders_;

// A device allocation that frees itself, so nothing below calls cudaFree by hand -- including on
// the early returns, which is where a manual free gets forgotten. Typed as the bytes it holds
// rather than void*: cudaMalloc hands back an untyped pointer, and this is the one place that says
// what it is.
struct CudaFree {
  void operator()(std::uint8_t* pointer) const noexcept { cudaFree(pointer); }
};
using DeviceBuffer = std::unique_ptr<std::uint8_t, CudaFree>;

// A device buffer of `bytes`, or null when the allocation fails.
[[nodiscard]] DeviceBuffer AllocateDeviceBuffer(std::size_t bytes) {
  void* allocation = nullptr;
  if (cudaMalloc(&allocation, bytes) != cudaSuccess) {
    return nullptr;
  }
  return DeviceBuffer(static_cast<std::uint8_t*>(allocation));
}

// holoscan::make_tensor()'s plain pointer+release_function overload (what the host frames below
// use) wraps the pointer in a TensorMemoryReference that cannot report a readiness stream, and
// the tensor transport refuses to stage any CUDA-resident Tensor whose memory reference cannot.
// This minimal override records whichever stream the producer sets and answers
// has_readiness_stream_contract() with true so the transport accepts it.
class CudaBufferMemoryReference final : public holoscan::TensorMemoryReference {
 public:
  explicit CudaBufferMemoryReference(DeviceBuffer buffer) : buffer_(std::move(buffer)) {}

  bool set_readiness_stream(cudaStream_t stream) override {
    readiness_stream_ = stream;
    return true;
  }
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }
  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override {
    return readiness_stream_;
  }

  // Also required: the transport calls this on the retained source Tensor after enqueuing a
  // copy on its own stream, so this buffer is not reused or freed until that copy completes.
  bool set_deallocation_stream(cudaStream_t stream) override {
    deallocation_stream_ = stream;
    return true;
  }

 private:
  DeviceBuffer buffer_;
  cudaStream_t readiness_stream_ = nullptr;
  cudaStream_t deallocation_stream_ = nullptr;
};

// Emits one kWidth x kHeight ImageT frame of the flat (kRed, kGreen, kBlue) colour each time its
// NotificationSource is posted to, in whichever encoding and memory kSource names. Templated
// rather than configured at run time because holoscan::compile() rebuilds operators into its own
// canonical graph instead of reusing compose()'s objects, so a field set during compose() would
// not reach the operator that actually runs; the type does reach it.
template <FrameSource kSource>
class ImageGeneratorOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(image_out, "image_out");
    spec.notification_source(trigger_, "trigger").capacity(kMessageCount);
    spec.lifecycle().stage(holoscan::LifecycleStage::kArm, &ImageGeneratorOp::arm);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnNotified{.event = trigger_});
    return contract;
  }

  holoscan::LifecycleStatus arm(holoscan::LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(trigger_);
    if (!sender) {
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    GeneratorTriggerProbe::publish(std::move(*sender));
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    static constexpr std::size_t kFrameBytes = FrameBytesOf(kSource);
    static constexpr std::uint32_t kChannels = ChannelsOf(kSource);
    std::vector<std::uint8_t> host_pixels(kFrameBytes);
    for (std::size_t i = 0; i < kFrameBytes; i += kChannels) {
      host_pixels[i] = kRed;
      host_pixels[i + 1] = kGreen;
      host_pixels[i + 2] = kBlue;
      if constexpr (kChannels == 4U) {
        host_pixels[i + 3] = kOpaqueAlpha;
      }
    }

    const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(kFrameBytes)};
    holoscan::expected<holoscan::Tensor, holoscan::Error> tensor =
        holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure, "unset"});
    if constexpr (IsDeviceResident(kSource)) {
      // Lazily created, then reused for the operator's lifetime: the transport requires a
      // non-null readiness stream on every CUDA-resident Tensor it stages, even though the
      // copy below is synchronous and the data is already ready by the time it is set.
      if (stream_ == nullptr && cudaStreamCreate(&stream_) != cudaSuccess) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kResourceExhausted, "cudaStreamCreate failed."});
      }
      DeviceBuffer device_pixels = AllocateDeviceBuffer(kFrameBytes);
      if (!device_pixels) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kResourceExhausted, "cudaMalloc failed."});
      }
      if (cudaMemcpy(device_pixels.get(), host_pixels.data(), kFrameBytes,
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "cudaMemcpy (H2D) failed."});
      }
      // The address is read before the buffer moves into its reference, which owns it from here.
      std::uint8_t* const address = device_pixels.get();
      tensor = holoscan::make_tensor(
          shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt, holoscan::MemoryStorageType::kDevice,
          address, std::make_shared<CudaBufferMemoryReference>(std::move(device_pixels)));
      if (tensor) {
        tensor->set_readiness_stream(stream_);
      }
    } else {
      // The pixels move into storage the release function owns, so the Tensor keeps them alive
      // and nothing is allocated or freed by hand. They live exactly as long as that function
      // object does, whether or not the Tensor ever calls it.
      auto storage = std::make_shared<std::vector<std::uint8_t>>(std::move(host_pixels));
      tensor = holoscan::make_tensor(
          shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt, holoscan::MemoryStorageType::kSystem,
          storage->data(),
          [storage](void*) -> holoscan::expected<void, holoscan::Error> { return {}; });
    }
    if (!tensor) {
      return holoscan::make_unexpected(tensor.error());
    }

    holoscan::schema::ImageT value;
    value.height = kHeight;
    value.width = kWidth;
    value.encoding = EncodingOf(kSource);
    value.color_space = holoscan::schema::ColorSpace_REC709;
    value.significant_bits = kSignificantBits;
    value.roi_offset_x = kRoiOffsetX;
    value.roi_offset_y = kRoiOffsetY;
    value.exposure_time_ns = kExposureTimeNs;
    value.gain = kGain;
    value.header = std::make_shared<holoscan::schema::HeaderT>();
    value.header->capture_timestamp_ns =
        kFirstCaptureTimestampNs + frame_index_ * kCaptureIntervalNs;
    value.header->frame_id = std::string(kFrameId);
    ++frame_index_;
    value.data = std::make_unique<holoscan::Tensor>(std::move(*tensor));

    return image_out.emit(value);
  }

  holoscan::Output<holoscan::schema::ImageT> image_out;

 private:
  holoscan::NotificationSource trigger_;
  cudaStream_t stream_ = nullptr;
  std::int64_t frame_index_ = 0;
};

// Two encoded topics recorded at once. Each attach() builds its own FrameEncoder, and with it
// its own NVENC session, input-buffer ring and serializing mutex, so this is where two encoders
// sharing one GPU would show up interfering -- or where a device would refuse a second encoding
// session after the first has opened. Nothing else covers it: every other graph here attaches a
// single encoded topic, and the reference application records one and is not run by CI.
//
// Both topics are fed from one generator output rather than two generators. That keeps a single
// trigger driving both, and fans out an identical tensor contract -- which the runtime requires
// of endpoints on one output anyway -- so the encoders are the only thing that differs, which is
// the point of the test.
class TwoEncodedTopicsGraph final : public holoscan::Graph {
 public:
  explicit TwoEncodedTopicsGraph(HsdkMcapLogger& logger)
      : Graph("mcap_logger_two_encoded_topics_test"), logger_(logger) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    // A generator each. One output fanned out to two exported topics is refused at run time
    // with RUNTIME_PORT_BINDING_UNAVAILABLE, so the two encoders need two sources.
    attach_encoded("camera", op<ImageGeneratorOp<FrameSource::kHostRgb8>>("generator")->image_out);
    attach_encoded("camera_two",
                   op<ImageGeneratorOp<FrameSource::kHostRgb8>>("generator_two")->image_out);
  }

 private:
  void attach_encoded(std::string_view topic,
                      const holoscan::Output<holoscan::schema::ImageT>& source) {
    logger_.attach<holoscan::schema::ImageT>(
        *this, topic, source,
        image_schema::ImageTensorFieldContract(kWidth, kHeight, EncodingOf(FrameSource::kHostRgb8),
                                               holoscan::MemoryKind::kHost),
        TopicLogOptions{.encode = VideoEncodeOptions{}});
  }

  HsdkMcapLogger& logger_;
};

// Identical to the raw tensor integration graph except for the VideoEncodeOptions, which is the
// whole opt-in: the topic is recorded as foxglove.CompressedVideo plus a metadata sidecar
// instead of raw TensorFrames.
class EncodedVideoTestGraph final : public holoscan::Graph {
 public:
  EncodedVideoTestGraph(HsdkMcapLogger& logger, VideoEncodeOptions encode, FrameSource source)
      : Graph("mcap_logger_encoded_video_integration_test"),
        logger_(logger),
        encode_(std::move(encode)),
        source_(source) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("clock"));
    switch (source_) {
      case FrameSource::kDeviceRgb8:
        attach_generator<FrameSource::kDeviceRgb8>();
        return;
      case FrameSource::kDeviceRgba8:
        attach_generator<FrameSource::kDeviceRgba8>();
        return;
      case FrameSource::kHostRgba8:
        attach_generator<FrameSource::kHostRgba8>();
        return;
      case FrameSource::kHostRgb8:
        break;
    }
    attach_generator<FrameSource::kHostRgb8>();
  }

 private:
  // memory_kind has to match where the generator's Tensor really lives: left at the default
  // kHost, the transport would copy a device-resident frame down to the host before the logger
  // ever saw it, and the encoder would take its staged host path instead of the zero-copy one
  // this graph exists to exercise.
  template <FrameSource kSource>
  void attach_generator() {
    auto generator = op<ImageGeneratorOp<kSource>>("generator");
    logger_.attach<holoscan::schema::ImageT>(
        *this, "camera", generator->image_out,
        image_schema::ImageTensorFieldContract(kWidth, kHeight, EncodingOf(kSource),
                                               IsDeviceResident(kSource)
                                                   ? holoscan::MemoryKind::kCudaDevice
                                                   : holoscan::MemoryKind::kHost),
        TopicLogOptions{.encode = encode_});
  }

  HsdkMcapLogger& logger_;
  VideoEncodeOptions encode_;
  FrameSource source_;
};

// One decoded pixel, read back as RGB.
struct DecodedPixel {
  int red = 0;
  int green = 0;
  int blue = 0;
};

// Converts one NVDEC output frame (NV12, device memory, owned by the decoder) to RGB on the GPU
// and reads a single pixel back to the host. Returns nullopt if any CUDA/NPP step fails.
//
// The conversion runs in the decoder's own CUDA context, because that is the context the NV12
// frame pointer belongs to.
std::optional<DecodedPixel> ReadDecodedPixel(nv_video_decoder::NVVideoDecoder& decoder,
                                             const std::uint8_t* nv12, int x, int y) {
  const int width = decoder.get_width();
  const int height = decoder.get_height();
  const int pitch = decoder.get_pitch();
  if (width <= 0 || height <= 0 || pitch <= 0 || x >= width || y >= height) {
    return std::nullopt;
  }
  if (cuCtxPushCurrent(decoder.get_cuda_context()) != CUDA_SUCCESS) {
    return std::nullopt;
  }

  std::optional<DecodedPixel> result;
  const int rgb_step = width * 3;
  CUdeviceptr rgb = 0;
  if (cuMemAlloc(&rgb, static_cast<std::size_t>(height) * rgb_step) == CUDA_SUCCESS) {
    // NV12 is two planes: full-resolution luma, then interleaved half-resolution chroma.
    const Npp8u* planes[2] = {nv12, nv12 + static_cast<std::size_t>(pitch) * height};
    const NppiSize roi = {width, height};
    const NppStreamContext npp_context{};
    if (nppiNV12ToRGB_8u_P2C3R_Ctx(planes, pitch, reinterpret_cast<Npp8u*>(rgb), rgb_step, roi,
                                   npp_context) == NPP_SUCCESS) {
      std::array<std::uint8_t, 3> pixel{};
      const CUdeviceptr offset = rgb + static_cast<std::size_t>(y) * rgb_step + (x * 3);
      if (cuMemcpyDtoH(pixel.data(), offset, pixel.size()) == CUDA_SUCCESS) {
        result = DecodedPixel{.red = pixel[0], .green = pixel[1], .blue = pixel[2]};
      }
    }
    cuMemFree(rgb);
  }

  cuCtxPopCurrent(nullptr);
  return result;
}

// A flat RGB8 frame of one colour in host memory, kept alive by the Tensor itself.
std::shared_ptr<holoscan::Tensor> MakeHostRgbTensor(std::uint8_t red, std::uint8_t green,
                                                    std::uint8_t blue) {
  auto storage = std::make_shared<std::vector<std::uint8_t>>(FrameBytesOf(FrameSource::kHostRgb8));
  for (std::size_t i = 0; i < storage->size(); i += 3) {
    (*storage)[i] = red;
    (*storage)[i + 1] = green;
    (*storage)[i + 2] = blue;
  }
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(storage->size())};
  auto tensor = holoscan::make_tensor(
      shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt, holoscan::MemoryStorageType::kSystem,
      storage->data(),
      [storage](void*) -> holoscan::expected<void, holoscan::Error> { return {}; });
  if (!tensor) {
    return nullptr;
  }
  return std::make_shared<holoscan::Tensor>(std::move(*tensor));
}

// A flat (kRed, kGreen, kBlue) RGB8 frame allocated in a context of its own on device 0, rather
// than in the primary context the encoder holds. Returns nullptr if a second context cannot be
// created. The context outlives the tensor and is destroyed with it, which frees the allocation.
std::shared_ptr<holoscan::Tensor> MakeForeignContextRgbTensor() {
  if (cuInit(0) != CUDA_SUCCESS) {
    return nullptr;
  }
  CUdevice device = 0;
  CUcontext context = nullptr;
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
      cuCtxCreate(&context, nullptr, 0, device) != CUDA_SUCCESS) {
    return nullptr;
  }
  // cuCtxCreate leaves the new context current, so the allocation below lands inside it.
  std::vector<std::uint8_t> host_pixels(FrameBytesOf(FrameSource::kHostRgb8));
  for (std::size_t i = 0; i < host_pixels.size(); i += 3) {
    host_pixels[i] = kRed;
    host_pixels[i + 1] = kGreen;
    host_pixels[i + 2] = kBlue;
  }
  CUdeviceptr pixels = 0;
  if (cuMemAlloc(&pixels, host_pixels.size()) != CUDA_SUCCESS ||
      cuMemcpyHtoD(pixels, host_pixels.data(), host_pixels.size()) != CUDA_SUCCESS) {
    cuCtxPopCurrent(nullptr);
    cuCtxDestroy(context);
    return nullptr;
  }
  cuCtxPopCurrent(nullptr);

  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(host_pixels.size())};
  auto tensor =
      holoscan::make_tensor(shape, DLDataType{kDLUInt, 8U, 1U}, std::nullopt,
                            holoscan::MemoryStorageType::kDevice, reinterpret_cast<void*>(pixels),
                            [context](void*) -> holoscan::expected<void, holoscan::Error> {
                              cuCtxDestroy(context);
                              return {};
                            });
  if (!tensor) {
    cuCtxDestroy(context);
    return nullptr;
  }
  return std::make_shared<holoscan::Tensor>(std::move(*tensor));
}

class EncodedVideoIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override { GeneratorTriggerProbe::reset(); }

  // Runs the encoding graph to completion, writing exactly kMessageCount frames to
  // `output_path` and closing the file before returning. Uses ASSERT_* internally, so callers
  // must check ::testing::Test::HasFatalFailure() (gtest cannot return from the test body out
  // of a helper).
  static void RunGraph(const std::filesystem::path& output_path,
                       const VideoEncodeOptions& encode = VideoEncodeOptions{},
                       FrameSource source = FrameSource::kHostRgb8) {
    HsdkMcapLogger logger(
        HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
    EncodedVideoTestGraph graph(logger, encode, source);
    holoscan::CompileOptions compile_options;
    if (IsDeviceResident(source)) {
      // The compiler refuses to assume device 0 for a Tensor-bearing output port -- it rejects
      // an implicit choice (IMPLICIT_DEVICE_ZERO_REJECTED) rather than guessing wrong -- so a
      // device-resident generator has to name its device here.
      compile_options.deployment.bind_tensor_output_device(
          holoscan::TensorOutputDevicePlacement{.operator_path = "generator",
                                                .output_port = "image_out",
                                                .device = holoscan::DeviceId{0}});
    }
    const holoscan::ExecutionPlan plan = holoscan::compile(graph, compile_options);
    if (!plan.ok()) {
      // The diagnostic tokens are the only way to tell a misspelled operator_path from a real
      // wiring problem, so print them rather than only asserting plan.ok().
      for (const auto& diagnostic : plan.diagnostics()) {
        ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
      }
    }
    ASSERT_TRUE(plan.ok());

    logger.start(plan.endpoint_catalog());
    auto run_session = holoscan::run_async(plan);

    auto sender = GeneratorTriggerProbe::wait_for_sender();
    ASSERT_TRUE(sender.has_value()) << "the generator never published its notification sender";
    for (int i = 0; i < kMessageCount; ++i) {
      auto posted = post_with_retry(*sender);
      ASSERT_TRUE(posted) << holoscan::runtime_error_message(posted.error());
      std::this_thread::sleep_for(kNotificationPeriod);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    run_session.request_stop();
    run_session.wait();
    logger.shutdown();
  }
};

TEST_F(EncodedVideoIntegrationTest, WritesFramesAnyNvdecConsumerCanDecode) {
  if (!FrameEncoder::hardware_available(0) || !nv_video_decoder::NVVideoDecoder::is_available()) {
    GTEST_SKIP() << "This host has no usable NVENC/NVDEC pair.";
  }

  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_encoded_video_integration_test.mcap");
  RunGraph(output_path);
  ASSERT_FALSE(HasFatalFailure());

  mcap::McapReader reader;
  const auto status = reader.open(output_path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  int packet_count = 0;
  int decoded_count = 0;
  nv_video_decoder::NVVideoDecoder decoder("h264", 0);
  ASSERT_TRUE(decoder.initialize());

  for (const auto& view : ReadInLogTimeOrder(reader)) {
    // The file also carries the camera/metadata sidecar; this test is only about the video.
    if (view.channel->topic != "camera") {
      continue;
    }
    // The exact pair a replayer dispatches on.
    EXPECT_EQ(view.channel->messageEncoding, "protobuf");
    ASSERT_NE(view.schema, nullptr);
    EXPECT_EQ(view.schema->name, "foxglove.CompressedVideo");

    const auto parsed = ReadCompressedVideo(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(view.message.data), view.message.dataSize));
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->format, "h264");  // Lowercase, as Foxglove expects, not "H264".
    ASSERT_FALSE(parsed->data.empty());

    std::vector<std::uint8_t*> frame_ptrs;
    const int decoded = decoder.decode_frame(parsed->data.data(), parsed->data.size(), frame_ptrs);
    // Negative means the decoder rejected the bitstream outright -- the failure this whole test
    // exists to catch. Zero is not a failure: a packet that is not a whole picture
    // produces no frame yet, so the counts are only settled after the flush below.
    ASSERT_GE(decoded, 0) << "packet " << packet_count << " was rejected by the decoder";
    decoded_count += decoded;
    ++packet_count;
  }

  // An empty packet is the end-of-stream marker; it releases anything the parser is still
  // holding, which is nothing when every packet was a whole picture.
  std::vector<std::uint8_t*> tail_ptrs;
  const int flushed = decoder.decode_frame(nullptr, 0, tail_ptrs);
  ASSERT_GE(flushed, 0) << "the decoder failed while draining";
  decoded_count += flushed;

  EXPECT_EQ(packet_count, kMessageCount);
  // Every frame handed to the encoder comes back out of the decoder: none was silently dropped,
  // truncated, or merged.
  EXPECT_EQ(decoded_count, kMessageCount);
}

// Two encoded topics at once, each with its own encoder session and its own poll thread, so they
// encode concurrently. Exact per-topic frame counts are what would expose contention between them.
TEST_F(EncodedVideoIntegrationTest, RecordsTwoEncodedTopicsAtOnce) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }

  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_two_encoded_topics_test.mcap");

  {
    HsdkMcapLogger logger(
        HsdkMcapLoggerOptions{.output_path = output_path, .timestamp_output_path = false});
    TwoEncodedTopicsGraph graph(logger);
    const holoscan::ExecutionPlan plan = holoscan::compile(graph);
    for (const auto& diagnostic : plan.diagnostics()) {
      ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
    }
    ASSERT_TRUE(plan.ok());

    logger.start(plan.endpoint_catalog());
    auto run_session = holoscan::run_async(plan);

    auto senders = GeneratorTriggerProbe::wait_for_senders(2);
    ASSERT_TRUE(senders.has_value()) << "both generators never published a notification sender";
    for (int i = 0; i < kMessageCount; ++i) {
      for (auto& sender : *senders) {
        auto posted = post_with_retry(sender);
        ASSERT_TRUE(posted) << holoscan::runtime_error_message(posted.error());
      }
      std::this_thread::sleep_for(kNotificationPeriod);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    run_session.request_stop();
    run_session.wait();
    logger.shutdown();
  }

  mcap::McapReader reader;
  const auto status = reader.open(output_path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  int first_topic_frames = 0;
  int second_topic_frames = 0;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    if (view.schema == nullptr || view.schema->name != "foxglove.CompressedVideo") {
      continue;  // The two metadata sidecars share this file.
    }
    if (view.channel->topic == "camera") {
      ++first_topic_frames;
    } else if (view.channel->topic == "camera_two") {
      ++second_topic_frames;
    }
  }

  // Every frame, not merely some: a device that refused the second session leaves one of these
  // at zero, and two encoders contending on one device -- for encoder sessions, for the default
  // stream their conversions share, or for the sink they both write to -- shows up as a
  // shortfall rather than a total loss, which a "greater than zero" check would wave through.
  EXPECT_EQ(first_topic_frames, kMessageCount) << "the first encoded topic lost frames";
  EXPECT_EQ(second_topic_frames, kMessageCount) << "the second encoded topic lost frames";

  // Each channel's own messages sit in the file in the order they were received: one thread per
  // topic receives, encodes and writes one frame at a time, so a topic's frames reach the encoder
  // in capture order. The channels may interleave however they land, so only each channel against
  // itself is checked, never one against another; reading across topics in time order is what
  // ReadInLogTimeOrder is for.
  std::map<std::string, std::uint64_t> last_log_time;
  for (const auto& view : ReadInFileOrder(reader)) {
    const auto [position, first] =
        last_log_time.try_emplace(view.channel->topic, view.message.logTime);
    if (!first) {
      EXPECT_GE(view.message.logTime, position->second)
          << "channel '" << view.channel->topic << "' has a message out of order in the file";
      position->second = view.message.logTime;
    }
  }
  EXPECT_EQ(last_log_time.size(), 4U) << "both video channels and both sidecars";
}

// The sidecar is what keeps an encoded recording from silently losing capture_timestamp_ns,
// exposure_time_ns, gain, color_space and the original encoding -- none of which
// foxglove.CompressedVideo has a field for.
TEST_F(EncodedVideoIntegrationTest, WritesAMetadataSidecarAlongsideTheVideo) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }

  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_encoded_video_sidecar_test.mcap");
  RunGraph(output_path);
  ASSERT_FALSE(HasFatalFailure());

  mcap::McapReader reader;
  const auto status = reader.open(output_path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  int video_count = 0;
  int metadata_count = 0;
  std::vector<std::uint64_t> video_times;
  std::vector<std::uint64_t> metadata_times;
  std::vector<std::int64_t> capture_timestamps;
  for (const auto& view : ReadInLogTimeOrder(reader)) {
    ASSERT_NE(view.schema, nullptr);
    if (view.channel->topic == "camera") {
      EXPECT_EQ(view.schema->name, "foxglove.CompressedVideo");
      video_times.push_back(view.message.logTime);
      ++video_count;
    } else if (view.channel->topic == "camera/metadata") {
      EXPECT_EQ(view.channel->messageEncoding, "flatbuffer");
      EXPECT_EQ(view.schema->name, "holoscan.schema.Image");
      // Decodable directly as the channel's own schema -- no unwrapping step, unlike
      // TensorFrame's nested opaque metadata bytes. Verify before GetRoot(): GetRoot() never
      // returns null, so a null check alone would prove nothing.
      const auto* data = reinterpret_cast<const std::uint8_t*>(view.message.data);
      flatbuffers::Verifier verifier(data, view.message.dataSize);
      ASSERT_TRUE(verifier.VerifyBuffer<holoscan::schema::Image>(nullptr));
      const auto* image = flatbuffers::GetRoot<holoscan::schema::Image>(data);
      EXPECT_EQ(image->encoding(), holoscan::schema::ImageEncoding_RGB8);
      EXPECT_EQ(image->width(), static_cast<std::int32_t>(kWidth));
      EXPECT_EQ(image->height(), static_cast<std::int32_t>(kHeight));
      // Everything else the producer stated, which the video channel cannot hold: the sidecar is
      // the received metadata passed through untouched, so each field has to come out as it went
      // in.
      EXPECT_EQ(image->color_space(), holoscan::schema::ColorSpace_REC709);
      EXPECT_EQ(image->significant_bits(), kSignificantBits);
      EXPECT_EQ(image->roi_offset_x(), kRoiOffsetX);
      EXPECT_EQ(image->roi_offset_y(), kRoiOffsetY);
      EXPECT_EQ(image->exposure_time_ns(), kExposureTimeNs);
      EXPECT_EQ(image->gain(), kGain);
      const auto* header = image->header();
      ASSERT_NE(header, nullptr)
          << "the sidecar lost the header, and with it the capture timestamp";
      EXPECT_EQ(header->frame_id()->string_view(), kFrameId);
      capture_timestamps.push_back(header->capture_timestamp_ns());
      metadata_times.push_back(view.message.logTime);
      ++metadata_count;
    }
  }
  EXPECT_EQ(video_count, kMessageCount);
  EXPECT_EQ(metadata_count, kMessageCount);
  // Identical log_time on both channels is what lets a replayer join them frame for frame.
  EXPECT_EQ(video_times, metadata_times);
  // Every frame's own capture timestamp, in capture order: none lost, none repeated, none taken
  // from another frame. The timestamps are the ones the generator stamped, one interval apart.
  std::vector<std::int64_t> expected_timestamps;
  for (int frame = 0; frame < kMessageCount; ++frame) {
    expected_timestamps.push_back(kFirstCaptureTimestampNs + frame * kCaptureIntervalNs);
  }
  EXPECT_EQ(capture_timestamps, expected_timestamps);
}

// Staging RGB into the encoder's four-byte pixels has to exchange red and blue, because the
// encoder's input buffer is byte-ordered B, G, R, A. Getting that backwards would still produce
// a perfectly valid, perfectly decodable H.264 stream -- every other assertion in this file
// would still pass -- and the only symptom would be that every recorded frame has its red and
// blue swapped, visible to a human watching the video and to nothing else. This decodes a
// recorded frame back to RGB and checks the channels came home in the order they left.
//
// Run once per FrameSource, because the encoder reaches that swap by a different route for each
// and a wrong pointer, step or argument order on any one of them is invisible to every other
// assertion in this file: a conversion that writes the wrong buffer, or none, still leaves the
// encoder a well-formed frame to encode and still yields a decodable stream. Only comparing the
// decoded colour against the colour that went in can tell those routes apart.
class EncodedVideoChannelOrderTest : public EncodedVideoIntegrationTest,
                                     public ::testing::WithParamInterface<FrameSource> {};

INSTANTIATE_TEST_SUITE_P(AllFrameSources, EncodedVideoChannelOrderTest,
                         ::testing::Values(FrameSource::kHostRgb8, FrameSource::kHostRgba8,
                                           FrameSource::kDeviceRgb8, FrameSource::kDeviceRgba8),
                         [](const ::testing::TestParamInfo<FrameSource>& info) {
                           return NameOf(info.param);
                         });

TEST_P(EncodedVideoChannelOrderTest, PreservesChannelOrderThroughTheRoundTrip) {
  if (!FrameEncoder::hardware_available(0) || !nv_video_decoder::NVVideoDecoder::is_available()) {
    GTEST_SKIP() << "This host has no usable NVENC/NVDEC pair.";
  }

  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_encoded_video_color_test.mcap");
  const std::uint64_t read_in_place_before = FrameEncoder::frames_read_in_place();
  RunGraph(output_path, VideoEncodeOptions{}, GetParam());
  ASSERT_FALSE(HasFatalFailure());

  // Which staging path the frames took is invisible in the recording -- both produce the same
  // picture -- so without this a guard that had quietly begun staging every frame would leave
  // the colour checks below passing while testing nothing they were written to test.
  const std::uint64_t read_in_place = FrameEncoder::frames_read_in_place() - read_in_place_before;
  if (IsDeviceResident(GetParam())) {
    EXPECT_GT(read_in_place, 0U) << "device frames were staged instead of read where they lay";
  } else {
    EXPECT_EQ(read_in_place, 0U) << "a host frame cannot be read where it lies";
  }

  mcap::McapReader reader;
  const auto status = reader.open(output_path.string());
  ASSERT_TRUE(status.ok()) << status.message;

  nv_video_decoder::NVVideoDecoder decoder("h264", 0);
  ASSERT_TRUE(decoder.initialize());

  int checked = 0;
  // This is a lossy round trip through 4:2:0 chroma subsampling and a YUV colour matrix, so the
  // tolerance is deliberately wide: the claim under test is "red is still red and blue is still
  // blue", not fidelity. A swap would land red near kBlue and blue near kRed, 150 counts away --
  // far outside this band.
  constexpr int kTolerance = 40;
  const auto check_frames = [&](const std::vector<std::uint8_t*>& frame_ptrs) {
    for (const std::uint8_t* frame : frame_ptrs) {
      ASSERT_NE(frame, nullptr);
      // Centre of the frame: the source colour is flat, but sampling away from the edges keeps
      // the check clear of any border artifacts the codec may introduce.
      const auto pixel = ReadDecodedPixel(decoder, frame, static_cast<int>(kWidth) / 2,
                                          static_cast<int>(kHeight) / 2);
      ASSERT_TRUE(pixel.has_value());
      EXPECT_NEAR(pixel->red, kRed, kTolerance) << "frame " << checked;
      EXPECT_NEAR(pixel->green, kGreen, kTolerance) << "frame " << checked;
      EXPECT_NEAR(pixel->blue, kBlue, kTolerance) << "frame " << checked;
      // Redundant with the bands above on today's numbers, and deliberately so: it is the
      // assertion that stays meaningful if the source colour is ever changed.
      EXPECT_GT(pixel->red, pixel->green) << "frame " << checked;
      EXPECT_GT(pixel->green, pixel->blue) << "frame " << checked;
      ++checked;
    }
  };

  for (const auto& view : ReadInLogTimeOrder(reader)) {
    if (view.channel->topic != "camera") {
      continue;
    }
    const auto parsed = ReadCompressedVideo(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(view.message.data), view.message.dataSize));
    ASSERT_TRUE(parsed.has_value());
    ASSERT_FALSE(parsed->data.empty());

    // Each batch is read before the next decode_frame() call, which is when the decoder reuses
    // the GPU buffers these point into.
    std::vector<std::uint8_t*> frame_ptrs;
    ASSERT_GE(decoder.decode_frame(parsed->data.data(), parsed->data.size(), frame_ptrs), 0);
    check_frames(frame_ptrs);
    ASSERT_FALSE(HasFatalFailure());
  }

  // Drains anything still inside NVDEC's parser -- see the decode test above.
  std::vector<std::uint8_t*> tail_ptrs;
  ASSERT_GE(decoder.decode_frame(nullptr, 0, tail_ptrs), 0);
  check_frames(tail_ptrs);
  ASSERT_FALSE(HasFatalFailure());

  // Without this the checks above could pass by never running.
  EXPECT_EQ(checked, kMessageCount);
}

// A cuda_device this host does not have is the realistic way to configure a GPU that cannot
// encode -- a mistyped ordinal in a config file. start() has to refuse it, naming the topic and the
// device, rather than record raw tensors in place of the video that was asked for: that run would
// look healthy, and only opening the file would show it was not what was configured.
TEST_F(EncodedVideoIntegrationTest, StartThrowsWhenTheConfiguredDeviceCannotEncode) {
  if (!FrameEncoder::hardware_available(0)) {
    GTEST_SKIP() << "No usable NVENC device on this host.";
  }
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  // The first ordinal past the last real device, so it is absent whatever this host has.
  const auto missing_device = static_cast<std::uint32_t>(device_count);
  // Without this the refusal below could be caused by anything at all: device 0 encodes on this
  // host, so the configured device is the only thing that differs.
  ASSERT_FALSE(FrameEncoder::hardware_available(missing_device));

  ScopedTempDirectory temp_directory;
  const auto output_path = temp_directory.path("mcap_logger_encoded_video_unavailable_test.mcap");
  try {
    RunGraph(output_path, VideoEncodeOptions{.cuda_device = missing_device});
    FAIL() << "start() accepted a topic whose GPU cannot encode.";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("topic 'camera'"), std::string::npos) << message;
    EXPECT_NE(message.find("GPU " + std::to_string(missing_device)), std::string::npos) << message;
  }
}

// A frame allocated in a second context on the encoder's own device is reachable -- unified
// addressing makes one process's device allocations addressable from any context on that device
// -- so the encoder reads it where it lies like any other device frame. Both halves of that are
// checked: the counter says which path was taken, which no assertion on the recording could,
// and the decoded colour says the path produced the right picture.
//
// This lives here rather than beside the other FrameEncoder unit tests because deciding it
// needs a decoder, and only this target links one. Asserting a non-empty bitstream would not
// decide anything: a conversion that read the wrong address would still produce one.
TEST_F(EncodedVideoIntegrationTest, PreservesChannelOrderForAFrameFromAnotherContext) {
  if (!FrameEncoder::hardware_available(0) || !nv_video_decoder::NVVideoDecoder::is_available()) {
    GTEST_SKIP() << "This host has no usable NVENC/NVDEC pair.";
  }
  const auto tensor = MakeForeignContextRgbTensor();
  ASSERT_NE(tensor, nullptr);
  ASSERT_EQ(tensor->device().device_type, kDLCUDA);

  const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});
  const std::uint64_t read_in_place_before = FrameEncoder::frames_read_in_place();
  const auto encoded =
      encoder->encode(*tensor, kWidth, kHeight, holoscan::schema::ImageEncoding_RGB8);
  ASSERT_TRUE(encoded.has_value());
  ASSERT_FALSE(encoded->empty());
  EXPECT_EQ(FrameEncoder::frames_read_in_place() - read_in_place_before, 1U)
      << "a frame from another context on the encoder's device is reachable and must not be "
         "staged";

  nv_video_decoder::NVVideoDecoder decoder("h264", 0);
  ASSERT_TRUE(decoder.initialize());
  std::vector<std::uint8_t*> frame_ptrs;
  ASSERT_GE(decoder.decode_frame(reinterpret_cast<const std::uint8_t*>(encoded->data()),
                                 encoded->size(), frame_ptrs),
            0);
  // Each packet is flagged end-of-picture, so the frame comes back from the call above; the
  // end-of-stream marker below only drains whatever a packet that was not a whole picture left.
  std::vector<std::uint8_t*> tail_ptrs;
  ASSERT_GE(decoder.decode_frame(nullptr, 0, tail_ptrs), 0);
  frame_ptrs.insert(frame_ptrs.end(), tail_ptrs.begin(), tail_ptrs.end());
  ASSERT_FALSE(frame_ptrs.empty());

  const auto pixel = ReadDecodedPixel(decoder, frame_ptrs.front(), static_cast<int>(kWidth) / 2,
                                      static_cast<int>(kHeight) / 2);
  ASSERT_TRUE(pixel.has_value());
  constexpr int kTolerance = 40;
  EXPECT_NEAR(pixel->red, kRed, kTolerance);
  EXPECT_NEAR(pixel->green, kGreen, kTolerance);
  EXPECT_NEAR(pixel->blue, kBlue, kTolerance);
}

// One encoder encodes frame after frame through staging buffers it keeps for the session, so each
// frame's pixels must replace the last one's rather than show through. A recording of identical
// frames cannot tell: stale bytes would be the same bytes. Red-dominant and blue-dominant frames
// alternate here, and each decoded frame has to be its own colour, in order -- a buffer that kept
// the previous frame, or an encoder that was handed the wrong one, would show up as a frame of the
// other colour.
TEST_F(EncodedVideoIntegrationTest, EachFrameThroughOneEncoderKeepsItsOwnPixels) {
  if (!FrameEncoder::hardware_available(0) || !nv_video_decoder::NVVideoDecoder::is_available()) {
    GTEST_SKIP() << "This host has no usable NVENC/NVDEC pair.";
  }
  constexpr int kFrames = 8;
  const auto red_frame = MakeHostRgbTensor(kRed, kGreen, kBlue);
  const auto blue_frame = MakeHostRgbTensor(kBlue, kGreen, kRed);
  ASSERT_NE(red_frame, nullptr);
  ASSERT_NE(blue_frame, nullptr);

  const auto encoder = std::make_shared<FrameEncoder>(VideoEncodeOptions{});
  nv_video_decoder::NVVideoDecoder decoder("h264", 0);
  ASSERT_TRUE(decoder.initialize());

  constexpr int kTolerance = 40;
  int checked = 0;
  const auto check_frames = [&](const std::vector<std::uint8_t*>& frame_ptrs) {
    for (const std::uint8_t* frame : frame_ptrs) {
      ASSERT_NE(frame, nullptr);
      const auto pixel = ReadDecodedPixel(decoder, frame, static_cast<int>(kWidth) / 2,
                                          static_cast<int>(kHeight) / 2);
      ASSERT_TRUE(pixel.has_value());
      const bool expect_red_frame = checked % 2 == 0;
      EXPECT_NEAR(pixel->red, expect_red_frame ? kRed : kBlue, kTolerance) << "frame " << checked;
      EXPECT_NEAR(pixel->blue, expect_red_frame ? kBlue : kRed, kTolerance) << "frame " << checked;
      ++checked;
    }
  };

  for (int frame = 0; frame < kFrames; ++frame) {
    const auto& source = frame % 2 == 0 ? red_frame : blue_frame;
    const auto encoded =
        encoder->encode(*source, kWidth, kHeight, holoscan::schema::ImageEncoding_RGB8);
    ASSERT_TRUE(encoded.has_value()) << "frame " << frame;
    // Each batch is read before the next decode_frame() call, which is when the decoder reuses
    // the GPU buffers these point into.
    std::vector<std::uint8_t*> frame_ptrs;
    ASSERT_GE(decoder.decode_frame(reinterpret_cast<const std::uint8_t*>(encoded->data()),
                                   encoded->size(), frame_ptrs),
              0);
    check_frames(frame_ptrs);
    ASSERT_FALSE(HasFatalFailure());
  }
  // Drains the pictures still inside NVDEC's parser.
  std::vector<std::uint8_t*> tail_ptrs;
  ASSERT_GE(decoder.decode_frame(nullptr, 0, tail_ptrs), 0);
  check_frames(tail_ptrs);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(checked, kFrames) << "every frame came back out of the decoder";
}

}  // namespace
}  // namespace isaac::os::mcap_logger
