// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Foxglove converters for Isaac OS flatbuffer schemas: the isaac.*Cpu geometry/camera types and the
// generic isaac.mcap_logger.TensorFrame tensor envelope.
//
// registerMessageConverter's toSchemaName is fixed per registration -- a converter cannot branch
// its output schema at runtime based on message content. TensorFrame is a single shared schema
// used by every tensor-bearing type (see tensor_frame.fbs), so type-specific rendering (e.g.
// routing an Image-flavored frame to foxglove.RawImage) means registering the same fromSchemaName
// once per target type, with each converter checking metadata_schema_name and returning undefined
// when it doesn't apply. Only one such registration (Image -> RawImage) exists today; adding a
// second tensor-bearing type's native rendering later means adding another registration here, not
// changing this one.
import { ExtensionContext } from "@foxglove/extension";
import * as flatbuffers from "flatbuffers";

import { convertTimestamp, imageEncodingMap, quaternionFromXAxisToDirection } from "./common";
import { Image as HoloscanSchemaImage } from "./generated/holoscan-schema/holoscan/schema/image";

interface Vector3dInput {
  x: number;
  y: number;
  z: number;
}

interface QuaterniondInput {
  x: number;
  y: number;
  z: number;
  w: number;
}

// holoscan.schema.Header, shared by every geometry/camera type below.
// It carries capture_timestamp_ns directly; `stamp` is a deprecated,
// no-longer-populated field kept only for old producers, so it is not read here.
interface HeaderInput {
  frame_id?: string;
  // Always bigint: this is what Foxglove's own generic flatbuffer decode hands back for an int64
  // field (see convertTimestamp's comment). Not number/string -- there is no producer here that
  // supplies either.
  capture_timestamp_ns?: bigint;
}

interface Matrix3dInput {
  m?: number[];
}

interface Matrix3x4dInput {
  m?: number[];
}

interface PoseCpuInput {
  position?: Vector3dInput;
  orientation?: QuaterniondInput;
  header?: HeaderInput;
}

// Schema converter to transform isaac::PoseCpu to foxglove::PoseInFrame.
export function convertPoseCpuToFoxglovePoseInFrame(inputMessage: unknown): unknown {
  const pose = inputMessage as PoseCpuInput;

  if (!pose.position) {
    console.debug("PoseCpu: missing position");
    return undefined;
  }
  if (!pose.orientation) {
    console.debug("PoseCpu: missing orientation");
    return undefined;
  }

  const timestamp = convertTimestamp(pose.header?.capture_timestamp_ns);
  if (!timestamp) {
    return undefined;
  }

  return {
    timestamp,
    frame_id: pose.header?.frame_id ?? "",
    pose: {
      position: { x: pose.position.x, y: pose.position.y, z: pose.position.z },
      orientation: {
        x: pose.orientation.x,
        y: pose.orientation.y,
        z: pose.orientation.z,
        w: pose.orientation.w,
      },
    },
  };
}

interface TwistCpuInput {
  linear?: Vector3dInput;
  angular?: Vector3dInput;
  header?: HeaderInput;
}

// Schema converter to transform isaac::TwistCpu to foxglove::SceneUpdate.
export function convertTwistCpuToFoxgloveSceneUpdate(inputMessage: unknown): unknown {
  const twist = inputMessage as TwistCpuInput;

  if (!twist.linear) {
    console.debug("TwistCpu: missing linear velocity");
    return undefined;
  }

  const lx = twist.linear.x;
  const ly = twist.linear.y;
  const lz = twist.linear.z;

  const velocityMag = Math.sqrt(lx * lx + ly * ly + lz * lz);
  if (velocityMag < 0.001) {
    // Velocity too small to visualize.
    return undefined;
  }

  const timestamp = convertTimestamp(twist.header?.capture_timestamp_ns);
  if (!timestamp) {
    return undefined;
  }
  const frameId = twist.header?.frame_id ?? "";

  const orientation = quaternionFromXAxisToDirection(lx, ly, lz);

  const arrowScale = 0.5;
  const shaftLength = Math.min(velocityMag * arrowScale, 2.0);

  const entity = {
    timestamp,
    frame_id: frameId,
    id: "twist_cpu_linear",
    lifetime: { sec: 0, nsec: 100_000_000 },
    frame_locked: true,
    arrows: [
      {
        pose: {
          position: { x: 0, y: 0, z: 0 },
          orientation,
        },
        shaft_length: shaftLength,
        shaft_diameter: 0.02,
        head_length: 0.06,
        head_diameter: 0.04,
        color: { r: 1.0, g: 0.5, b: 0.0, a: 1.0 },
      },
    ],
  };

  return { entities: [entity] };
}

interface TransformCpuInput {
  translation?: Vector3dInput;
  rotation?: QuaterniondInput;
  header?: HeaderInput;
  child_frame_id?: string;
}

// Schema converter to transform isaac::TransformCpu to foxglove::FrameTransform.
export function convertTransformCpuToFoxgloveFrameTransform(inputMessage: unknown): unknown {
  const transform = inputMessage as TransformCpuInput;

  if (!transform.translation) {
    console.debug("TransformCpu: missing translation");
    return undefined;
  }
  if (!transform.rotation) {
    console.debug("TransformCpu: missing rotation");
    return undefined;
  }

  const timestamp = convertTimestamp(transform.header?.capture_timestamp_ns);
  if (!timestamp) {
    return undefined;
  }

  return {
    timestamp,
    parent_frame_id: transform.header?.frame_id ?? "",
    child_frame_id: transform.child_frame_id ?? "",
    translation: {
      x: transform.translation.x,
      y: transform.translation.y,
      z: transform.translation.z,
    },
    rotation: {
      x: transform.rotation.x,
      y: transform.rotation.y,
      z: transform.rotation.z,
      w: transform.rotation.w,
    },
  };
}

interface CameraInfoCpuInput {
  header?: HeaderInput;
  height?: number;
  width?: number;
  distortion_model?: string;
  d?: number[];
  k?: Matrix3dInput;
  r?: Matrix3dInput;
  p?: Matrix3x4dInput;
  // binning_x/binning_y/roi_* are carried by isaac.CameraInfoCpu but have no equivalent field in
  // foxglove.CameraCalibration, so they are read from the input type above but never referenced
  // below -- not an oversight, foxglove.CameraCalibration simply has nowhere to put them.
}

// Schema converter to transform isaac::CameraInfoCpu to foxglove::CameraCalibration. K/R are
// holoscan.schema.Matrix3d (9 doubles, row-major); P is isaac.Matrix3x4d (12 doubles, row-major)
// -- both already in the layout foxglove.CameraCalibration expects, so no reshaping is needed.
export function convertCameraInfoCpuToFoxgloveCameraCalibration(inputMessage: unknown): unknown {
  const cameraInfo = inputMessage as CameraInfoCpuInput;

  if (
    typeof cameraInfo.height !== "number" ||
    typeof cameraInfo.width !== "number" ||
    cameraInfo.height <= 0 ||
    cameraInfo.width <= 0
  ) {
    console.debug("CameraInfoCpu: missing or non-positive height/width");
    return undefined;
  }
  if (
    !cameraInfo.k?.m ||
    !cameraInfo.r?.m ||
    !cameraInfo.p?.m ||
    cameraInfo.k.m.length !== 9 ||
    cameraInfo.r.m.length !== 9 ||
    cameraInfo.p.m.length !== 12
  ) {
    console.debug("CameraInfoCpu: missing or wrong-length K, R, or P matrix");
    return undefined;
  }

  const timestamp = convertTimestamp(cameraInfo.header?.capture_timestamp_ns);
  if (!timestamp) {
    return undefined;
  }

  return {
    timestamp,
    frame_id: cameraInfo.header?.frame_id ?? "",
    height: cameraInfo.height,
    width: cameraInfo.width,
    distortion_model: cameraInfo.distortion_model ?? "",
    D: cameraInfo.d ?? [],
    K: cameraInfo.k.m,
    R: cameraInfo.r.m,
    P: cameraInfo.p.m,
  };
}

interface TensorEntryInput {
  shape?: number[];
  dtype?: number;
  data?: Uint8Array;
}

interface TensorFrameInput {
  tensors?: TensorEntryInput[];
  metadata?: Uint8Array;
  metadata_schema_name?: string;
}

// Decodes the raw bytes of a TensorFrame.metadata field as a real holoscan.schema.Image FlatBuffers
// message, using flatc-generated bindings checked in under generated/holoscan-schema/. This is a
// SECOND, independent decode step from Foxglove's own generic decode of the outer TensorFrame
// message: Foxglove's built-in decoder is bound to one schema per MCAP channel (TensorFrame's own),
// so it hands back `metadata` as an opaque Uint8Array -- it has no way to know, or a mechanism to be
// told, that those particular bytes should be reinterpreted under a second, different schema named
// by the sibling metadata_schema_name field. That indirection only exists because TensorFrame is
// deliberately generic across arbitrary tensor-bearing types (see tensor_frame.fbs); decoding it is
// therefore necessarily this converter's job, not something a schema-driven generic decoder could
// ever do automatically -- today.
//
// TensorFrame now also carries metadata_binary_schema (T's compiled reflection schema, the same
// .bfbs bytes this function's own hand-written bindings were generated from) precisely so that
// can change: a future MR should replace this Image-specific function with one generic decoder
// that walks `metadata` via FlatBuffers' own runtime reflection against metadata_binary_schema
// (mirroring flatbuffers/reflection.h's GetAnyField*/Verify on the C++ side -- see
// tensor_topic_poller.h's comment where metadata_binary_schema is populated), working for any
// future tensor-bearing type with no new checked-in bindings or decode function required. This
// function, and the generated/holoscan-schema/ bindings it depends on, would then be deleted.
function decodeHoloscanSchemaImageMetadata(metadata: Uint8Array):
  | {
      height: number;
      width: number;
      encoding: number;
      frameId: string;
      captureTimestampNs: bigint | undefined;
    }
  | undefined {
  try {
    const byteBuffer = new flatbuffers.ByteBuffer(metadata);
    const image = HoloscanSchemaImage.getRootAsImage(byteBuffer);
    const header = image.header();
    return {
      height: image.height(),
      width: image.width(),
      encoding: image.encoding(),
      frameId: header?.frameId() ?? "",
      // No fallback here: a missing header/timestamp should surface via convertTimestamp's own
      // handling below (log + drop the message), not be silently masked before it gets there.
      captureTimestampNs: header?.captureTimestampNs(),
    };
  } catch (error) {
    console.debug("TensorFrame: failed to decode nested holoscan.schema.Image metadata", error);
    return undefined;
  }
}

// Schema converter to transform an Image-flavored isaac.mcap_logger.TensorFrame to
// foxglove::RawImage. Returns undefined for any TensorFrame whose metadata_schema_name isn't
// "holoscan.schema.Image" -- per the multi-registration pattern described in this file's header
// comment, a future second tensor-bearing type gets its own converter/registration pair, not a
// branch added here.
//
// Pixel bytes come from tensors[0].data (Foxglove's own generic decode of TensorFrame's own
// schema, already a plain Uint8Array) -- NOT from anything inside the decoded Image metadata.
// ImageTensorFieldContract (common/image_schema) transports Image's tensor field
// out-of-band, so the Image table's own `data` field is never populated on the wire.
export function convertTensorFrameImageToFoxgloveRawImage(inputMessage: unknown): unknown {
  const frame = inputMessage as TensorFrameInput;

  if (frame.metadata_schema_name !== "holoscan.schema.Image") {
    return undefined;
  }
  if (!frame.metadata) {
    console.debug("TensorFrame (Image): missing metadata bytes");
    return undefined;
  }
  const tensor = frame.tensors?.[0];
  if (!tensor?.data) {
    console.debug("TensorFrame (Image): missing tensor data");
    return undefined;
  }

  const image = decodeHoloscanSchemaImageMetadata(frame.metadata);
  if (!image) {
    return undefined;
  }

  if (image.width <= 0 || image.height <= 0) {
    console.debug(
      `TensorFrame (Image): invalid image dimensions: width=${image.width}, height=${image.height}`,
    );
    return undefined;
  }

  const encodingInfo = imageEncodingMap[image.encoding];
  if (!encodingInfo) {
    console.debug(`TensorFrame (Image): unsupported ImageEncoding value ${image.encoding}`);
    return undefined;
  }

  const expectedDataSize = image.width * image.height * encodingInfo.bytesPerPixel;
  if (tensor.data.length !== expectedDataSize) {
    console.debug(
      `TensorFrame (Image): data size mismatch. Expected ${expectedDataSize} bytes for ` +
        `${image.width}x${image.height} ${encodingInfo.format}, got ${tensor.data.length} bytes`,
    );
    return undefined;
  }

  const timestamp = convertTimestamp(image.captureTimestampNs);
  if (!timestamp) {
    return undefined;
  }

  return {
    timestamp,
    frame_id: image.frameId,
    width: image.width,
    height: image.height,
    encoding: encodingInfo.format,
    step: image.width * encodingInfo.bytesPerPixel,
    data: tensor.data,
  };
}

export function activateConverters(extensionContext: ExtensionContext): void {
  extensionContext.registerMessageConverter({
    fromSchemaName: "isaac.PoseCpu",
    toSchemaName: "foxglove.PoseInFrame",
    converter: convertPoseCpuToFoxglovePoseInFrame,
  });

  extensionContext.registerMessageConverter({
    fromSchemaName: "isaac.TwistCpu",
    toSchemaName: "foxglove.SceneUpdate",
    converter: convertTwistCpuToFoxgloveSceneUpdate,
  });

  extensionContext.registerMessageConverter({
    fromSchemaName: "isaac.TransformCpu",
    toSchemaName: "foxglove.FrameTransform",
    converter: convertTransformCpuToFoxgloveFrameTransform,
  });

  extensionContext.registerMessageConverter({
    fromSchemaName: "isaac.CameraInfoCpu",
    toSchemaName: "foxglove.CameraCalibration",
    converter: convertCameraInfoCpuToFoxgloveCameraCalibration,
  });

  extensionContext.registerMessageConverter({
    fromSchemaName: "isaac.mcap_logger.TensorFrame",
    toSchemaName: "foxglove.RawImage",
    converter: convertTensorFrameImageToFoxgloveRawImage,
  });
}
