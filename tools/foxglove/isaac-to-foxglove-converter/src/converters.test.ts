// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as flatbuffers from "flatbuffers";

import {
  activateConverters,
  convertCameraInfoCpuToFoxgloveCameraCalibration,
  convertPoseCpuToFoxglovePoseInFrame,
  convertTensorFrameImageToFoxgloveRawImage,
  convertTransformCpuToFoxgloveFrameTransform,
  convertTwistCpuToFoxgloveSceneUpdate,
} from "./converters";
import { Header as HoloscanSchemaHeader } from "./generated/holoscan-schema/holoscan/schema/header";
import { Image as HoloscanSchemaImage } from "./generated/holoscan-schema/holoscan/schema/image";
import { ImageEncoding } from "./generated/holoscan-schema/holoscan/schema/image-encoding";

describe("activateConverters", () => {
  it("registers the tensor-frame converter for the current MCAP schema", () => {
    const registrations: Array<{ fromSchemaName: string; toSchemaName: string }> = [];
    const extensionContext = {
      registerMessageConverter(registration: { fromSchemaName: string; toSchemaName: string }) {
        registrations.push(registration);
      },
    } as unknown as Parameters<typeof activateConverters>[0];

    activateConverters(extensionContext);

    expect(registrations).toContainEqual(
      expect.objectContaining({
        fromSchemaName: "isaac.mcap_logger.TensorFrame",
        toSchemaName: "foxglove.RawImage",
      }),
    );
  });
});

describe("convertPoseCpuToFoxglovePoseInFrame", () => {
  it("maps position/orientation/header fields directly, with no reordering", () => {
    const result = convertPoseCpuToFoxglovePoseInFrame({
      position: { x: 1, y: 2, z: 3 },
      orientation: { x: 0.1, y: 0.2, z: 0.3, w: 0.9 },
      header: { frame_id: "world", capture_timestamp_ns: BigInt(5_000_000_000) },
    });

    expect(result).toEqual({
      timestamp: { sec: 5, nsec: 0 },
      frame_id: "world",
      pose: {
        position: { x: 1, y: 2, z: 3 },
        orientation: { x: 0.1, y: 0.2, z: 0.3, w: 0.9 },
      },
    });
  });

  it("returns undefined when position is missing", () => {
    expect(
      convertPoseCpuToFoxglovePoseInFrame({ orientation: { x: 0, y: 0, z: 0, w: 1 } }),
    ).toBeUndefined();
  });

  it("returns undefined when orientation is missing", () => {
    expect(convertPoseCpuToFoxglovePoseInFrame({ position: { x: 0, y: 0, z: 0 } })).toBeUndefined();
  });
});

describe("convertTwistCpuToFoxgloveSceneUpdate", () => {
  it("produces one arrow entity oriented along the linear velocity vector", () => {
    const result = convertTwistCpuToFoxgloveSceneUpdate({
      linear: { x: 1, y: 0, z: 0 },
      angular: { x: 0, y: 0, z: 0 },
      header: { frame_id: "base_link", capture_timestamp_ns: BigInt(0) },
    }) as { entities: Array<{ frame_id: string; arrows: unknown[] }> };

    expect(result.entities).toHaveLength(1);
    expect(result.entities[0]?.frame_id).toBe("base_link");
    expect(result.entities[0]?.arrows).toHaveLength(1);
  });

  it("returns undefined when velocity magnitude is too small to visualize", () => {
    expect(
      convertTwistCpuToFoxgloveSceneUpdate({
        linear: { x: 0.0001, y: 0, z: 0 },
        header: {},
      }),
    ).toBeUndefined();
  });

  it("returns undefined when linear velocity is missing", () => {
    expect(convertTwistCpuToFoxgloveSceneUpdate({ header: {} })).toBeUndefined();
  });
});

describe("convertTransformCpuToFoxgloveFrameTransform", () => {
  it("maps translation/rotation directly and reads child_frame_id from the top-level field", () => {
    const result = convertTransformCpuToFoxgloveFrameTransform({
      translation: { x: 1, y: 2, z: 3 },
      rotation: { x: 0, y: 0, z: 0, w: 1 },
      header: { frame_id: "world", capture_timestamp_ns: BigInt(0) },
      child_frame_id: "camera_link",
    });

    expect(result).toEqual({
      timestamp: { sec: 0, nsec: 0 },
      parent_frame_id: "world",
      child_frame_id: "camera_link",
      translation: { x: 1, y: 2, z: 3 },
      rotation: { x: 0, y: 0, z: 0, w: 1 },
    });
  });

  it("returns undefined when translation is missing", () => {
    expect(
      convertTransformCpuToFoxgloveFrameTransform({ rotation: { x: 0, y: 0, z: 0, w: 1 } }),
    ).toBeUndefined();
  });

  it("returns undefined when rotation is missing", () => {
    expect(
      convertTransformCpuToFoxgloveFrameTransform({ translation: { x: 0, y: 0, z: 0 } }),
    ).toBeUndefined();
  });
});

describe("convertCameraInfoCpuToFoxgloveCameraCalibration", () => {
  it("maps K/R/P matrices to CameraCalibration's D/K/R/P fields with no reshaping", () => {
    const k = Array.from({ length: 9 }, (_, i) => i);
    const r = Array.from({ length: 9 }, (_, i) => i + 100);
    const p = Array.from({ length: 12 }, (_, i) => i + 200);

    const result = convertCameraInfoCpuToFoxgloveCameraCalibration({
      header: { frame_id: "camera", capture_timestamp_ns: BigInt(0) },
      height: 480,
      width: 640,
      distortion_model: "plumb_bob",
      d: [0.1, 0.2, 0.3],
      k: { m: k },
      r: { m: r },
      p: { m: p },
    });

    expect(result).toEqual({
      timestamp: { sec: 0, nsec: 0 },
      frame_id: "camera",
      height: 480,
      width: 640,
      distortion_model: "plumb_bob",
      D: [0.1, 0.2, 0.3],
      K: k,
      R: r,
      P: p,
    });
  });

  it("returns undefined when K, R, or P is missing", () => {
    expect(
      convertCameraInfoCpuToFoxgloveCameraCalibration({
        header: {},
        height: 480,
        width: 640,
        k: { m: Array(9).fill(0) },
        r: { m: Array(9).fill(0) },
        // p missing
      }),
    ).toBeUndefined();
  });

  it("returns undefined when height/width are missing", () => {
    expect(
      convertCameraInfoCpuToFoxgloveCameraCalibration({
        header: {},
        k: { m: Array(9).fill(0) },
        r: { m: Array(9).fill(0) },
        p: { m: Array(12).fill(0) },
      }),
    ).toBeUndefined();
  });

  it("returns undefined when height or width is non-positive", () => {
    expect(
      convertCameraInfoCpuToFoxgloveCameraCalibration({
        header: {},
        height: 0,
        width: 640,
        k: { m: Array(9).fill(0) },
        r: { m: Array(9).fill(0) },
        p: { m: Array(12).fill(0) },
      }),
    ).toBeUndefined();
    expect(
      convertCameraInfoCpuToFoxgloveCameraCalibration({
        header: {},
        height: 480,
        width: -1,
        k: { m: Array(9).fill(0) },
        r: { m: Array(9).fill(0) },
        p: { m: Array(12).fill(0) },
      }),
    ).toBeUndefined();
  });

  it("returns undefined when K, R, or P has the wrong length", () => {
    expect(
      convertCameraInfoCpuToFoxgloveCameraCalibration({
        header: {},
        height: 480,
        width: 640,
        k: { m: Array(8).fill(0) }, // wrong length: should be 9
        r: { m: Array(9).fill(0) },
        p: { m: Array(12).fill(0) },
      }),
    ).toBeUndefined();
    expect(
      convertCameraInfoCpuToFoxgloveCameraCalibration({
        header: {},
        height: 480,
        width: 640,
        k: { m: Array(9).fill(0) },
        r: { m: Array(9).fill(0) },
        p: { m: Array(11).fill(0) }, // wrong length: should be 12
      }),
    ).toBeUndefined();
  });
});

// Builds a real, serialized holoscan.schema.Image FlatBuffers message -- the same shape
// TensorFrame.metadata carries on the wire -- for use as a TensorFrame test fixture.
function buildImageMetadataBytes(options: {
  height: number;
  width: number;
  encoding: ImageEncoding;
  frameId?: string;
  captureTimestampNs?: bigint;
  omitHeader?: boolean;
}): Uint8Array {
  const builder = new flatbuffers.Builder();

  let headerOffset: number | undefined;
  if (options.omitHeader !== true) {
    const frameIdOffset = builder.createString(options.frameId ?? "");
    HoloscanSchemaHeader.startHeader(builder);
    HoloscanSchemaHeader.addFrameId(builder, frameIdOffset);
    HoloscanSchemaHeader.addCaptureTimestampNs(builder, options.captureTimestampNs ?? BigInt(0));
    headerOffset = HoloscanSchemaHeader.endHeader(builder);
  }

  HoloscanSchemaImage.startImage(builder);
  if (headerOffset != null) {
    HoloscanSchemaImage.addHeader(builder, headerOffset);
  }
  HoloscanSchemaImage.addHeight(builder, options.height);
  HoloscanSchemaImage.addWidth(builder, options.width);
  HoloscanSchemaImage.addEncoding(builder, options.encoding);
  const imageOffset = HoloscanSchemaImage.endImage(builder);

  HoloscanSchemaImage.finishImageBuffer(builder, imageOffset);
  return builder.asUint8Array();
}

describe("convertTensorFrameImageToFoxgloveRawImage", () => {
  it("decodes nested Image metadata and pairs it with tensors[0].data for RGB8", () => {
    const metadata = buildImageMetadataBytes({
      height: 2,
      width: 4,
      encoding: ImageEncoding.RGB8,
      frameId: "camera",
      captureTimestampNs: BigInt(3_000_000_000),
    });
    const pixelData = new Uint8Array(2 * 4 * 3).fill(7);

    const result = convertTensorFrameImageToFoxgloveRawImage({
      metadata_schema_name: "holoscan.schema.Image",
      metadata,
      tensors: [{ data: pixelData }],
    });

    expect(result).toEqual({
      timestamp: { sec: 3, nsec: 0 },
      frame_id: "camera",
      width: 4,
      height: 2,
      encoding: "rgb8",
      step: 12,
      data: pixelData,
    });
  });

  it("drops the message instead of defaulting to epoch when the nested Image has no header/timestamp", () => {
    const metadata = buildImageMetadataBytes({
      height: 2,
      width: 4,
      encoding: ImageEncoding.RGB8,
      omitHeader: true,
    });
    const pixelData = new Uint8Array(2 * 4 * 3).fill(7);

    expect(
      convertTensorFrameImageToFoxgloveRawImage({
        metadata_schema_name: "holoscan.schema.Image",
        metadata,
        tensors: [{ data: pixelData }],
      }),
    ).toBeUndefined();
  });

  it("returns undefined for a non-Image metadata_schema_name (lets another registration handle it)", () => {
    expect(
      convertTensorFrameImageToFoxgloveRawImage({
        metadata_schema_name: "some.other.Schema",
        metadata: new Uint8Array([1, 2, 3]),
        tensors: [{ data: new Uint8Array(1) }],
      }),
    ).toBeUndefined();
  });

  it("returns undefined when tensor data size doesn't match the decoded image dimensions", () => {
    const metadata = buildImageMetadataBytes({ height: 2, width: 4, encoding: ImageEncoding.RGB8 });

    expect(
      convertTensorFrameImageToFoxgloveRawImage({
        metadata_schema_name: "holoscan.schema.Image",
        metadata,
        tensors: [{ data: new Uint8Array(1) }], // way too short for 2x4 RGB8
      }),
    ).toBeUndefined();
  });

  it("returns undefined when tensors is empty", () => {
    const metadata = buildImageMetadataBytes({ height: 2, width: 4, encoding: ImageEncoding.RGB8 });

    expect(
      convertTensorFrameImageToFoxgloveRawImage({
        metadata_schema_name: "holoscan.schema.Image",
        metadata,
        tensors: [],
      }),
    ).toBeUndefined();
  });
});
