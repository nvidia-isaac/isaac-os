# Changelog

## 0.0.3

- Rename the tensor-frame schema from `isaac.hsdk5.mcap_logger.TensorFrame` to
  `isaac.mcap_logger.TensorFrame` and its MCAP profile from `hsdk5_mcap_logger` to
  `isaac_mcap_logger`. The converter no longer matches recordings written with the old schema name.

## 0.0.2

- Add hsdk5 converters: `isaac.PoseCpu` → `foxglove.PoseInFrame`, `isaac.TwistCpu` →
  `foxglove.SceneUpdate`, `isaac.TransformCpu` → `foxglove.FrameTransform`, `isaac.CameraInfoCpu` →
  `foxglove.CameraCalibration`, and an Image-flavored `isaac.hsdk5.mcap_logger.TensorFrame` →
  `foxglove.RawImage`.
- Add the hsdk4 `isaac.*` schema and topic converters.
