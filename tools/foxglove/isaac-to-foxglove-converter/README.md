# Isaac to Foxglove Converter Extension

This Foxglove extension converts Isaac OS messages to Foxglove schemas for visualization.

Supported conversions:

- `isaac.PoseCpu` to `foxglove.PoseInFrame`
- `isaac.TwistCpu` to `foxglove.SceneUpdate`
- `isaac.TransformCpu` to `foxglove.FrameTransform`
- `isaac.CameraInfoCpu` to `foxglove.CameraCalibration`
- Image-flavored `isaac.mcap_logger.TensorFrame` to `foxglove.RawImage`

Tensor frames with another `metadata_schema_name` remain available through Foxglove's generic
decoder. Add a separate converter registration when another tensor-bearing type needs a native
panel.

## Development

```bash
npm ci
npm run build
npm test
```

Use `npm run local-install` to install the extension locally or `npm run package` to create a
shareable `.foxe` package.

The generated Holoscan FlatBuffers bindings used by the tests are documented under
`src/generated/holoscan-schema/README.md`.
