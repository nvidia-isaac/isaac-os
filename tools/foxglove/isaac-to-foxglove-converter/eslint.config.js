// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @ts-check

const foxglove = require('@foxglove/eslint-plugin');
const globals = require('globals');
const tseslint = require('typescript-eslint');

module.exports = tseslint.config({
  // flatc-generated FlatBuffers TypeScript bindings -- not hand-written, not ours to reformat.
  // See src/generated/holoscan-schema/README.md for how they're regenerated.
  ignores: ['src/generated/**'],
}, {
  files: ['src/**/*.ts', 'src/**/*.tsx'],
  extends: [foxglove.configs.base, foxglove.configs.react, foxglove.configs.typescript],
  languageOptions: {
    globals: {
      ...globals.es2020,
      ...globals.browser,
    },
    parserOptions: {
      project: 'tsconfig.json',
      tsconfigRootDir: __dirname,
    },
  },
  rules: {
    'react-hooks/exhaustive-deps': 'error',
  },
});
