// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ExtensionContext } from "@foxglove/extension";

import { activateConverters } from "./converters";

export function activate(extensionContext: ExtensionContext): void {
  activateConverters(extensionContext);
}
