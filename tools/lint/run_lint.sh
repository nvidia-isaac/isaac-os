#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Self-contained lint script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# Install pre-commit if not available
if ! command -v pre-commit &>/dev/null; then
    echo "pre-commit not found, installing..."
    pip install --only-binary :all: pre-commit
fi

# Skip hooks that only make sense during actual git commit.
export SKIP="${SKIP:+${SKIP},}no-commit-to-branch"

# Run pre-commit linting with optional arguments.
pre-commit run --show-diff-on-failure "$@"
