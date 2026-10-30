# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Documentation image for the Fern user guide.

ARG NODE_VERSION=22-bookworm-slim

FROM node:${NODE_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    'lsof=4.95.0-*' \
    python3 \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g fern-api@5.168.0 pnpm@9.15.9 --ignore-scripts

WORKDIR /workspace/isaac_os/docs
USER 1000
