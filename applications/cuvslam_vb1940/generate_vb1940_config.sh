#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Keep the documented `sh generate_vb1940_config.sh` invocation working.
case "${BASH_VERSION:-}" in
    '') exec bash "$0" "$@" ;;
    *) ;;
esac
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
HSB_VERSION=2.7.0
HSB_IMAGE="hololink-demo:$HSB_VERSION"
# An existing checkout is optional; missing or incompatible sources are cloned automatically.
HSB_DIR=${HSB_DIR:-"${XDG_CACHE_HOME:-"$HOME/.cache"}/isaac-os/holoscan-sensor-bridge-$HSB_VERSION"}

# Help must work before Docker or HSB is set up.
for argument in "$@"; do
    case "$argument" in
        -h|--help) exec python3 "$SCRIPT_DIR/vb1940_calibration.py" --help ;;
        *) continue ;;
    esac
done

# Mount template/output paths separately so paths outside the checkout work too.
output="$SCRIPT_DIR/config/config.yaml"
template=
exporter_args=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -o|--output|--template)
            option=$1
            if [[ "$#" -lt 2 ]]; then
                printf '[error] %s requires a file path.\n' "$option" >&2
                exit 2
            fi
            value=$2
            shift
            ;;
        --output=*) option=--output; value=${1#*=} ;;
        --template=*) option=--template; value=${1#*=} ;;
        -o?*) option=--output; value=${1#-o} ;;
        *) exporter_args+=("$1"); shift; continue ;;
    esac
    if [[ -z "$value" ]]; then
        printf '[error] %s requires a file path.\n' "$option" >&2
        exit 2
    fi
    case "$option" in
        --template) template=$(realpath -e -- "$value") ;;
        *) output=$(realpath -m -- "$value") ;;
    esac
    shift
done

# Docker uses CSV for --mount; quote the entire source field for commas and quotes.
bind_mount() {
    local source=${1//\"/\"\"}
    local destination=$2
    printf 'type=bind,"src=%s",dst=%s' "$source" "$destination"
}

mounts=(--mount "$(bind_mount "$SCRIPT_DIR/vb1940_calibration.py" /vb1940_calibration.py),readonly")
mounts+=(--mount "$(bind_mount "$(dirname -- "$output")" /output)")
exporter_args+=(--output "/output/$(basename -- "$output")")
if [[ -n "$template" ]]; then
    if [[ ! -f "$template" ]]; then
        printf '[error] The YAML template is not a regular file: %s\n' "$template" >&2
        exit 1
    fi
    mounts+=(--mount "$(bind_mount "$template" /template.yaml),readonly")
    exporter_args+=(--template /template.yaml)
fi

# Reuse the local image. Otherwise prepare HSB automatically for Thor's integrated GPU.
docker info >/dev/null
if ! docker image inspect "$HSB_IMAGE" >/dev/null 2>&1; then
    (
        if [[ ! -f "$HSB_DIR/docker/build.sh" || ! -f "$HSB_DIR/VERSION" ||
            "$(cat "$HSB_DIR/VERSION")" != "$HSB_VERSION" ]]; then
            printf 'Downloading Holoscan Sensor Bridge %s for calibration.\n' "$HSB_VERSION"
            cache_dir="${XDG_CACHE_HOME:-"$HOME/.cache"}/isaac-os"
            mkdir -p -- "$cache_dir"
            # Keep existing checkouts intact and remove temporary sources even if setup fails.
            HSB_DIR=$(mktemp -d "$cache_dir/holoscan-sensor-bridge-$HSB_VERSION.XXXXXX")
            trap 'rm -rf -- "$HSB_DIR"' EXIT
            git clone --branch "$HSB_VERSION" --depth 1 --recurse-submodules \
                https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git "$HSB_DIR"
        fi
        cd -- "$HSB_DIR"
        bash docker/build.sh --igpu
    )
fi

mkdir -p -- "$(dirname -- "$output")"
exec docker run --rm --pull=never \
    --network host --runtime=nvidia --gpus all \
    --user "$(id -u):$(id -g)" \
    "${mounts[@]}" \
    "$HSB_IMAGE" python3 /vb1940_calibration.py "${exporter_args[@]}"
