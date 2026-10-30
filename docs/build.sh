#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# Build and preview Isaac OS documentation with Fern.
#
# Usage:
#   docs/build.sh              Validate with `fern check --local --warnings`
#   docs/build.sh preview      Start `fern docs dev` (default port 3000)
#   docs/build.sh login        Log in to Fern for the shared NVIDIA theme
#   docs/build.sh --no-docker  Run Fern on the host instead of the docs image
#   docs/build.sh --img TAG    Use a specific container image
#   docs/build.sh --port PORT  Serve preview on a specific port
#
# Prerequisites:
#   Python 3 on the host, plus Docker (default) or a host install of
#   fern-api@5.168.0 matching docs/fern/fern.config.json.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_IMAGE="isaac_os_docs:latest"
IMAGE="${DEFAULT_IMAGE}"
USE_DOCKER=1
TARGET="check"
WATCH_PORT="3000"
SOURCE_REPOSITORY=""
SOURCE_REF=""
RUN_ROOT="${REPO_ROOT}"
RUN_DOCS_DIR="${SCRIPT_DIR}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-repository|--source-ref)
            [[ $# -ge 2 ]] || { echo "Error: $1 requires an argument" >&2; exit 1; }
            if [[ "$1" == "--source-repository" ]]; then
                SOURCE_REPOSITORY="$2"
            else
                SOURCE_REF="$2"
            fi
            shift 2
            ;;
        --img)
            [[ $# -ge 2 ]] || { echo "Error: --img requires an argument" >&2; exit 1; }
            IMAGE="$2"
            shift 2
            ;;
        --port)
            [[ $# -ge 2 ]] || { echo "Error: --port requires an argument" >&2; exit 1; }
            WATCH_PORT="$2"
            shift 2
            ;;
        --no-docker)
            USE_DOCKER=0
            shift
            ;;
        check|preview|login)
            TARGET="$1"
            shift
            ;;
        -h|--help)
            echo "Usage: docs/build.sh [--no-docker] [--img IMAGE] [--port PORT] [check|preview|login]"
            echo ""
            echo "Targets:"
            echo "  check     Validate Fern config and MDX (default)"
            echo "  preview   Run the Fern local docs server"
            echo "  login     Log in to Fern and save credentials in ~/.fern"
            echo ""
            echo "Options:"
            echo "  --no-docker  Use the host Fern CLI instead of Docker"
            echo "  --img TAG    Use a specific container image (default: ${DEFAULT_IMAGE})"
            echo "  --port PORT  Serve preview on PORT (default: 3000)"
            echo "  --source-repository URL --source-ref REF  Preview resolved source links"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Run 'docs/build.sh --help' for usage." >&2
            exit 1
            ;;
    esac
done

if [[ -n "${SOURCE_REPOSITORY}" || -n "${SOURCE_REF}" ]] &&
    [[ -z "${SOURCE_REPOSITORY}" || -z "${SOURCE_REF}" || "${TARGET}" == "login" ]]; then
    echo "Error: source-link previews require both repository and ref, with check or preview." >&2
    exit 1
fi

if [[ "${TARGET}" != "login" ]]; then
    python3 "${SCRIPT_DIR}/scripts/sync_application_pages.py" --check
    python3 "${SCRIPT_DIR}/scripts/check_doc_links.py" --docs-root "${SCRIPT_DIR}"
    python3 "${SCRIPT_DIR}/scripts/check_doc_links.py" --docs-root "${SCRIPT_DIR}" --local-files-only
fi

if [[ -n "${SOURCE_REPOSITORY}" ]]; then
    publication_tmp="$(mktemp -d)"
    trap 'rm -rf -- "${publication_tmp}"' EXIT
    RUN_ROOT="${publication_tmp}/content"
    python3 "${SCRIPT_DIR}/scripts/prepare_publication.py" \
        --root "${REPO_ROOT}" --output "${RUN_ROOT}" \
        --source-repository "${SOURCE_REPOSITORY}" --source-ref "${SOURCE_REF}"
    RUN_DOCS_DIR="${RUN_ROOT}/docs"
fi

fern_cmd=(fern)
if [[ "${TARGET}" == "preview" ]]; then
    fern_cmd+=(docs dev --port "${WATCH_PORT}")
elif [[ "${TARGET}" == "login" ]]; then
    umask 077
    fern_cmd+=(login --device-code)
else
    fern_cmd+=(check --local --warnings)
fi

if [[ "${USE_DOCKER}" -eq 0 ]]; then
    echo "Running: ${fern_cmd[*]} (host)"
    (cd "${RUN_DOCS_DIR}" && "${fern_cmd[@]}")
    exit 0
fi

echo "Building docs image from docs/Dockerfile ..."
docker build \
    --network host \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE}" \
    "${SCRIPT_DIR}"

echo "Running: ${fern_cmd[*]} using image: ${IMAGE}"

docker_args=(
    run --rm --init --network host
    -v "${RUN_ROOT}:/workspace/isaac_os:rw"
    -w /workspace/isaac_os/docs
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp
)
if [[ "${TARGET}" == "login" ]]; then
    mkdir -p "${HOME}/.fern"
fi
if [[ -d "${HOME}/.fern" ]]; then
    docker_args+=(-v "${HOME}/.fern:/tmp/.fern:rw")
fi
if [[ -n "${FERN_TOKEN:-}" ]]; then
    docker_args+=(-e FERN_TOKEN)
fi

docker "${docker_args[@]}" "${IMAGE}" "${fern_cmd[@]}"
