#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
test_root="$(mktemp -d "${TMPDIR:-/tmp}/isaac-os-cli-test.XXXXXX")"
trap 'rm -rf "${test_root}"' EXIT

mkdir -p "${test_root}/bin" "${test_root}/home"
capture_path="${test_root}/python-invocation"
sentinel_path="${test_root}/ros-python"
cli_requirement="$(sed -nE 's/^[[:space:]]*"(holoscan-cli[^"]*)",?[[:space:]]*$/\1/p' "${repo_root}/pyproject.toml")"
cli_pin="${cli_requirement#holoscan-cli==}"
cli_pin="${cli_pin##*@}"
export ISAAC_TEST_REAL_PYTHON="$(command -v python3)"
fixture_version=0.0.0
if [[ "${cli_requirement}" == holoscan-cli==* ]]; then
    fixture_version="${cli_pin}"
fi
dist_info="${sentinel_path}/holoscan_cli-${fixture_version}.dist-info"
mkdir -p "${dist_info}"
printf 'Metadata-Version: 2.1\nName: holoscan-cli\nVersion: %s\n' \
    "${fixture_version}" > "${dist_info}/METADATA"
cat > "${dist_info}/direct_url.json" <<EOF
{
  "url": "https://github.com/nvidia-holoscan/holoscan-cli.git",
  "vcs_info": {"vcs": "git", "commit_id": "${cli_pin}"}
}
EOF

cat > "${test_root}/bin/python3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
    if [[ "$arg" == "-" ]]; then
        printf '%s\n' "${PYTHONPATH-__UNSET__}" > "${ISAAC_TEST_CAPTURE}.probe"
        exec "${ISAAC_TEST_REAL_PYTHON}" "$@"
    fi
done

{
    printf '%s\n' "${PYTHONPATH-__UNSET__}"
    printf '%s\n' "$@"
} > "${ISAAC_TEST_CAPTURE}"
{
    printf '%s\n' "${PWD}"
    printf '%s\n' "${HOLOSCAN_CLI_ROOT-__UNSET__}"
    printf '%s\n' "${HOLOSCAN_CLI_BUILD_PARENT_DIR-__UNSET__}"
    printf '%s\n' "${HOLOSCAN_CLI_SEARCH_PATH-__UNSET__}"
    printf '%s\n' "${HOLOSCAN_CLI_BASE_SDK_VERSION-__UNSET__}"
} > "${ISAAC_TEST_CAPTURE}.environment"
EOF
chmod +x "${test_root}/bin/python3"

(
    cd "${test_root}"
    env -u HOLOSCAN_CLI_BUILD_PARENT_DIR -u HOLOSCAN_CLI_SEARCH_PATH \
    -u HOLOSCAN_CLI_BASE_SDK_VERSION \
    HOME="${test_root}/home" \
    ISAAC_OS_CONTAINER=1 \
    ISAAC_TEST_CAPTURE="${capture_path}" \
    ISAAC_USE_SCCACHE=OFF \
    PATH="${test_root}/bin:${PATH}" \
    PYTHONPATH="${sentinel_path}" \
        "${repo_root}/isaac_os" list
)

read -r probe_pythonpath < "${capture_path}.probe"
if [[ "${probe_pythonpath}" != "${sentinel_path}" ]]; then
    printf 'Expected version probe PYTHONPATH:\n  %s\n' "${sentinel_path}" >&2
    printf 'Observed version probe PYTHONPATH:\n  %s\n' "${probe_pythonpath}" >&2
    exit 1
fi

mapfile -t captured < "${capture_path}"
expected=("${sentinel_path}" "-u" "-m" "holoscan_cli" "list")

if [[ "${captured[*]}" != "${expected[*]}" ]]; then
    printf 'Expected Python environment and arguments:\n  %s\n' "${expected[*]}" >&2
    printf 'Observed Python environment and arguments:\n  %s\n' "${captured[*]}" >&2
    exit 1
fi

mapfile -t environment < "${capture_path}.environment"
expected_environment=("${test_root}" "${repo_root}" "${repo_root}/build")
expected_environment+=("tutorials,benchmarks,subgraphs,applications,operators")
expected_environment+=("5.0.0")

if [[ "${environment[*]}" != "${expected_environment[*]}" ]]; then
    printf 'Expected wrapper environment:\n  %s\n' \
        "${expected_environment[*]}" >&2
    printf 'Observed wrapper environment:\n  %s\n' \
        "${environment[*]}" >&2
    exit 1
fi

printf 'Metadata-Version: 2.1\nName: holoscan-cli\nVersion: 0.0.0\n' > "${dist_info}/METADATA"
cat > "${dist_info}/direct_url.json" <<'EOF'
{
  "url": "https://github.com/nvidia-holoscan/holoscan-cli.git",
  "vcs_info": {"vcs": "git", "commit_id": "0000000000000000000000000000000000000000"}
}
EOF
if env ISAAC_OS_CONTAINER=1 ISAAC_TEST_CAPTURE="${capture_path}" \
    ISAAC_USE_SCCACHE=OFF PYTHONPATH="${sentinel_path}" \
    PATH="${test_root}/bin:${PATH}" \
    "${repo_root}/isaac_os" list > "${test_root}/outdated.log" 2>&1; then
    printf 'Expected an outdated container CLI to be rejected.\n' >&2
    exit 1
fi
grep -q 'Rebuild the image' "${test_root}/outdated.log"
