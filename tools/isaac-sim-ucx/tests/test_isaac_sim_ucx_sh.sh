#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# -- constants ---------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TOOL_DIR/../.." && pwd)"
RUNNER="$REPO_ROOT/isaac_sim"
RUNNER_NAME="$(basename "$RUNNER")"
HOST_APP_PATH="$TOOL_DIR/isaac_sim_ucx.py"
HOST_ENTRYPOINT_PATH="$TOOL_DIR/isaac_sim_entrypoint.sh"
CONTAINER_APP_DIR="/$(basename "$TOOL_DIR")"
CONTAINER_APP_PATH="$CONTAINER_APP_DIR/$(basename "$HOST_APP_PATH")"
CONTAINER_ENTRYPOINT_PATH="$CONTAINER_APP_DIR/$(basename "$HOST_ENTRYPOINT_PATH")"
CONTAINER_ASSET_ROOT="$CONTAINER_APP_DIR/assets"
HOST_USER="$(id -u):1234"
TMP_DIR="$(mktemp -d)"

# -- cleanup -----------------------------------------------------------------

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# -- assertions --------------------------------------------------------------

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    if [[ "$haystack" != *"$needle"* ]]; then
        echo "Expected to find:" >&2
        echo "  $needle" >&2
        echo "in:" >&2
        echo "$haystack" >&2
        exit 1
    fi
}

assert_not_contains() {
    local haystack="$1"
    local needle="$2"
    if [[ "$haystack" == *"$needle"* ]]; then
        echo "Expected not to find:" >&2
        echo "  $needle" >&2
        echo "in:" >&2
        echo "$haystack" >&2
        exit 1
    fi
}

# -- fixtures ----------------------------------------------------------------

make_fake_docker() {
    local fake_bin="$TMP_DIR/fake-bin"
    mkdir -p "$fake_bin"
    cat > "$fake_bin/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "$DOCKER_LOG"
printf '\n' >> "$DOCKER_LOG"
exit 0
EOF
    cat > "$fake_bin/sudo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-n" && "${2:-}" == "true" ]]; then
    exit 0
fi
if [[ "${1:-}" == "mkdir" ]]; then
    shift
    mkdir "$@"
    exit 0
fi
echo "unexpected sudo command: $*" >&2
exit 1
EOF
    cat > "$fake_bin/xhost" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "$XHOST_LOG"
printf '\n' >> "$XHOST_LOG"
exit 0
EOF
    cat > "$fake_bin/xauth" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
log="${XAUTH_LOG:-/dev/null}"
printf '%q ' "$@" >> "$log"
printf '\n' >> "$log"
if [[ "${1:-}" == "nlist" ]]; then
    echo "0100 fake-cookie"
    exit 0
fi
if [[ "${1:-}" == "-f" && "${3:-}" == "nmerge" ]]; then
    exit 0
fi
echo "unexpected xauth command: $*" >&2
exit 1
EOF
    chmod +x "$fake_bin/docker" "$fake_bin/sudo" "$fake_bin/xhost" "$fake_bin/xauth"
    echo "$fake_bin"
}

make_fake_sudo_denied() {
    local fake_bin="$TMP_DIR/fake-sudo-bin"
    mkdir -p "$fake_bin"
    cat > "$fake_bin/sudo" <<'EOF'
#!/usr/bin/env bash
exit 1
EOF
    chmod +x "$fake_bin/sudo"
    echo "$fake_bin"
}

make_isaacsim_repo() {
    local repo="$1"
    local build_log="$2"
    mkdir -p "$repo/tools/docker"
    cat > "$repo/build.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "unexpected-direct-source-build" >> "$build_log"
exit 1
EOF
    cat > "$repo/tools/docker/prep_docker_build.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" != "--build" ]]; then
    echo "prep requires --build" >&2
    exit 1
fi
echo "source-build" >> "$build_log"
echo "prep" >> "$build_log"
mkdir -p _build/linux-x86_64/release
EOF
    cat > "$repo/tools/docker/build_docker.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
while [[ \$# -gt 0 ]]; do
    case "\$1" in
        --tag)
            echo "docker-build \$2" >> "$build_log"
            shift 2
            ;;
        *)
            echo "unknown build_docker.sh option: \$1" >&2
            exit 1
            ;;
    esac
done
EOF
    chmod +x "$repo/build.sh" "$repo/tools/docker/prep_docker_build.sh" \
        "$repo/tools/docker/build_docker.sh"
    git -C "$repo" init >/dev/null
    git -C "$repo" checkout -b feature/test-branch >/dev/null 2>&1
    git -C "$repo" add build.sh tools/docker/prep_docker_build.sh tools/docker/build_docker.sh
    git -C "$repo" -c commit.gpgsign=false -c user.name='Test User' -c user.email='test@example.com' \
        commit -m 'test IsaacSim checkout' >/dev/null
}

make_fake_isaac_sim_root() {
    local root="$1"
    local license_status="$2"
    mkdir -p "$root"
    cat > "$root/license.sh" <<EOF
#!/usr/bin/env bash
echo "license notice"
if [[ "\${ACCEPT_EULA:-}" == "Y" ]]; then
    exit 0
fi
exit $license_status
EOF
    cat > "$root/privacy.sh" <<'EOF'
#!/usr/bin/env bash
echo "privacy notice"
EOF
    chmod +x "$root/license.sh" "$root/privacy.sh"
}

# -- tests -------------------------------------------------------------------

test_help_uses_invocation_name() {
    local output
    output="$("$RUNNER" --help)"

    assert_contains "$output" "  $RUNNER_NAME run [options]"
    assert_contains "$output" "$RUNNER_NAME run --scene scene.usd -- --port 50009"
}

test_default_run_uses_pinned_ngc_image_and_mounts_tooling() {
    local output
    local asset_root="$TMP_DIR/assets"
    mkdir -p "$asset_root"
    touch "$asset_root/scene.usd"

    output="$(
        DISPLAY= "$RUNNER" run \
            --dry-run \
            --cache-root "$TMP_DIR/cache-default" \
            --scene "$asset_root/scene.usd" \
            -y
    )"

    assert_contains "$output" "nvcr.io/nvidia/isaac-sim:6.2.0.1"
    assert_contains "$output" "Isaac Sim cache root: $TMP_DIR/cache-default"
    assert_contains "$output" "$asset_root:$CONTAINER_ASSET_ROOT:ro"
    assert_contains "$output" "--scene $CONTAINER_ASSET_ROOT/scene.usd"
    assert_contains "$output" "--env ACCEPT_EULA=Y"
    assert_contains "$output" "--env HOME=/isaac-sim"
    assert_contains "$output" "--env ISAAC_SIM_LAUNCHER=$RUNNER_NAME"
    assert_contains "$output" "--user $HOST_USER"
    assert_contains "$output" "/etc/passwd:/etc/passwd:ro"
    assert_contains "$output" "$TMP_DIR/cache-default/.config:/isaac-sim/.config:rw"
    assert_contains "$output" "$TMP_DIR/cache-default/.nv:/isaac-sim/.nv:rw"
    assert_contains "$output" "$TMP_DIR/cache-default/.nvidia-omniverse:/isaac-sim/.nvidia-omniverse:rw"
    assert_contains "$output" "$TMP_DIR/cache-default/.local/share/ov:/isaac-sim/.local/share/ov:rw"
    assert_contains "$output" "$TMP_DIR/cache-default/kit/cache:/isaac-sim/kit/cache:rw"
    assert_contains "$output" "$TMP_DIR/cache-default/kit/data:/isaac-sim/kit/data:rw"
    assert_contains "$output" "$TMP_DIR/cache-default/kit/logs:/isaac-sim/kit/logs:rw"
}

test_python_args_are_forwarded_after_separator() {
    local output
    local asset_root="$TMP_DIR/python-args-assets"
    mkdir -p "$asset_root"
    touch "$asset_root/scene.usd"

    output="$(
        DISPLAY= "$RUNNER" run \
            --dry-run \
            --cache-root "$TMP_DIR/cache-python-args" \
            --scene "$asset_root/scene.usd" \
            -y \
            -- --port 50009 --sim-dt 0.01
    )"

    assert_contains "$output" "--port 50009 --sim-dt 0.01"
}

test_warmup_uses_same_runner_and_exits_after_warmup() {
    local output

    output="$(
        DISPLAY= "$RUNNER" warmup \
            --dry-run \
            --cache-root "$TMP_DIR/cache-warmup" \
            -y
    )"

    assert_contains "$output" "nvcr.io/nvidia/isaac-sim:6.2.0.1"
    assert_contains "$output" "--warmup"
    assert_not_contains "$output" "--warmup 120"
    assert_not_contains "$output" "--exit-after-warmup"
    assert_not_contains "$output" "--scene"
}

test_run_without_scene_skips_asset_mount() {
    local output
    output="$(
        DISPLAY= "$RUNNER" run \
            --dry-run \
            --cache-root "$TMP_DIR/cache-without-scene" \
            -y
    )"

    assert_contains "$output" "nvcr.io/nvidia/isaac-sim:6.2.0.1"
    assert_not_contains "$output" "--scene"
}

test_privacy_consent_sets_container_envs() {
    local output
    output="$(
        DISPLAY= "$RUNNER" run \
            --dry-run \
            --cache-root "$TMP_DIR/cache-privacy" \
            --privacy-consent \
            -y
    )"

    assert_contains "$output" "--env PRIVACY_CONSENT=Y"
    assert_contains "$output" "--env OMNI_ENV_PRIVACY_CONSENT=Y"
}

test_docker_args_before_separator_are_forwarded_after_wrapper_defaults() {
    local output
    output="$(
        DISPLAY= "$RUNNER" run \
            --dry-run \
            --cache-root "$TMP_DIR/cache-docker-args" \
            -y \
            --name ucx-test \
            --ipc host \
            --entrypoint /bin/echo \
            -- --port 50009
    )"

    assert_contains "$output" "--name ucx-test"
    assert_contains "$output" "--ipc host"
    assert_contains "$output" "--entrypoint /bin/echo"
    assert_contains "$output" "--port 50009"
}

test_missing_display_defaults_to_headless() {
    local output
    output="$(
        DISPLAY= "$RUNNER" run \
            --dry-run \
            --cache-root "$TMP_DIR/cache-display-default" \
            -y
    )"

    assert_contains "$output" "--headless"
    assert_not_contains "$output" "/tmp/.X11-unix"
}

test_display_uses_xauth_for_x11_access() {
    local fake_bin
    fake_bin="$(make_fake_docker)"
    local docker_log="$TMP_DIR/display-docker.log"
    local xhost_log="$TMP_DIR/display-xhost.log"
    local xauth_log="$TMP_DIR/display-xauth.log"

    DOCKER_LOG="$docker_log" XHOST_LOG="$xhost_log" XAUTH_LOG="$xauth_log" \
        DISPLAY=:1 XDG_SESSION_TYPE=x11 \
        PATH="$fake_bin:$PATH" "$RUNNER" run \
            --cache-root "$TMP_DIR/cache-display" \
            -y >"$TMP_DIR/display.out"

    assert_contains "$(cat "$TMP_DIR/display.out")" "xauth -f"
    assert_contains "$(cat "$TMP_DIR/display.out")" "nmerge <cookie-from-:1>"
    assert_not_contains "$(cat "$TMP_DIR/display.out")" "xhost +local:docker"
    if [[ -e "$xhost_log" ]]; then
        assert_not_contains "$(cat "$xhost_log")" "+local:docker"
    fi
    assert_contains "$(cat "$xauth_log")" "nlist :1"
    assert_contains "$(cat "$xauth_log")" "nmerge -"
    assert_contains "$(cat "$docker_log")" "/tmp/.X11-unix:/tmp/.X11-unix:ro"
    assert_contains "$(cat "$docker_log")" "--env XAUTHORITY="
    assert_contains "$(cat "$docker_log")" ".docker.xauth."
}

test_display_dry_run_prints_xauth_mount() {
    local output

    output="$(
        DISPLAY=:2 "$RUNNER" run \
            --dry-run \
            --cache-root "$TMP_DIR/cache-display-dry-run" \
            -y
    )"

    assert_contains "$output" "xauth -f"
    assert_contains "$output" "nmerge <cookie-from-:2>"
    assert_contains "$output" "--env XAUTHORITY="
    assert_contains "$output" ".docker.xauth.DRYRUN"
}

test_ssh_display_uses_xauth_without_x11_socket_mount() {
    local fake_bin
    fake_bin="$(make_fake_docker)"
    local docker_log="$TMP_DIR/ssh-display-docker.log"
    local xauth_log="$TMP_DIR/ssh-display-xauth.log"

    DOCKER_LOG="$docker_log" XAUTH_LOG="$xauth_log" \
        DISPLAY=localhost:10.0 XDG_SESSION_TYPE=tty PATH="$fake_bin:$PATH" \
        "$RUNNER" run \
            --cache-root "$TMP_DIR/cache-ssh-display" \
            -y >"$TMP_DIR/ssh-display.out"

    assert_contains "$(cat "$xauth_log")" "nlist localhost:10.0"
    assert_contains "$(cat "$docker_log")" "--env DISPLAY=localhost:10.0"
    assert_contains "$(cat "$docker_log")" "--env XAUTHORITY="
    assert_not_contains "$(cat "$docker_log")" "/tmp/.X11-unix"
}

test_dry_run_allows_missing_scene_path() {
    local output
    local missing_scene="$TMP_DIR/missing-assets/scene.usd"

    output="$(
        DISPLAY= "$RUNNER" run \
            --dry-run \
            --cache-root "$TMP_DIR/cache-missing-scene" \
            --scene "$missing_scene" \
            -y
    )"

    assert_contains "$output" "$TMP_DIR/missing-assets:$CONTAINER_ASSET_ROOT:ro"
    assert_contains "$output" "--scene $CONTAINER_ASSET_ROOT/scene.usd"
}

test_shell_opens_interactive_container() {
    local output
    local asset_root="$TMP_DIR/shell-assets"
    mkdir -p "$asset_root"
    touch "$asset_root/scene.usd"

    output="$(
        DISPLAY= "$RUNNER" shell \
            --dry-run \
            --cache-root "$TMP_DIR/cache-shell" \
            --scene "$asset_root/scene.usd" \
            -y
    )"

    assert_contains "$output" "--interactive"
    assert_contains "$output" "/bin/bash"
}

test_entrypoint_reports_wrapper_eula_flag_on_prompt_decline() {
    local fake_root="$TMP_DIR/fake-isaac-decline"
    make_fake_isaac_sim_root "$fake_root" 1
    local stdout="$TMP_DIR/entrypoint-license.out"
    local stderr="$TMP_DIR/entrypoint-license.err"

    if ISAAC_SIM_LAUNCHER="$RUNNER_NAME" ISAAC_SIM_ROOT="$fake_root" "$HOST_ENTRYPOINT_PATH" \
        echo should-not-run \
        </dev/null >"$stdout" 2>"$stderr"; then
        fail "entrypoint should fail when EULA prompt is declined"
    fi

    assert_contains "$(cat "$stdout" "$stderr")" "license notice"
    assert_contains "$(cat "$stdout" "$stderr")" "Accept Isaac Sim EULA? [y/N]"
    assert_contains "$(cat "$stdout" "$stderr")" "-y or --accept-eula"
    assert_contains "$(cat "$stdout" "$stderr")" "when using $RUNNER_NAME"
    assert_not_contains "$(cat "$stdout")" "should-not-run"
}

test_entrypoint_accepts_eula_prompt_response() {
    local fake_root="$TMP_DIR/fake-isaac-prompt"
    make_fake_isaac_sim_root "$fake_root" 1
    local stderr="$TMP_DIR/entrypoint-prompt.err"
    local output

    output="$(printf 'yes\n' | ISAAC_SIM_ROOT="$fake_root" "$HOST_ENTRYPOINT_PATH" echo app-ran 2>"$stderr")"

    assert_contains "$output" "license notice"
    assert_contains "$(cat "$stderr")" "Accept Isaac Sim EULA? [y/N]"
    assert_contains "$output" "privacy notice"
    assert_contains "$output" "app-ran"
}

test_entrypoint_execs_command_after_license_and_privacy() {
    local fake_root="$TMP_DIR/fake-isaac-ok"
    make_fake_isaac_sim_root "$fake_root" 1
    local output

    output="$(ACCEPT_EULA=Y ISAAC_SIM_ROOT="$fake_root" "$HOST_ENTRYPOINT_PATH" echo app-ran)"

    assert_not_contains "$output" "license notice"
    assert_contains "$output" "privacy notice"
    assert_contains "$output" "app-ran"
}

test_cache_root_reports_permission_fix() {
    local fake_bin
    fake_bin="$(make_fake_sudo_denied)"
    local locked_parent="$TMP_DIR/locked-cache-parent"
    local stderr="$TMP_DIR/locked-cache.err"
    mkdir -p "$locked_parent"
    chmod 555 "$locked_parent"

    if PATH="$fake_bin:$PATH" "$RUNNER" shell \
        --cache-root "$locked_parent/isaac-sim" \
        -y >"$TMP_DIR/locked-cache.out" 2>"$stderr"; then
        chmod 755 "$locked_parent"
        fail "unwritable cache parent should fail before docker run"
    fi
    chmod 755 "$locked_parent"

    assert_contains "$(cat "$stderr")" "cannot create or update Isaac Sim cache directories"
    assert_contains "$(cat "$stderr")" "sudo mkdir -p"
    assert_contains "$(cat "$stderr")" "sudo chown -R"
    assert_contains "$(cat "$stderr")" "$HOST_USER"
}

test_from_source_builds_and_runs_commit_image() {
    local fake_bin
    fake_bin="$(make_fake_docker)"
    local docker_log="$TMP_DIR/docker-build.log"
    local build_log="$TMP_DIR/base-build.log"
    local repo="$TMP_DIR/IsaacSim"
    local asset_root="$TMP_DIR/source-assets"
    make_isaacsim_repo "$repo" "$build_log"
    mkdir -p "$asset_root"
    touch "$asset_root/scene.usd"
    local commit_sha
    commit_sha="$(git -C "$repo" rev-parse --short=12 HEAD)"

    DOCKER_LOG="$docker_log" PATH="$fake_bin:$PATH" \
        "$RUNNER" build --from-source "$repo" -y

    local build_commands
    build_commands="$(cat "$build_log")"
    assert_contains "$build_commands" "source-build"
    assert_contains "$build_commands" "prep"
    assert_contains "$build_commands" "docker-build"

    : > "$build_log"
    DOCKER_LOG="$docker_log" DISPLAY= PATH="$fake_bin:$PATH" \
        "$RUNNER" run \
            --from-source "$repo" \
            --cache-root "$TMP_DIR/cache-run" \
            --scene "$asset_root/scene.usd" \
            -y

    build_commands="$(cat "$build_log")"
    assert_contains "$build_commands" "source-build"
    assert_contains "$build_commands" "prep"
    assert_contains "$build_commands" "docker-build"

    local docker_commands
    docker_commands="$(cat "$docker_log")"
    assert_contains "$docker_commands" "isaac-sim-docker:$commit_sha"
}

test_source_and_image_options_are_mutually_exclusive() {
    local repo="$TMP_DIR/IsaacSim-conflict"
    local build_log="$TMP_DIR/source-image-conflict-build.log"
    local stderr="$TMP_DIR/source-image-conflict.err"
    make_isaacsim_repo "$repo" "$build_log"

    if "$RUNNER" run \
        --from-source "$repo" \
        --image other:tag \
        --dry-run \
        --cache-root "$TMP_DIR/cache-source-image-conflict" \
        -y >"$TMP_DIR/source-image-conflict.out" 2>"$stderr"; then
        fail "--from-source with --image should fail"
    fi

    assert_contains "$(cat "$stderr")" "--from-source cannot be combined with --image"
}

test_source_image_rejects_explicit_tags() {
    local repo="$TMP_DIR/IsaacSim-source-image-tag"
    local build_log="$TMP_DIR/source-image-tag-build.log"
    local stderr="$TMP_DIR/source-image-tag.err"
    make_isaacsim_repo "$repo" "$build_log"

    if "$RUNNER" build \
        --from-source "$repo" \
        --source-image isaac-sim-docker:custom \
        --dry-run \
        -y >"$TMP_DIR/source-image-tag.out" 2>"$stderr"; then
        fail "--source-image with a tag should fail"
    fi

    assert_contains "$(cat "$stderr")" "without a tag or digest"
}

test_build_requires_source_checkout() {
    local stderr="$TMP_DIR/build-without-source.err"
    if "$RUNNER" build --dry-run -y >"$TMP_DIR/build-without-source.out" 2>"$stderr"; then
        fail "build without --from-source should fail"
    fi

    assert_contains "$(cat "$stderr")" "build requires --from-source"
}

test_build_dry_run_prints_without_executing() {
    local repo="$TMP_DIR/IsaacSim-build-dry-run"
    local build_log="$TMP_DIR/build-dry-run.log"
    make_isaacsim_repo "$repo" "$build_log"

    local output
    output="$("$RUNNER" build --from-source "$repo" --dry-run -y)"

    assert_contains "$output" "prep_docker_build.sh --build"
    assert_contains "$output" "build_docker.sh --tag"
    if [[ -e "$build_log" ]]; then
        fail "build --dry-run should not execute source build scripts"
    fi
}

# -- main --------------------------------------------------------------------

test_help_uses_invocation_name
test_default_run_uses_pinned_ngc_image_and_mounts_tooling
test_python_args_are_forwarded_after_separator
test_warmup_uses_same_runner_and_exits_after_warmup
test_run_without_scene_skips_asset_mount
test_privacy_consent_sets_container_envs
test_docker_args_before_separator_are_forwarded_after_wrapper_defaults
test_missing_display_defaults_to_headless
test_display_uses_xauth_for_x11_access
test_display_dry_run_prints_xauth_mount
test_ssh_display_uses_xauth_without_x11_socket_mount
test_dry_run_allows_missing_scene_path
test_shell_opens_interactive_container
test_entrypoint_reports_wrapper_eula_flag_on_prompt_decline
test_entrypoint_accepts_eula_prompt_response
test_entrypoint_execs_command_after_license_and_privacy
test_cache_root_reports_permission_fix
test_from_source_builds_and_runs_commit_image
test_source_and_image_options_are_mutually_exclusive
test_source_image_rejects_explicit_tags
test_build_requires_source_checkout
test_build_dry_run_prints_without_executing

echo "isaac_sim wrapper tests passed"
