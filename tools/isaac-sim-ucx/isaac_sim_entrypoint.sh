#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# -- constants ---------------------------------------------------------------

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/isaac-sim}"
ISAAC_SIM_LAUNCHER="${ISAAC_SIM_LAUNCHER:-the Isaac Sim launcher}"
EULA_HINT="That can also be done with -y or --accept-eula when using $ISAAC_SIM_LAUNCHER."
PRIVACY_HINT="That can also be done with --privacy-consent when using $ISAAC_SIM_LAUNCHER."

# -- eula --------------------------------------------------------------------

if [[ "${ACCEPT_EULA:-}" != "Y" ]]; then
    "$ISAAC_SIM_ROOT/license.sh" 2>&1 || true
    echo "$EULA_HINT"
    printf '\nAccept Isaac Sim EULA? [y/N] ' >&2
    IFS= read -r accept_eula_response
    case "${accept_eula_response,,}" in
        y|yes)  export ACCEPT_EULA=Y;;
        *)      exit 2;;
    esac
    echo
fi

# -- privacy -----------------------------------------------------------------

if [[ "${PRIVACY_CONSENT:-}" != "Y" ]] && [[ "${OMNI_ENV_PRIVACY_CONSENT:-}" != "Y" ]]; then
    "$ISAAC_SIM_ROOT/privacy.sh"
    echo "$PRIVACY_HINT"
    echo
fi

# -- isaac sim python.sh signal handling -------------------------------------
# /isaac-sim/python.sh starts Python as a child instead of execing it, and does
# not exit cleanly on Ctrl+C. Keep this entrypoint alive only to translate
# Ctrl+C into SIGTERM for that child. Docker --init handles docker stop/SIGTERM.

if [[ "${1:-}" == "$ISAAC_SIM_ROOT/python.sh" ]]; then
  "$@" &
  child_pid="$!"

  stop_child_on_interrupt() {
      trap - INT
      kill -TERM "$child_pid" >/dev/null 2>&1 || true
      wait "$child_pid" >/dev/null 2>&1 || true
      exit 130
  }
  trap stop_child_on_interrupt INT

  set +e
  wait "$child_pid"
  child_status="$?"
  set -e
  exit "$child_status"
fi

# -- command -----------------------------------------------------------------

exec "$@"
