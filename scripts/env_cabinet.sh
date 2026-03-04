#!/usr/bin/env bash
#
# Convenience script to enter the cabinet-door project environment.
# Usage (IMPORTANT: must be sourced, not executed):
#   source scripts/env_cabinet.sh
#
# This will:
#   - cd to the repo root
#   - activate .venv if it exists
#   - cd into cabinet_door_project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ -d ".venv" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
  echo "[env_cabinet] Activated virtualenv: ${VIRTUAL_ENV:-.venv}"
else
  echo "[env_cabinet] WARNING: .venv not found; run ./install.sh first?" >&2
fi

cd cabinet_door_project
echo "[env_cabinet] Now in: $(pwd)"

