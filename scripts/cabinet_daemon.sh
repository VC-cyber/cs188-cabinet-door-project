#!/usr/bin/env bash
set -euo pipefail

# Simple daemon to run cabinet-door experiments defined in PLANS.md.
#
# It looks for the first unchecked TODO item that has a following
#   Command: scripts/cabinet_experiment.sh ...
# line, and executes that command from the repo root.
#
# Usage:
#   scripts/cabinet_daemon.sh --once
#   scripts/cabinet_daemon.sh --interval 1800      # poll every 30 minutes
#   scripts/cabinet_daemon.sh --dry-run --once     # just print the next command
#
# This script is intentionally simple and token-cheap: an agent only needs to
# decide *when* to run it, not re-specify full experiment commands.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLANS_FILE="${REPO_ROOT}/PLANS.md"
LOCK_FILE="${REPO_ROOT}/.cabinet_daemon.lock"

INTERVAL_SECONDS=1800   # default: 30 minutes
RUN_ONCE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      shift
      INTERVAL_SECONDS="${1:-1800}"
      ;;
    --once)
      RUN_ONCE=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --plans)
      shift
      PLANS_FILE="${1:-${PLANS_FILE}}"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: scripts/cabinet_daemon.sh [--once] [--interval SECONDS] [--dry-run] [--plans PATH]" >&2
      exit 1
      ;;
  esac
  shift || true
done

if [[ ! -f "${PLANS_FILE}" ]]; then
  echo "ERROR: PLANS file not found at ${PLANS_FILE}" >&2
  exit 1
fi

log() {
  # Timestamped log line to stderr
  printf '[%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*" >&2
}

find_next_command() {
  # Parse PLANS.md and return the first Command: line under an unchecked task.
  # Convention in PLANS.md:
  #   - [ ] YYYY-MM-DD ... summary
  #         Command: scripts/cabinet_experiment.sh ...
  awk '
    /^- \[ \]/ { in_task=1 }                       # start of an unchecked task
    /^[[:space:]]*Command:[[:space:]]*/ && in_task {
      cmd=$0
      sub(/^[[:space:]]*Command:[[:space:]]*/, "", cmd)
      print cmd
      exit
    }
  ' "${PLANS_FILE}"
}

is_busy() {
  # Check lock file and whether the recorded PID is still alive.
  if [[ ! -f "${LOCK_FILE}" ]]; then
    return 1
  fi

  local pid cmd
  # shellcheck disable=SC2162
  read pid cmd < "${LOCK_FILE}" || {
    rm -f "${LOCK_FILE}"
    return 1
  }

  if ps -p "${pid}" > /dev/null 2>&1; then
    log "Existing run in progress (pid=${pid}, cmd=${cmd})"
    return 0
  fi

  # Stale lock; clean up.
  log "Stale lock file found for pid=${pid}; removing."
  rm -f "${LOCK_FILE}"
  return 1
}

run_once() {
  local cmd

  if is_busy; then
    # Another daemon / run is active; skip starting a new one.
    return 0
  fi

  cmd="$(find_next_command || true)"

  if [[ -z "${cmd}" ]]; then
    log "No runnable TODO with Command: line found in ${PLANS_FILE}"
    return 0
  fi

  log "Next experiment command from PLANS.md:"
  log "  ${cmd}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "Dry run enabled; not executing."
    return 0
  fi

  log "Executing from repo root: ${REPO_ROOT}"
  printf '%d\t%s\n' "$$" "${cmd}" > "${LOCK_FILE}"

  (
    cd "${REPO_ROOT}"
    bash -lc "${cmd}"
  )

  local status=$?
  rm -f "${LOCK_FILE}" || true
  return "${status}"
}

log "cabinet_daemon starting (PLANS=${PLANS_FILE}, interval=${INTERVAL_SECONDS}s, once=${RUN_ONCE}, dry_run=${DRY_RUN})"

if [[ "${RUN_ONCE}" -eq 1 ]]; then
  run_once
  log "cabinet_daemon finished single run."
  exit 0
fi

while true; do
  run_once || log "Warning: run_once returned non-zero exit code."
  log "Sleeping for ${INTERVAL_SECONDS} seconds..."
  sleep "${INTERVAL_SECONDS}"
done

