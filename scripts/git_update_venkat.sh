#!/usr/bin/env bash
set -euo pipefail

# Helper to commit and push experiment changes on the current branch
# (intended for venkat, but works on any branch).
#
# This script is designed so that a human or an agent can run:
#   scripts/git_update_venkat.sh "short commit message"
#
# Behavior:
#   1. Runs python3 -m py_compile on all modified .py files.
#   2. Stages tracked changes (`git add -u`) plus new whitelisted files
#      (PLANS.md, AGENTS.md, scripts/, cabinet_door_project/*.py, configs).
#   3. Creates a commit with the provided message (or a default).
#   4. Pushes to origin for the current branch.
#
# It intentionally avoids staging obvious artifact paths; rely on .gitignore
# for runs/, logs/, etc.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

COMMIT_MSG="${1:-Update experiments and plans}"

echo "[git_update_venkat] Using repo root: ${REPO_ROOT}"
echo "[git_update_venkat] Commit message: ${COMMIT_MSG}"

echo "[git_update_venkat] Checking modified Python files..."
MOD_PY_FILES=()
while IFS= read -r line; do
  # line format from porcelain: "XY path"
  path="${line#?? }"
  MOD_PY_FILES+=("${path}")
done < <(git status --porcelain | awk '$1 ~ /[AM]/ && $2 ~ /\.py$/ {print $0}')

if [[ ${#MOD_PY_FILES[@]} -gt 0 ]]; then
  echo "[git_update_venkat] Running python3 -m py_compile on modified .py files..."
  for f in "${MOD_PY_FILES[@]}"; do
    echo "  - ${f}"
    python3 -m py_compile "${f}"
  done
else
  echo "[git_update_venkat] No modified .py files to validate."
fi

echo "[git_update_venkat] Staging tracked changes (git add -u)..."
git add -u

echo "[git_update_venkat] Staging new whitelisted files..."
git add \
  AGENTS.md PLANS.md \
  scripts/*.sh scripts/*.py \
  cabinet_door_project/*.py \
  cabinet_door_project/configs/diffusion_policy.yaml \
  2>/dev/null || true

echo "[git_update_venkat] Checking if there is anything staged..."
if git diff --cached --quiet; then
  echo "[git_update_venkat] No staged changes; nothing to commit."
  exit 0
fi

echo "[git_update_venkat] Creating commit..."
git commit -m "${COMMIT_MSG}"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "[git_update_venkat] Pushing branch '${BRANCH}' to origin..."
git push -u origin "${BRANCH}"

echo "[git_update_venkat] Done."

