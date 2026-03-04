#!/usr/bin/env bash
set -euo pipefail

# Lightweight CLI for cabinet-door experiments.
# Usage examples:
#   scripts/cabinet_experiment.sh train --epochs 200
#   scripts/cabinet_experiment.sh train --config cabinet_door_project/configs/diffusion_policy.yaml
#   scripts/cabinet_experiment.sh eval --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt
#   scripts/cabinet_experiment.sh viz  --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt --offscreen
#
# This wrapper exists so agents/daemons can call a single, short command
# instead of remembering individual script paths.

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/cabinet_experiment.sh <train|eval|viz> [args...]" >&2
  exit 1
fi

subcommand="$1"
shift || true

case "${subcommand}" in
  train)
    # Train an action-chunking BC policy (Step 6).
    exec python cabinet_door_project/06_train_policy.py "$@"
    ;;
  eval|evaluate)
    # Evaluate a trained policy (Step 7).
    exec python cabinet_door_project/07_evaluate_policy.py "$@"
    ;;
  viz|visualize|rollout)
    # Visualize rollouts of a trained policy (Step 8).
    exec python cabinet_door_project/08_visualize_policy_rollout.py "$@"
    ;;
  *)
    echo "Unknown subcommand: ${subcommand}" >&2
    echo "Valid subcommands: train, eval, viz" >&2
    exit 1
    ;;
esac

