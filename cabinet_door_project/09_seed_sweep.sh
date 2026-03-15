#!/usr/bin/env bash
set -euo pipefail

# Simple overnight seed sweep runner with live terminal output + log file.
#
# Usage:
#   bash 09_seed_sweep.sh
#   bash 09_seed_sweep.sh "0 1 2 3 4 5" 30 50 600 pretrain
#
# Args:
#   $1 seeds (space-separated)        default: "0 1 2 3 4"
#   $2 epochs                         default: 30
#   $3 num_rollouts                   default: 50
#   $4 max_steps                      default: 600
#   $5 split                          default: pretrain

SEEDS="${1:-0 1 2 3 4}"
EPOCHS="${2:-30}"
ROLLOUTS="${3:-50}"
MAX_STEPS="${4:-600}"
SPLIT="${5:-pretrain}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/sweep_results"
mkdir -p "${OUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUT_DIR}/seed_sweep_${TS}.txt"
CKPT_ROOT="/tmp/cabinet_policy_seed_sweep_${TS}"
mkdir -p "${CKPT_ROOT}"

echo "====================================================================" | tee -a "${LOG_FILE}"
echo "OpenCabinet Seed Sweep (bash runner)" | tee -a "${LOG_FILE}"
echo "====================================================================" | tee -a "${LOG_FILE}"
echo "Started:      $(date)" | tee -a "${LOG_FILE}"
echo "Seeds:        ${SEEDS}" | tee -a "${LOG_FILE}"
echo "Epochs:       ${EPOCHS}" | tee -a "${LOG_FILE}"
echo "Rollouts:     ${ROLLOUTS}" | tee -a "${LOG_FILE}"
echo "Max steps:    ${MAX_STEPS}" | tee -a "${LOG_FILE}"
echo "Split:        ${SPLIT}" | tee -a "${LOG_FILE}"
echo "Checkpoint root: ${CKPT_ROOT}" | tee -a "${LOG_FILE}"
echo "Log file:     ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"

for seed in ${SEEDS}; do
  SEED_CKPT_DIR="${CKPT_ROOT}/seed_${seed}"
  mkdir -p "${SEED_CKPT_DIR}"

  echo "####################################################################" | tee -a "${LOG_FILE}"
  echo "Seed ${seed}" | tee -a "${LOG_FILE}"
  echo "####################################################################" | tee -a "${LOG_FILE}"

  (
    cd "${SCRIPT_DIR}"
    PYTHONUNBUFFERED=1 python -u 06_train_policy.py \
      --policy_type bc_unet \
      --epochs "${EPOCHS}" \
      --checkpoint_dir "${SEED_CKPT_DIR}"
  ) 2>&1 | tee -a "${LOG_FILE}"

  (
    cd "${SCRIPT_DIR}"
    PYTHONUNBUFFERED=1 python -u 07_evaluate_policy.py \
      --checkpoint "${SEED_CKPT_DIR}/best_policy.pt" \
      --num_rollouts "${ROLLOUTS}" \
      --max_steps "${MAX_STEPS}" \
      --split "${SPLIT}" \
      --seed "${seed}"
  ) 2>&1 | tee -a "${LOG_FILE}"
done

echo | tee -a "${LOG_FILE}"
echo "Finished: $(date)" | tee -a "${LOG_FILE}"
echo "Done. Results in ${LOG_FILE}"
