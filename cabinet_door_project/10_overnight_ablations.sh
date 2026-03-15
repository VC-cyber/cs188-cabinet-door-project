#!/usr/bin/env bash
set -euo pipefail

# OpenCabinet overnight ablation runner
# -------------------------------------
# Runs multiple ablation configs over multiple seeds, with live output and saved logs.
#
# Usage:
#   bash 10_overnight_ablations.sh
#   bash 10_overnight_ablations.sh "0 1 2 3 4 5" 30 50 "500 600" pretrain
#
# Args:
#   $1 seeds (space-separated)        default: "0 1 2 3 4"
#   $2 epochs_default                 default: 30
#   $3 num_rollouts                   default: 50
#   $4 max_steps_list (space-sep)     default: "500 600"
#   $5 split                          default: pretrain

SEEDS="${1:-0 1 2 3 4}"
EPOCHS_DEFAULT="${2:-30}"
ROLLOUTS="${3:-50}"
MAX_STEPS_LIST="${4:-500 600}"
SPLIT="${5:-pretrain}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/sweep_results"
mkdir -p "${OUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_DIR}/ablations_${TS}"
mkdir -p "${RUN_DIR}"

LOG_FILE="${RUN_DIR}/ablations.log"
CSV_FILE="${RUN_DIR}/results.csv"
SUMMARY_MD="${RUN_DIR}/summary.md"
CKPT_ROOT="/tmp/cabinet_ablations_${TS}"
mkdir -p "${CKPT_ROOT}"

# Config format:
# name|epochs|chunk_size|n_action_steps|handle_mode
CONFIGS=(
  "baseline|${EPOCHS_DEFAULT}|16|8|both"
  "epochs20|20|16|8|both"
  "epochs50|50|16|8|both"
  "chunk12_6|${EPOCHS_DEFAULT}|12|6|both"
  "chunk20_10|${EPOCHS_DEFAULT}|20|10|both"
  "relative_only|${EPOCHS_DEFAULT}|16|8|relative_only"
)

echo "config,seed,handle_mode,epochs,chunk_size,n_action_steps,max_steps,successes,episodes,success_rate,status" > "${CSV_FILE}"

log() {
  echo "$@" | tee -a "${LOG_FILE}"
}

run_and_tee() {
  # shellcheck disable=SC2068
  "$@" 2>&1 | tee -a "${LOG_FILE}"
}

parse_eval_from_log() {
  local tmp_log="$1"
  local succ_line rate_line successes episodes rate
  succ_line="$(awk '/Successes:/ {line=$0} END {print line}' "${tmp_log}")"
  rate_line="$(awk '/Success rate:/ {line=$0} END {print line}' "${tmp_log}")"

  successes="$(echo "${succ_line}" | awk '{print $2}' | cut -d'/' -f1)"
  episodes="$(echo "${succ_line}" | awk '{print $2}' | cut -d'/' -f2)"
  rate="$(echo "${rate_line}" | awk '{print $3}' | tr -d '%')"

  if [[ -z "${successes}" || -z "${episodes}" || -z "${rate}" ]]; then
    echo "NA,NA,NA"
  else
    echo "${successes},${episodes},${rate}"
  fi
}

log "===================================================================="
log "OpenCabinet Overnight Ablations"
log "===================================================================="
log "Started:         $(date)"
log "Script dir:      ${SCRIPT_DIR}"
log "Seeds:           ${SEEDS}"
log "Default epochs:  ${EPOCHS_DEFAULT}"
log "Eval rollouts:   ${ROLLOUTS}"
log "Max steps list:  ${MAX_STEPS_LIST}"
log "Eval split:      ${SPLIT}"
log "Checkpoint root: ${CKPT_ROOT}"
log "Run dir:         ${RUN_DIR}"
log "===================================================================="

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r CFG_NAME CFG_EPOCHS CFG_CHUNK CFG_EXEC CFG_HANDLE_MODE <<< "${cfg}"

  log ""
  log "####################################################################"
  log "Config: ${CFG_NAME}  (epochs=${CFG_EPOCHS}, chunk=${CFG_CHUNK}, exec=${CFG_EXEC}, handle_mode=${CFG_HANDLE_MODE})"
  log "####################################################################"

  for seed in ${SEEDS}; do
    SEED_DIR="${CKPT_ROOT}/${CFG_NAME}/seed_${seed}"
    mkdir -p "${SEED_DIR}"
    BEST_CKPT="${SEED_DIR}/best_policy.pt"

    log ""
    log "---- Train: config=${CFG_NAME}, seed=${seed} ----"
    (
      cd "${SCRIPT_DIR}"
      CABINET_HANDLE_FEATURE_MODE="${CFG_HANDLE_MODE}" PYTHONUNBUFFERED=1 python -u 06_train_policy.py \
        --policy_type bc_unet \
        --epochs "${CFG_EPOCHS}" \
        --chunk_size "${CFG_CHUNK}" \
        --n_action_steps "${CFG_EXEC}" \
        --seed "${seed}" \
        --checkpoint_dir "${SEED_DIR}"
    ) 2>&1 | tee -a "${LOG_FILE}"

    if [[ ! -f "${BEST_CKPT}" ]]; then
      log "Train failed: missing checkpoint ${BEST_CKPT}"
      for max_steps in ${MAX_STEPS_LIST}; do
        echo "${CFG_NAME},${seed},${CFG_HANDLE_MODE},${CFG_EPOCHS},${CFG_CHUNK},${CFG_EXEC},${max_steps},NA,NA,NA,train_failed" >> "${CSV_FILE}"
      done
      continue
    fi

    for max_steps in ${MAX_STEPS_LIST}; do
      log ""
      log "---- Eval: config=${CFG_NAME}, seed=${seed}, max_steps=${max_steps} ----"

      TMP_EVAL_LOG="$(mktemp)"
      (
        cd "${SCRIPT_DIR}"
        CABINET_HANDLE_FEATURE_MODE="${CFG_HANDLE_MODE}" PYTHONUNBUFFERED=1 python -u 07_evaluate_policy.py \
          --checkpoint "${BEST_CKPT}" \
          --num_rollouts "${ROLLOUTS}" \
          --max_steps "${max_steps}" \
          --split "${SPLIT}" \
          --seed "${seed}"
      ) 2>&1 | tee -a "${LOG_FILE}" | tee "${TMP_EVAL_LOG}"

      parsed="$(parse_eval_from_log "${TMP_EVAL_LOG}")"
      rm -f "${TMP_EVAL_LOG}"
      successes="$(echo "${parsed}" | cut -d',' -f1)"
      episodes="$(echo "${parsed}" | cut -d',' -f2)"
      rate="$(echo "${parsed}" | cut -d',' -f3)"

      if [[ "${rate}" == "NA" ]]; then
        status="eval_parse_failed"
      else
        status="ok"
      fi

      echo "${CFG_NAME},${seed},${CFG_HANDLE_MODE},${CFG_EPOCHS},${CFG_CHUNK},${CFG_EXEC},${max_steps},${successes},${episodes},${rate},${status}" >> "${CSV_FILE}"
    done
  done
done

# Build markdown summary from CSV (top 15 by success rate)
{
  echo "# Overnight Ablation Summary"
  echo
  echo "- Started: $(date -r "${LOG_FILE}" 2>/dev/null || date)"
  echo "- Finished: $(date)"
  echo "- Full log: \`${LOG_FILE}\`"
  echo "- CSV: \`${CSV_FILE}\`"
  echo
  echo "## Top Results"
  echo
  echo "| config | seed | handle_mode | epochs | chunk | exec | max_steps | successes | episodes | rate |"
  echo "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"
  awk -F',' 'NR>1 && $10 != "NA" {print}' "${CSV_FILE}" \
    | sort -t',' -k10,10nr \
    | head -n 15 \
    | awk -F',' '{printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s%% |\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10)}'
} > "${SUMMARY_MD}"

log ""
log "===================================================================="
log "Done"
log "===================================================================="
log "Log:     ${LOG_FILE}"
log "CSV:     ${CSV_FILE}"
log "Summary: ${SUMMARY_MD}"
