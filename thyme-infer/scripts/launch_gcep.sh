#!/usr/bin/env bash
# Modified for anonymous review: release configuration and launch checks.
# Launch released ReVuE/OPD variants with an explicit local teacher checkpoint.
# Usage: bash launch_gcep.sh --mode strong_opd_gcep_combined_v2 --teacher-ckpt PATH [OPTIONS] [OUT_DIR]
# Set JUDGE_MODEL, judge endpoint variables, SFT_CKPT and RL_DATA_DIR first.
# Matched-teacher baselines are not part of this release.
set -uo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THYME_INFER_ROOT="${THYME_INFER_ROOT:-${PKG_ROOT}/thyme-infer}"
SCRIPTS="${THYME_INFER_ROOT}/scripts"

MODE=""
TEACHER_CKPT="${GCEP_TEACHER_CKPT:-}"
LR=""
MAX_STEPS=""
NUM_GENERATIONS="${GCEP_NUM_GENERATIONS:-8}"
SEED=42
LAMBDA_OPD=0.01
TOPK=32
JUDGE_CONCURRENCY=4
ARCHIVE=1

while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --mode)              MODE="$2";              shift 2 ;;
        --teacher-ckpt)      TEACHER_CKPT="$2";      shift 2 ;;
        --lr)                LR="$2";                shift 2 ;;
        --max-steps)         MAX_STEPS="$2";         shift 2 ;;
        --num-generations)   NUM_GENERATIONS="$2";   shift 2 ;;
        --seed)              SEED="$2";              shift 2 ;;
        --lambda-opd)        LAMBDA_OPD="$2";        shift 2 ;;
        --topk)              TOPK="$2";              shift 2 ;;
        --judge-concurrency) JUDGE_CONCURRENCY="$2"; shift 2 ;;
        --no-archive)        ARCHIVE=0;              shift 1 ;;
        --dry-run)           DRYRUN=1;               shift 1 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

case "${MODE}" in
    vision_opd_mt|vzero_mt|vad_mt) echo "FATAL: matched-teacher baseline implementations are not included in this release." >&2; exit 2 ;;
    strong_opd_clean|strong_opd_gcep|strong_opd_gcep_ps|opsd_gcep|grpo_gcep_opd|grpo_gcep_ps_opd|strong_opd_gcep_v2|strong_opd_gcep_impact_v2|strong_opd_gcep_combined_v2|strong_opd_gt_privilege) ;;
    "") echo "FATAL: --mode is required (strong_opd_clean|strong_opd_gcep|strong_opd_gcep_ps|opsd_gcep|grpo_gcep_opd|grpo_gcep_ps_opd|strong_opd_gcep_v2|strong_opd_gcep_impact_v2|strong_opd_gcep_combined_v2|strong_opd_gt_privilege)" >&2; exit 1 ;;
    *) echo "FATAL: unknown --mode ${MODE}" >&2; exit 1 ;;
esac

case "${MODE}" in
    strong_opd_clean)   EXP_PREFIX="A0_strong_opd_clean" ;;
    strong_opd_gcep)    EXP_PREFIX="A1_strong_opd_gcep" ;;
    strong_opd_gcep_ps) EXP_PREFIX="A2_strong_opd_gcep_ps" ;;
    opsd_gcep)          EXP_PREFIX="B1_opsd_gcep" ;;
    grpo_gcep_opd)      EXP_PREFIX="C1_grpo_gcep_opd" ;;
    grpo_gcep_ps_opd)   EXP_PREFIX="C2_grpo_gcep_ps_opd" ;;
    strong_opd_gcep_v2)        EXP_PREFIX="A1v2_strong_opd_gcep_v2" ;;
    strong_opd_gcep_impact_v2) EXP_PREFIX="A2v2_strong_opd_gcep_impact_v2" ;;
    strong_opd_gcep_combined_v2) EXP_PREFIX="A3v2_strong_opd_gcep_combined_v2" ;;
    strong_opd_gt_privilege)    EXP_PREFIX="GTONLY_strong_opd_gt_privilege" ;;
esac

if [[ "${MODE}" == grpo_* ]]; then
    LR_VAL="${LR:-5e-7}"
    MAX_STEPS_VAL="${MAX_STEPS:-2760}"
    BETA_VAL="${SCRIPTGEOM_BETA:-0.01}"
    EPOCHS_VAL="${SCRIPTGEOM_NUM_TRAIN_EPOCHS:-2}"
else
    LR_VAL="${LR:-1e-6}"
    MAX_STEPS_VAL="${MAX_STEPS:-1075}"
    BETA_VAL="${SCRIPTGEOM_BETA:-0}"
    EPOCHS_VAL="${SCRIPTGEOM_NUM_TRAIN_EPOCHS:-1}"
fi

if [[ "${SEED}" != "42" ]]; then
    echo "WARNING: --seed ${SEED} requested, but the scriptgeom runner does not expose a"
    echo "         seed passthrough and swift TrainingArguments default seed is already 42"
    echo "         This release will continue with seed=42." >&2
fi

source "${THYME_INFER_ROOT}/activate_thyme.sh" || exit 1
if [[ -z "${TEACHER_CKPT}" || ! -d "${TEACHER_CKPT}" ]]; then
    echo "FATAL: provide an existing local teacher checkpoint with --teacher-ckpt or GCEP_TEACHER_CKPT." >&2
    exit 2
fi
for NAME in JUDGE_MODEL_NAME REMOTE_VLM_MODEL; do
    if [[ -z "${!NAME:-}" || "${!NAME}" == *'<'* || "${!NAME}" == your-* ]]; then
        echo "FATAL: set JUDGE_MODEL to the served model alias, or set ${NAME} explicitly." >&2
        exit 2
    fi
done
case "${MODE}" in
    strong_opd_clean|strong_opd_gt_privilege) ;;
    *) if [[ -z "${GCEP_JUDGE_MODEL:-}" || "${GCEP_JUDGE_MODEL}" == *'<'* || "${GCEP_JUDGE_MODEL}" == your-* ]]; then
           echo "FATAL: set JUDGE_MODEL or GCEP_JUDGE_MODEL to the critic's served model alias." >&2
           exit 2
       fi ;;
esac
if [[ -z "${REWARD_API_ADDRESS:-}" || "${REWARD_API_ADDRESS}" == *'<'* ]]; then
    echo "FATAL: set REWARD_API_ADDRESS to the reward judge host; set QWEN_API_PORT if needed." >&2
    exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${EXP_PREFIX}_${TS}}"

if [ "${1:-}" != "" ] && [[ "${1:-}" != --* ]]; then
    OUT_DIR="$1"; shift
else
    OUT_DIR="${THYME_INFER_ROOT}/outputs/${RUN_ID}"
fi
EXTRA_ARGS=("$@")
if [[ "${DRYRUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=("--dry-run")
fi

mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
PID_FILE="${OUT_DIR}/launch.pid"
LAUNCH_LOG="${OUT_DIR}/launch_gcep.log"

echo "[launch-gcep] OUT_DIR         = ${OUT_DIR}"
echo "[launch-gcep] RUN_ID          = ${RUN_ID}"
echo "[launch-gcep] mode            = ${MODE}"
echo "[launch-gcep] teacher_ckpt    = ${TEACHER_CKPT:-<mode default>}"
echo "[launch-gcep] lr=${LR_VAL} max_steps=${MAX_STEPS_VAL} epochs=${EPOCHS_VAL} beta=${BETA_VAL}"
echo "[launch-gcep] num_generations = ${NUM_GENERATIONS}  seed=42"
echo "[launch-gcep] lambda_opd=${LAMBDA_OPD} topk=${TOPK} judge_concurrency=${JUDGE_CONCURRENCY} archive=${ARCHIVE}"
echo "[launch-gcep] PID_FILE        = ${PID_FILE}"

echo "=== [1/4] GPU free check ==="
STALE_PROCS=$(ps aux | grep -E "rlhf_ds|deepspeed.*rlhf" | grep -v grep | wc -l)
if [ "${STALE_PROCS}" -gt 0 ]; then
    echo "FATAL: found ${STALE_PROCS} stale rlhf_ds/deepspeed processes. Refusing to launch." >&2
    ps aux | grep -E "rlhf_ds|deepspeed.*rlhf" | grep -v grep | head -5 >&2
    echo "To clean: kill by PID (do NOT use pkill -9 -f rlhf_ds)." >&2
    exit 1
fi
GPU_BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>100{c++} END{print c+0}')
if [ "${GPU_BUSY}" -gt 0 ]; then
    echo "FATAL: ${GPU_BUSY} GPUs have >100 MiB used. Another job may be running." >&2
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2
    exit 1
fi
echo "[launch-gcep] GPUs free, no stale procs."

echo "=== [2/4] export env ==="
export GCEP_TRAINING_MODE="${MODE}"
export GCEP_LAMBDA_OPD="${LAMBDA_OPD}"
export GCEP_TOPK="${TOPK}"
export GCEP_JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY}"
export GCEP_ARCHIVE="${ARCHIVE}"
export GCEP_EXPERIMENT_ID="${RUN_ID}"
if [[ -n "${TEACHER_CKPT}" ]]; then
    export GCEP_TEACHER_CKPT="${TEACHER_CKPT}"
fi

export SCRIPTGEOM_LR="${LR_VAL}"
export SCRIPTGEOM_FULL_MAX_STEPS="${MAX_STEPS_VAL}"
export SCRIPTGEOM_NUM_TRAIN_EPOCHS="${EPOCHS_VAL}"
export SCRIPTGEOM_BETA="${BETA_VAL}"
export SCRIPTGEOM_NUM_GENERATIONS="${NUM_GENERATIONS}"
export SCRIPTGEOM_GRAD_ACCUM="${SCRIPTGEOM_GRAD_ACCUM:-20}"
export VQA_NORM=0
export VQA_WEIGHT=1
export RUN_ID
export REWARD_API_ADDRESS="${REWARD_API_ADDRESS:-localhost}"
export QWEN_API_PORT="${QWEN_API_PORT:-8000}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-${JUDGE_MODEL:-}}"
export REWARD_API_KEY="${REWARD_API_KEY:-EMPTY}"
export REMOTE_VLM_BASE_URL="${REMOTE_VLM_BASE_URL:-${JUDGE_BASE_URL:-http://localhost:8000/v1}}"
export REMOTE_VLM_API_KEY="${REMOTE_VLM_API_KEY:-EMPTY}"
export REMOTE_VLM_MODEL="${REMOTE_VLM_MODEL:-${JUDGE_MODEL:-}}"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,${REWARD_API_ADDRESS}"
export NO_PROXY="$no_proxy"
export SCRIPTGEOM_SAVE_STEPS="${SCRIPTGEOM_SAVE_STEPS:-50}"
export EVAL_EVERY="${EVAL_EVERY:-250}"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export WANDB_PROJECT="${WANDB_PROJECT:-anonymous-reproduction}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_ID}}"
echo "[launch-gcep] GCEP_TRAINING_MODE=${MODE} RUN_ID=${RUN_ID}"

echo "=== [3/4] launch with nohup setsid ==="
SCRIPTGEOM_RUNNER="${SCRIPTGEOM_RUNNER:-${SCRIPTS}/run_baseline_scriptgeom_full_monitored.sh}"
nohup setsid bash "${SCRIPTGEOM_RUNNER}" \
    "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}" \
    > "${LAUNCH_LOG}" 2>&1 &
LAUNCH_PID=$!
echo "${LAUNCH_PID}" > "${PID_FILE}"
disown "${LAUNCH_PID}" 2>/dev/null || true
echo "[launch-gcep] launched PID=${LAUNCH_PID}"
echo "[launch-gcep] PID written to ${PID_FILE}"
echo "[launch-gcep] log: ${LAUNCH_LOG}"

echo "=== [4/4] verify alive (5s) ==="
sleep 5
if kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    echo "[launch-gcep] OK: PID ${LAUNCH_PID} alive after 5s."
    echo "[launch-gcep] To stop:   kill -TERM \$(cat ${PID_FILE})"
    echo "[launch-gcep] To resume: re-run with same OUT_DIR (monitor auto-resumes from latest checkpoint)."
    echo "[launch-gcep] To watch:  tail -f ${LAUNCH_LOG}"
else
    echo "FATAL: PID ${LAUNCH_PID} died within 5s. Check ${LAUNCH_LOG}:" >&2
    tail -20 "${LAUNCH_LOG}" >&2
    exit 1
fi
