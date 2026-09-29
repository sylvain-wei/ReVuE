#!/usr/bin/env bash
# Modified for anonymous review: release configuration and launch checks.
# Run the Qwen training schedule with periodic checkpoint evaluation and resume.
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THYME_INFER_ROOT="${THYME_INFER_ROOT:-${PKG_ROOT}/thyme-infer}"
source "${THYME_INFER_ROOT}/activate_thyme.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS="${HF_DATASETS_DISABLE_PROGRESS_BARS:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

export REWARD_API_ADDRESS="${REWARD_API_ADDRESS:-localhost}"
export QWEN_API_PORT="${QWEN_API_PORT:-8000}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-${JUDGE_MODEL:-}}"
export REWARD_API_KEY="${REWARD_API_KEY:-EMPTY}"
export VQA_WEIGHT="${VQA_WEIGHT:-1}"
export FMT_WEIGHT="${FMT_WEIGHT:-0.5}"
export CST_WEIGHT="${CST_WEIGHT:-0.5}"
export REMOTE_VLM_BASE_URL="${REMOTE_VLM_BASE_URL:-${JUDGE_BASE_URL:-http://localhost:8000/v1}}"
export REMOTE_VLM_API_KEY="${REMOTE_VLM_API_KEY:-EMPTY}"
export REMOTE_VLM_MODEL="${REMOTE_VLM_MODEL:-${JUDGE_MODEL:-}}"

export WANDB_PROJECT="${WANDB_PROJECT:-anonymous-reproduction}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export WANDB_RESUME="allow"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-thyme_rl_baseline_scriptgeom_full}"

export SCRIPTGEOM_GRAD_ACCUM="${SCRIPTGEOM_GRAD_ACCUM:-20}"
export SCRIPTGEOM_NUM_GENERATIONS="${SCRIPTGEOM_NUM_GENERATIONS:-4}"
export SCRIPTGEOM_BETA="${SCRIPTGEOM_BETA:-0.01}"
export SCRIPTGEOM_MAX_PIXELS="${SCRIPTGEOM_MAX_PIXELS:-2408448}"
export SCRIPTGEOM_EVAL_MAX_PIXELS="${SCRIPTGEOM_EVAL_MAX_PIXELS:-0}"
export SCRIPTGEOM_VLLM_GPU_UTIL="${SCRIPTGEOM_VLLM_GPU_UTIL:-0.40}"
export SCRIPTGEOM_SAVE_STEPS="${SCRIPTGEOM_SAVE_STEPS:-50}"
export SCRIPTGEOM_NUM_TRAIN_EPOCHS="${SCRIPTGEOM_NUM_TRAIN_EPOCHS:-2}"
export SCRIPTGEOM_FULL_MAX_STEPS="${SCRIPTGEOM_FULL_MAX_STEPS:-2760}"
export LOGP_MAX_SEQ_LEN="${LOGP_MAX_SEQ_LEN:-100000}"

EVAL_EVERY="${EVAL_EVERY:-50}"
RUN_ID="${RUN_ID:-thyme_rl_baseline_scriptgeom_full}"

EXTRA_ARGS=()
if [ "${1:-}" != "" ] && [[ "${1:-}" != --* ]]; then
    OUT_DIR="$1"
    shift
else
    OUT_DIR="${THYME_INFER_ROOT}/outputs/rl_baseline_scriptgeom_full_$(date +%Y%m%d_%H%M%S)"
fi
EXTRA_ARGS=("$@")
case "${OUT_DIR}" in
    /*) : ;;
    *) OUT_DIR="$(pwd)/${OUT_DIR}" ;;
esac
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

echo "[scriptgeom-full] OUT_DIR=${OUT_DIR}"
echo "[scriptgeom-full] full_max_steps=${SCRIPTGEOM_FULL_MAX_STEPS} eval_every=${EVAL_EVERY}"
echo "[scriptgeom-full] grad_accum=${SCRIPTGEOM_GRAD_ACCUM} num_gen=${SCRIPTGEOM_NUM_GENERATIONS} beta=${SCRIPTGEOM_BETA}"
echo "[scriptgeom-full] max_pixels=${SCRIPTGEOM_MAX_PIXELS} eval_max_pixels=${SCRIPTGEOM_EVAL_MAX_PIXELS} vllm=${SCRIPTGEOM_VLLM_GPU_UTIL}"
echo "[scriptgeom-full] wandb=${WANDB_ENTITY}/${WANDB_PROJECT} group=${RUN_ID}"

cd "${THYME_INFER_ROOT}/Thyme"
python "${THYME_INFER_ROOT}/scripts/run_thyme_rl_baseline_scriptgeom_monitored.py" \
    --out-dir "${OUT_DIR}" \
    --eval-every "${EVAL_EVERY}" \
    --full-max-steps "${SCRIPTGEOM_FULL_MAX_STEPS}" \
    --wandb-run-id "${RUN_ID}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/orchestrator.log"
