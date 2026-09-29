#!/usr/bin/env bash
# Modified for anonymous review: release configuration and launch checks.
# Run the InternVL training schedule with its required vLLM attention patch.
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THYME_INFER_ROOT="${THYME_INFER_ROOT:-${PKG_ROOT}/thyme-infer}"
source "${THYME_INFER_ROOT}/activate_thyme.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS="${HF_DATASETS_DISABLE_PROGRESS_BARS:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

export PYTHONPATH="${THYME_INFER_ROOT}/scripts/internvl_vllm_patch${PYTHONPATH:+:${PYTHONPATH}}"

export REWARD_API_ADDRESS="${REWARD_API_ADDRESS:-localhost}"
export QWEN_API_PORT="${QWEN_API_PORT:-8000}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-${JUDGE_MODEL:-}}"
export REWARD_API_KEY="${REWARD_API_KEY:-EMPTY}"
export REWARD_JUDGE_ENABLE_THINKING="${REWARD_JUDGE_ENABLE_THINKING:-0}"
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
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-internvl35_thyme_grpo_baseline}"

export SCRIPTGEOM_GRAD_ACCUM="${SCRIPTGEOM_GRAD_ACCUM:-20}"
export SCRIPTGEOM_NUM_GENERATIONS="${SCRIPTGEOM_NUM_GENERATIONS:-4}"
export SCRIPTGEOM_BETA="${SCRIPTGEOM_BETA:-0.01}"
export SCRIPTGEOM_MAX_PIXELS="${SCRIPTGEOM_MAX_PIXELS:-0}"
export SCRIPTGEOM_EVAL_MAX_PIXELS="${SCRIPTGEOM_EVAL_MAX_PIXELS:-0}"
export SCRIPTGEOM_VLLM_GPU_UTIL="${SCRIPTGEOM_VLLM_GPU_UTIL:-0.40}"
export SCRIPTGEOM_SAVE_STEPS="${SCRIPTGEOM_SAVE_STEPS:-50}"
export SCRIPTGEOM_NUM_TRAIN_EPOCHS="${SCRIPTGEOM_NUM_TRAIN_EPOCHS:-2}"
export SCRIPTGEOM_FULL_MAX_STEPS="${SCRIPTGEOM_FULL_MAX_STEPS:-2760}"
export LOGP_MAX_SEQ_LEN="${LOGP_MAX_SEQ_LEN:-100000}"

EVAL_EVERY="${EVAL_EVERY:-50}"
RUN_ID="${RUN_ID:-internvl35_thyme_grpo_baseline}"

EXTRA_ARGS=()
if [ "${1:-}" != "" ] && [[ "${1:-}" != --* ]]; then
    OUT_DIR="$1"
    shift
else
    OUT_DIR="${THYME_INFER_ROOT}/outputs/rl_baseline_internvl35_vqanorm0_beta0.01_$(date +%Y%m%d_%H%M%S)"
fi
EXTRA_ARGS=("$@")
case "${OUT_DIR}" in
    /*) : ;;
    *) OUT_DIR="$(pwd)/${OUT_DIR}" ;;
esac
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

echo "[scriptgeom-full-internvl] OUT_DIR=${OUT_DIR}"
echo "[scriptgeom-full-internvl] full_max_steps=${SCRIPTGEOM_FULL_MAX_STEPS} eval_every=${EVAL_EVERY}"
echo "[scriptgeom-full-internvl] grad_accum=${SCRIPTGEOM_GRAD_ACCUM} num_gen=${SCRIPTGEOM_NUM_GENERATIONS} beta=${SCRIPTGEOM_BETA}"
echo "[scriptgeom-full-internvl] judge=${JUDGE_MODEL_NAME} vllm_util=${SCRIPTGEOM_VLLM_GPU_UTIL}"
echo "[scriptgeom-full-internvl] PYTHONPATH patch=${THYME_INFER_ROOT}/scripts/internvl_vllm_patch"
echo "[scriptgeom-full-internvl] wandb=${WANDB_ENTITY}/${WANDB_PROJECT} group=${RUN_ID}"

cd "${THYME_INFER_ROOT}/Thyme"
python "${THYME_INFER_ROOT}/scripts/run_thyme_rl_baseline_scriptgeom_monitored_internvl.py" \
    --out-dir "${OUT_DIR}" \
    --eval-every "${EVAL_EVERY}" \
    --full-max-steps "${SCRIPTGEOM_FULL_MAX_STEPS}" \
    --wandb-run-id "${RUN_ID}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/orchestrator.log"
