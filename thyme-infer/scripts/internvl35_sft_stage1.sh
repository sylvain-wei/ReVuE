#!/usr/bin/env bash
# Modified for anonymous review: release configuration and launch checks.
# InternVL3.5 two-stage SFT, stage 1: provide MODEL and DATA_DIR (stage1-*.parquet).
# Original numerical training settings are preserved.
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THYME_INFER_ROOT="${THYME_INFER_ROOT:-${PKG_ROOT}/thyme-infer}"
THYME_ROOT="${THYME_INFER_ROOT}/Thyme"
source "${THYME_INFER_ROOT}/activate_thyme.sh"

MODEL="${MODEL:-${THYME_INFER_ROOT}/checkpoints/InternVL3_5-4B-Instruct}"
DATA_DIR="${DATA_DIR:-${THYME_INFER_ROOT}/data/thyme_sft_internvl/stage1}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-${THYME_ROOT}/scripts/prompt.txt}"
TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${THYME_INFER_ROOT}/outputs/internvl35_thyme_sft_stage1_${TS}}"
mkdir -p "${OUTPUT_DIR}"

shopt -s nullglob
DATA_FILES=("${DATA_DIR}"/stage1-*.parquet)
if [ "${#DATA_FILES[@]}" -eq 0 ]; then
    echo "FATAL: no stage1-*.parquet files in ${DATA_DIR}; prepare the documented SFT data first." >&2
    exit 2
fi
echo "[stage1] model=${MODEL}"
echo "[stage1] n_data_shards=${#DATA_FILES[@]} data_dir=${DATA_DIR}"
echo "[stage1] output=${OUTPUT_DIR}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export NPROC_PER_NODE=8
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export input_size="${input_size:-448}"
export max_num="${max_num:-12}"

cd "${THYME_ROOT}"
swift sft \
    --model "${MODEL}" \
    --model_type internvl3_5 \
    --dataset "${DATA_FILES[@]}" \
    --train_type full \
    --torch_dtype bfloat16 \
    --system "${SYSTEM_PROMPT}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
  --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 16 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --max_length 10240 \
    --truncation_strategy delete \
    --logging_steps 5 \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio 0.05 \
  --dataloader_num_workers 0 \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --gradient_checkpointing true \
  --report_to tensorboard \
2>&1 | tee "${OUTPUT_DIR}/internvl35_thyme_sft_stage1.log"
