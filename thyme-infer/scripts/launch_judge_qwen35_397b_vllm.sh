#!/usr/bin/env bash
# Modified for anonymous review: explicit serving environment and portable configuration.
# Run in a separate Qwen3.5-capable vLLM environment; the training environment is insufficient.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL="${MODEL:?Set MODEL to the local Qwen3.5-397B-A17B-FP8 checkpoint directory}"
API_KEY="${API_KEY:?Set a nonempty API_KEY for the judge service}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-397b-judge}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":8,\"video\":0}}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-0}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/outputs/judge_vllm}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/server.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/server.pid}"
# Both the version check and serving use this interpreter.
JUDGE_PYTHON="${JUDGE_PYTHON:-python}"
command -v "$JUDGE_PYTHON" >/dev/null || { echo "JUDGE_PYTHON is unavailable" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "MODEL is not a local directory: $MODEL" >&2; exit 2; }
[[ "$API_KEY" != EMPTY ]] || { echo "Set a private API_KEY, not EMPTY" >&2; exit 2; }
[[ "$RUN_IN_BACKGROUND" == 0 || "$RUN_IN_BACKGROUND" == 1 ]] || { echo "RUN_IN_BACKGROUND must be 0 or 1" >&2; exit 2; }
# The supplement reports multiple serving builds; none is a fresh-install certification.
# Check actual architecture availability rather than asserting a version threshold is sufficient.
"$JUDGE_PYTHON" - "$HOST" "$PORT" <<'PY'
import socket, sys
import vllm, transformers
from vllm.model_executor.models.registry import ModelRegistry
arch = "Qwen3_5MoeForConditionalGeneration"
if arch not in ModelRegistry.get_supported_archs():
    raise SystemExit(f"Serving environment does not register {arch}")
print(f"[judge] vllm={vllm.__version__} transformers={transformers.__version__}")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind((sys.argv[1], int(sys.argv[2])))
PY
export CUDA_VISIBLE_DEVICES MAX_PIXELS TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true WANDB_MODE=disabled
export no_proxy="${no_proxy:-${NO_PROXY:-}},127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
mkdir -p "$LOG_DIR" "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
CMD=("$JUDGE_PYTHON" -m vllm.entrypoints.openai.api_server
  --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST" --port "$PORT" --api-key "$API_KEY"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT"
  --mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB" --trust-remote-code)
echo "[judge] alias=$SERVED_MODEL_NAME host=$HOST port=$PORT log=$LOG_FILE"
echo "[judge] Client requests must set chat_template_kwargs.enable_thinking=false."
if [[ "$RUN_IN_BACKGROUND" == 1 ]]; then
  command -v setsid >/dev/null || { echo "Background mode requires setsid (Linux)" >&2; exit 2; }
  setsid nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  echo "$!" >"$PID_FILE"
  echo "[judge] Started process; this does not confirm readiness. PID file: $PID_FILE"
else
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
fi
