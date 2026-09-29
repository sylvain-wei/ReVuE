#!/usr/bin/env bash
# Activate the conda env used for ReVuE training/evaluation and export the
# package-relative paths the runners expect. Adapt CONDA_DIR / ENV_NAME / HF_HOME
# to your cluster. This file replaces the original server-specific activation script.

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_DIR="${CONDA_DIR:-${HOME}/miniconda3}"
ENV_NAME="${ENV_NAME:-thyme}"

if [ ! -x "${CONDA_DIR}/bin/conda" ]; then
    echo "[activate_thyme] ERROR: conda not found at ${CONDA_DIR} (set CONDA_DIR)" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_DIR}/etc/profile.d/conda.sh"
while [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
    conda deactivate >/dev/null 2>&1 || break
done
conda activate "${ENV_NAME}"
_act_status=$?
if [ "${_act_status}" -ne 0 ]; then
    echo "[activate_thyme] ERROR: failed to activate ${ENV_NAME}" >&2
    return "${_act_status}" 2>/dev/null || exit "${_act_status}"
fi
unset _act_status

# Optional caches / proxies: set these yourself if your cluster needs them, e.g.
#   export HF_HOME=/path/to/hf_cache
#   export http_proxy=http://your-proxy:port
export TOKENIZERS_PARALLELISM=false

export THYME_INFER_ROOT="${THYME_INFER_ROOT:-${PKG_ROOT}/thyme-infer}"
export THYME_REPO_ROOT="${THYME_INFER_ROOT}/Thyme"
export VLMEVALKIT_ROOT="${THYME_REPO_ROOT}/eval/VLMEvalKit"
export PYTHONPATH="${PKG_ROOT}/src:${THYME_REPO_ROOT}:${VLMEVALKIT_ROOT}:${PYTHONPATH:-}"

# Modified for anonymous review: disable online experiment logging in this release.
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export JUDGE_MODEL="${JUDGE_MODEL:-}"
export GCEP_JUDGE_MODEL="${GCEP_JUDGE_MODEL:-${JUDGE_MODEL}}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-${JUDGE_MODEL}}"
export REMOTE_VLM_MODEL="${REMOTE_VLM_MODEL:-${JUDGE_MODEL}}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8000/v1}"
export GCEP_JUDGE_BASE_URL="${GCEP_JUDGE_BASE_URL:-${JUDGE_BASE_URL}}"
export REMOTE_VLM_BASE_URL="${REMOTE_VLM_BASE_URL:-${JUDGE_BASE_URL}}"
export GCEP_JUDGE_API_KEY="${GCEP_JUDGE_API_KEY:-${JUDGE_API_KEY:-EMPTY}}"
export REMOTE_VLM_API_KEY="${REMOTE_VLM_API_KEY:-${JUDGE_API_KEY:-EMPTY}}"
export REWARD_API_KEY="${REWARD_API_KEY:-${JUDGE_API_KEY:-EMPTY}}"

# Sandbox scratch dir used by the code-execution sandbox (must be writable).
export SANDBOX_TMP="${SANDBOX_TMP:-${THYME_INFER_ROOT}/sandbox_scratch}"
mkdir -p "${SANDBOX_TMP}"

# VLMEvalKit dataset cache (auto-downloads benchmark TSVs on first use).
export LMUData="${LMUData:-${THYME_INFER_ROOT}/data/lmudata}"
mkdir -p "${LMUData}"

echo "[activate_thyme] env             : ${CONDA_DEFAULT_ENV}"
echo "[activate_thyme] THYME_INFER_ROOT: ${THYME_INFER_ROOT}"
echo "[activate_thyme] SANDBOX_TMP     : ${SANDBOX_TMP}"
echo "[activate_thyme] LMUData         : ${LMUData}"
python -c "import torch, transformers; print(f'[activate_thyme] torch={torch.__version__} cuda={torch.cuda.is_available()} transformers={transformers.__version__}')" 2>/dev/null ||     echo "[activate_thyme] note            : torch/transformers not installed in env yet"
