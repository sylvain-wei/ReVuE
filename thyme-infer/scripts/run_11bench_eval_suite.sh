#!/usr/bin/env bash
# Modified for anonymous review: release configuration and launch checks.
# Evaluate the 11 benchmark families (VisualProbe has three difficulty splits).
# Usage: bash run_11bench_eval_suite.sh CKPT_DIR OUT_DIR [BENCHMARK ...]
# Set SMOKE_N to a positive integer for a small diagnostic run.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <CKPT_DIR> <OUT_BASE> [benchmarks...]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PKG_ROOT}/thyme-infer/activate_thyme.sh"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8000/v1}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
export REMOTE_VLM_API_KEY="${JUDGE_API_KEY}"
export JUDGE_MODEL="${JUDGE_MODEL:-}"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export NO_PROXY="$no_proxy"

CKPT_DIR="$1"; shift || { echo "usage: $0 <CKPT_DIR> <OUT_BASE> [benchmarks...]"; exit 1; }
OUT_BASE="$1"; shift || { echo "usage: $0 <CKPT_DIR> <OUT_BASE> [benchmarks...]"; exit 1; }
if [ $# -gt 0 ]; then
    BENCHES=("$@")
else
    BENCHES=(HRBench4K HRBench8K VStarBench
             TreeBench VisualProbe_Easy VisualProbe_Medium VisualProbe_Hard
             MathVista_MINI MathVerseVO VisuLogic_GCEP HallusionBench_GCEP
             ChartQAPro InfographicVQA_val)
fi
if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "FATAL: checkpoint directory does not exist: ${CKPT_DIR}" >&2
    exit 2
fi
for BENCH in "${BENCHES[@]}"; do
    case "${BENCH}" in
        MathVista_MINI|VisualProbe_*|HallusionBench_GCEP|MathVerseVO|VisuLogic_GCEP)
            if [[ -z "${JUDGE_MODEL}" || "${JUDGE_MODEL}" == *'<'* || "${JUDGE_MODEL}" == your-* ]]; then
                echo "FATAL: set JUDGE_MODEL to the served model alias for ${BENCH}." >&2
                exit 2
            fi ;;
    esac
    case "${BENCH}" in
        VisualProbe_*|ChartQAPro|InfographicVQA_val|HallusionBench_GCEP|MathVerseVO|VisuLogic_GCEP)
            if [[ ! -f "${LMUData}/${BENCH}.tsv" ]]; then
                echo "FATAL: missing ${LMUData}/${BENCH}.tsv; run the matching gcep_bench/datasets/builders.py command in README first." >&2
                exit 2
            fi ;;
    esac
done

SMOKE_N="${SMOKE_N:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"

mkdir -p "${OUT_BASE}"
echo "[eval-suite] ckpt=${CKPT_DIR}"
echo "[eval-suite] out=${OUT_BASE}"
echo "[eval-suite] benchmarks=${BENCHES[*]} smoke_n=${SMOKE_N}"

for BENCH in "${BENCHES[@]}"; do
    LOG="${OUT_BASE}/${BENCH}_$(date +%Y%m%d_%H%M%S).log"
    echo "[eval-suite] === ${BENCH} (log: ${LOG}) ==="
    ARGS=(
        --model-path "${CKPT_DIR}"
        --output-dir "${OUT_BASE}"
        --benchmarks "${BENCH}"
        --prompt-recipe official_thyme_vlmevalkit
        --no-ats
        --rank-wait-timeout-sec 43200
    )
    case "${BENCH}" in
        MathVista_MINI|VisualProbe_*|HallusionBench_GCEP|MathVerseVO|VisuLogic_GCEP)
            ARGS+=(--semantic-judge --judge-base-url "${JUDGE_BASE_URL}"
                   --judge-model "${JUDGE_MODEL}")
            ;;
    esac
    if [ "${SMOKE_N}" != "0" ]; then
        ARGS+=(--max-examples "${SMOKE_N}")
    fi
    if [ "${MAX_NEW_TOKENS}" != "0" ]; then
        ARGS+=(--max-new-tokens "${MAX_NEW_TOKENS}")
    fi
    set +e
    torchrun --standalone --nproc_per_node=8 \
        "${REPO_ROOT}/scripts/phase3_eval_opd.py" "${ARGS[@]}" 2>&1 | tee "${LOG}"
    rc=${PIPESTATUS[0]}
    set -e
    echo "[eval-suite] === ${BENCH} done (exit=${rc}) ==="
    if [ "${rc}" != "0" ]; then
        echo "[eval-suite] ABORT: ${BENCH} failed with exit=${rc} (gate: stop pipeline)"
        exit "${rc}"
    fi
    EVAL_JSON="${OUT_BASE}/eval_${BENCH}.json"
    if ! python3 - "${EVAL_JSON}" "${BENCH}" <<'PYRESULT'
import json
import math
import sys
from pathlib import Path

path, benchmark = Path(sys.argv[1]), sys.argv[2]
try:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")
    if result.get("benchmark", benchmark) != benchmark:
        raise ValueError("result benchmark does not match the requested benchmark")
    status_required = benchmark in {
        "TreeBench", "ZoomBench", "VisualProbe_Easy", "VisualProbe_Medium",
        "VisualProbe_Hard", "ReasonMapPlus", "ChartQAPro", "InfographicVQA_val",
        "PerceptionBench", "HallusionBench_GCEP", "MathVerseVO", "VisuLogic_GCEP",
    }
    if "run_status" in result or status_required:
        if result.get("run_status") != "COMPLETE":
            raise ValueError(f"run_status is {result.get('run_status', 'MISSING')!r}")
    accuracy = result.get("accuracy")
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        raise ValueError("accuracy must be a numeric score")
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be finite and between 0 and 1")
    if result.get("coverage_ok") is False:
        raise ValueError("prediction coverage is incomplete")
except (OSError, ValueError, TypeError) as exc:
    print(f"[eval-suite] invalid or missing result for {benchmark}: {exc}", file=sys.stderr)
    sys.exit(1)
PYRESULT
    then
        echo "[eval-suite] ABORT: ${BENCH} has no valid complete evaluation result." >&2
        exit 3
    fi
done
echo "[eval-suite] ALL DONE → ${OUT_BASE}"
