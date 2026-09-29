#!/usr/bin/env bash
# Modified for anonymous review: release configuration and launch checks.
# InternVL3.5 variant: supply INTERNVL_SFT_CKPT and --teacher-ckpt PATH.
# Other algorithm and optimization options are passed through to launch_gcep.sh.
set -uo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THYME_INFER_ROOT="${THYME_INFER_ROOT:-${PKG_ROOT}/thyme-infer}"
SCRIPTS="${THYME_INFER_ROOT}/scripts"

DEFAULT_TEACHER_CKPT="${THYME_INFER_ROOT}/checkpoints/InternVL3_5-4B-Thyme-RL"

MODE="" TEACHER="" MAX_STEPS="" NUM_GENERATIONS="" OUT_DIR=""
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_EVERY_IN="${EVAL_EVERY_IN:-250}"
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --mode)            MODE="$2";            shift 2 ;;
        --teacher-ckpt)    TEACHER="$2";         shift 2 ;;
        --max-steps)       MAX_STEPS="$2";       shift 2 ;;
        --num-generations) NUM_GENERATIONS="$2"; shift 2 ;;
        --save-steps)      SAVE_STEPS="$2";      shift 2 ;;
        --eval-every)      EVAL_EVERY_IN="$2";   shift 2 ;;
        --no-archive)      EXTRA+=("$1");        shift 1 ;;
        --dry-run)         EXTRA+=("$1");        shift 1 ;;
        --*)               EXTRA+=("$1" "$2");   shift 2 ;;
        *)  if [ -z "${OUT_DIR}" ]; then OUT_DIR="$1"; else EXTRA+=("$1"); fi; shift 1 ;;
    esac
done

if [ -z "${MODE}" ]; then
    echo "FATAL: --mode is required (see launch_gcep.sh for the mode list)" >&2
    exit 1
fi
if [ -z "${TEACHER}" ]; then
    TEACHER="${GCEP_TEACHER_CKPT:-${IV_TEACHER:-${DEFAULT_TEACHER_CKPT}}}"
fi
if [ ! -d "${TEACHER}" ]; then
    echo "FATAL: teacher ckpt not found: ${TEACHER}" >&2
    echo "       pass --teacher-ckpt explicitly, or set IV_TEACHER=<path>" >&2
    exit 1
fi
if [ -z "${NUM_GENERATIONS}" ]; then
    NUM_GENERATIONS="${IV_NUM_GENERATIONS:-8}"
fi

export SCRIPTGEOM_RUNNER="${SCRIPTS}/run_baseline_scriptgeom_full_monitored_internvl.sh"
export SCRIPTGEOM_MAX_PIXELS="${SCRIPTGEOM_MAX_PIXELS:-0}"
export SCRIPTGEOM_SAVE_STEPS="${SAVE_STEPS}"
export EVAL_EVERY="${EVAL_EVERY_IN}"

STUDENT_CKPT="${INTERNVL_SFT_CKPT:-${THYME_INFER_ROOT}/checkpoints/InternVL3_5-4B-Thyme-SFT-ckpt288}"

echo "[launch-gcep-internvl] student(init) = ${STUDENT_CKPT}"
echo "[launch-gcep-internvl] teacher       = ${TEACHER}"
echo "[launch-gcep-internvl] runner        = ${SCRIPTGEOM_RUNNER}"
echo "[launch-gcep-internvl] save_steps    = ${SCRIPTGEOM_SAVE_STEPS}   eval_every = ${EVAL_EVERY}"
echo "[launch-gcep-internvl] max_pixels    = ${SCRIPTGEOM_MAX_PIXELS} (0 = InternVL dynamic tiling)"
echo "[launch-gcep-internvl] num_generations = ${NUM_GENERATIONS}   max_steps = ${MAX_STEPS:-<launch_gcep default>}"

ARGS=(--mode "${MODE}" --teacher-ckpt "${TEACHER}" --num-generations "${NUM_GENERATIONS}")
[ -n "${MAX_STEPS}" ] && ARGS+=(--max-steps "${MAX_STEPS}")
ARGS+=("${EXTRA[@]}")
[ -n "${OUT_DIR}" ] && ARGS+=("${OUT_DIR}")

exec bash "${SCRIPTS}/launch_gcep.sh" "${ARGS[@]}"
