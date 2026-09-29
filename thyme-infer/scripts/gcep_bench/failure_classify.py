# Modified for anonymous review: release paths, configuration, and documentation.
"""Failure-status taxonomy for benchmark scoring.

Only a model that answered wrongly under a healthy inference+tool chain may be
recorded as a model error. Infra/judge failures form a recovery queue and must
never be silently converted to 0.

Classes:
- MODEL_INCORRECT          — healthy chain, wrong answer (score 0)
- MODEL_EMPTY_OR_REFUSAL   — model produced no final answer (sentinel / empty)
- PARSE_ERROR              — full extraction cascade exhausted (incl. LLM 'None')
- PRED_INFRA_ERROR         — prediction crashed / timed out
- TOOL_INFRA_ERROR         — sandbox/tool infrastructure failure
- JUDGE_INFRA_ERROR        — judge unreachable or unparseable after retries (UNSCORED)
"""

from __future__ import annotations

MODEL_INCORRECT = "MODEL_INCORRECT"
MODEL_EMPTY_OR_REFUSAL = "MODEL_EMPTY_OR_REFUSAL"
PARSE_ERROR = "PARSE_ERROR"
PRED_INFRA_ERROR = "PRED_INFRA_ERROR"
TOOL_INFRA_ERROR = "TOOL_INFRA_ERROR"
JUDGE_INFRA_ERROR = "JUDGE_INFRA_ERROR"

RECOVERY_QUEUE = {PRED_INFRA_ERROR, TOOL_INFRA_ERROR, JUDGE_INFRA_ERROR}


def classify_failure(*, prediction_status: str, prediction_error: str | None,
                     score, meta: dict) -> str | None:
    """Return the failure class for a scored sample, or None when correct."""
    meta = meta or {}
    score_method = str(meta.get("score_method", ""))
    if prediction_status and prediction_status != "SUCCESS":
        err = (prediction_error or "").lower()
        if "sandbox" in err:
            return TOOL_INFRA_ERROR
        return PRED_INFRA_ERROR
    if score is None:  # UNSCORED — infra by definition (never a model result)
        return JUDGE_INFRA_ERROR
    if float(score) >= 1.0:
        return None
    if score_method == "model_no_answer":
        return MODEL_EMPTY_OR_REFUSAL
    if str(meta.get("parse_stage", "")) == "llm_none":
        return PARSE_ERROR
    return MODEL_INCORRECT
