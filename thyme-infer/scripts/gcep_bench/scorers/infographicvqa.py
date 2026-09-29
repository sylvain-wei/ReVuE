# Modified for anonymous review: release paths, configuration, and documentation.
"""InfographicVQA-val scorer. OFFICIAL_DETERMINISTIC.

ANLS: lowercase + normalized Levenshtein + 0.5 threshold, max over multiple GT
answers. NO aggressive normalization (no punctuation/article stripping).
RRC test server is out of scope; val split only (2801).
"""

from __future__ import annotations

from typing import Optional

from ..common import extract_answer_tag, is_model_no_answer
from .anls_lite import anls_score

PROTOCOL_NAME = "infographicvqa_val_anls_official_v1"
SCORING_LAYER = "OFFICIAL_DETERMINISTIC"
REQUIRES_JUDGE = False


def _prediction_view(prediction: str) -> str:
    ans = extract_answer_tag(prediction)
    return ans if ans else (prediction or "").strip()


def _gold_list(sample: dict) -> list[str]:
    answers = sample.get("answers", sample.get("answer", ""))
    if isinstance(answers, str):
        # TSV stores multi-GT joined by "|"; single GT stays a 1-element list.
        return [a.strip() for a in answers.split("|") if a.strip()]
    if isinstance(answers, (list, tuple)):
        return [str(a).strip() for a in answers if str(a).strip()]
    return [str(answers).strip()]


def score_sample(sample: dict, prediction: str, **ctx) -> tuple[Optional[float], dict]:
    golds = _gold_list(sample)
    if is_model_no_answer(prediction):
        return 0.0, {"status": "SCORED", "score_method": "model_no_answer",
                     "score_continuous": 0.0, "gt": golds, "extracted_prediction": ""}
    pred = _prediction_view(prediction)
    if pred == "":
        return None, {"status": "UNSCORED", "error": "empty prediction", "gt": golds}
    score = anls_score(prediction=pred.lower(), gold_labels=[g.lower() for g in golds],
                       threshold=0.5)
    return score, {
        "status": "SCORED",
        "score_method": "anls_lower_thresh0.5_max_over_golds",
        "score_continuous": score,
        "gt": golds,
        "extracted_prediction": pred[:500],
    }


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["correct"] is not None]
    n = len(scored)
    return {
        "anls": (sum(r["correct"] for r in scored) / n) if n else 0.0,
        "n": len(rows),
        "n_scored": n,
    }
