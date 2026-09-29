# Modified for anonymous review: release paths, configuration, and documentation.
"""TreeBench scorer. OFFICIAL_DETERMINISTIC.

Answer: extract <answer> letter A-F, exact match with GT. No judge.
Metrics: answer_accuracy (+ per-dimension breakdown mirroring
upstream get_dimension_rating), tool_crop_IoU as CUSTOM_DIAGNOSTIC
(NOT_APPLICABLE when no tool trace, never 0).
"""

from __future__ import annotations

import re
from typing import Optional

from ..common import extract_answer_tag, is_model_no_answer

PROTOCOL_NAME = "treebench_official_deterministic_v1"
SCORING_LAYER = "OFFICIAL_DETERMINISTIC"
REQUIRES_JUDGE = False

P_SUBTASKS = ["Attributes", "Material", "Physical State", "Object Retrieval", "OCR"]
R_SUBTASKS = ["Perspective Transform", "Ordering", "Contact and Occlusion",
              "Spatial Containment", "Comparison"]

_LETTER_RE = re.compile(r"\b([A-F])\b")


def _extract_letter(prediction: str) -> str:
    """Extract answer letter A-F: <answer> tag first, then last standalone letter."""
    ans = extract_answer_tag(prediction)
    cand = ans if ans else (prediction or "")
    m = _LETTER_RE.search(ans.strip()) if ans else None
    if m:
        return m.group(1)
    if not ans:
        # fall back: last standalone A-F letter in the tail of the response
        tail = (prediction or "").strip()[-200:]
        matches = _LETTER_RE.findall(tail)
        if matches:
            return matches[-1]
    return ""


def score_sample(sample: dict, prediction: str, **ctx) -> tuple[Optional[float], dict]:
    gt = str(sample.get("answer", "")).strip().upper()
    pred_letter = _extract_letter(prediction)
    if not pred_letter:
        # Model-side format failure (the no-answer-tag sentinel, or prose
        # without any A-F letter): the model failed to answer as requested.
        # Counts as WRONG, never UNSCORED — UNSCORED is reserved for
        # infrastructure/data failures only.
        return 0.0, {"status": "SCORED",
                     "score_method": ("model_no_answer" if is_model_no_answer(prediction)
                                      else "no_letter_extracted"),
                     "extracted": "",
                     "gt": gt,
                     "category": sample.get("category", "")}
    correct = 1.0 if pred_letter == gt else 0.0
    return correct, {
        "status": "SCORED",
        "score_method": "answer_letter_exact",
        "extracted": pred_letter,
        "gt": gt,
        "category": sample.get("category", ""),
    }


def aggregate(rows: list[dict]) -> dict:
    n_scored = [r for r in rows if r["correct"] is not None]
    n = len(n_scored)
    acc = (sum(r["correct"] for r in n_scored) / n) if n else 0.0

    breakdown: dict[str, dict] = {"Perception": {}, "Reasoning": {}}
    for r in n_scored:
        cat = str(r["meta"].get("category", ""))
        parts = cat.split("/")
        if len(parts) != 2 or parts[0] not in breakdown:
            continue
        task, sub = parts
        breakdown[task].setdefault(sub, []).append(r["correct"])
    sub_acc = {
        task: {sub: (sum(v) / len(v) if v else 0.0) for sub, v in subs.items()}
        for task, subs in breakdown.items()
    }
    return {
        "answer_accuracy": acc,
        "n": len(rows),
        "n_scored": n,
        "per_dimension": sub_acc,
        # Grounding metrics: computed only when tool traces + GT boxes exist.
        "official_demo_box_recall_mIoU": "NOT_APPLICABLE",
        "tool_crop_IoU": "NOT_APPLICABLE",
    }
