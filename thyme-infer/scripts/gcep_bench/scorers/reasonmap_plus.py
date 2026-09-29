# Modified for anonymous review: release paths, configuration, and documentation.
"""ReasonMap-Plus scorer. OFFICIAL_DETERMINISTIC.

Port of upstream main_plus.py per-sample logic + cal_metrics.py
analyze_results_v2_weighted aggregation:
  - Counting1: model letter a-d -> 0-3, exact vs int GT;
  - Counting2/3: int exact;
  - TorF1/2: GT 1->"yes"/0->"no", lowercase exact;
  - difficulty weights easy/middle/hard = 1/1.5/2; weighted acc = sum(a*w)/sum(w).
Answer extraction mirrors upstream: boxed -> "the answer is"-style flags ->
rstrip('.'). Upstream uses utils.find_math_answer (boxed extraction); we use the
last \\boxed{...} content, recorded as a documented equivalence.
"""

from __future__ import annotations

from typing import Optional

from ..common import extract_answer_tag, extract_boxed

PROTOCOL_NAME = "reasonmap_plus_official_deterministic_v1"
SCORING_LAYER = "OFFICIAL_DETERMINISTIC"
REQUIRES_JUDGE = False

DIFFICULTY_WEIGHTS = {"easy": 1.0, "middle": 1.5, "hard": 2.0}
_ANSWER_FLAGS = ["the final answer is", "the answer is",
                 "the correct answer is", "the answer should be"]


def extract_model_answer_reasonmap(response: str) -> str:
    """Upstream extraction (main_plus.py): boxed handling + flag splitting."""
    model_answer = response or ""
    # Thyme models wrap the final answer in <answer> tags; unwrap first so the
    # upstream boxed/flag logic sees the same string it would see upstream.
    tagged = extract_answer_tag(model_answer)
    if tagged:
        model_answer = tagged
    if "oxed{" not in model_answer:
        for flag in _ANSWER_FLAGS:
            raw = model_answer
            model_answer = model_answer.split(flag)[-1].strip()
            if flag in raw:
                model_answer = model_answer.split("\n")[0].split(". ")[0]
            cap = flag.replace("the", "The")
            raw = model_answer
            model_answer = model_answer.split(cap)[-1].strip()
            if cap in raw:
                model_answer = model_answer.split("\n")[0].split(". ")[0]
    elif model_answer.count("oxed{") > 1:
        model_answer = "\\boxed{" + model_answer.split("oxed{")[-1]
    boxed = extract_boxed(model_answer)
    if boxed:
        model_answer = boxed
    return model_answer.rstrip(".").lstrip(":").strip()


def score_sample(sample: dict, prediction: str, **ctx) -> tuple[Optional[float], dict]:
    qtype = str(sample.get("type", ""))
    gt_raw = str(sample.get("answer", "")).strip()
    model_answer = extract_model_answer_reasonmap(prediction)

    correct: Optional[bool]
    detail = ""
    try:
        gt_int = int(gt_raw)
    except ValueError:
        return None, {"status": "UNSCORED", "error": f"non-int GT {gt_raw!r}"}

    if "TorF" in qtype:
        gt = "yes" if gt_int == 1 else "no"
        correct = model_answer.lower() == gt
        detail = f"torf gt={gt}"
    elif qtype in ("Counting2", "Counting3"):
        try:
            correct = int(model_answer) == gt_int
        except (ValueError, TypeError):
            correct = False
        detail = f"int gt={gt_int}"
    elif qtype == "Counting1":
        ma = model_answer.lower()
        if ma in ("a", "b", "c", "d"):
            correct = {"a": 0, "b": 1, "c": 2, "d": 3}[ma] == gt_int
        else:
            correct = False
        detail = f"letter->idx gt={gt_int}"
    else:
        return None, {"status": "UNSCORED", "error": f"unknown type {qtype!r}"}

    difficulty = str(sample.get("difficulty_city", "easy")).lower()
    weight = DIFFICULTY_WEIGHTS.get(difficulty, 1.0)
    return (1.0 if correct else 0.0), {
        "status": "SCORED",
        "score_method": "reasonmap_plus_exact",
        "type": qtype,
        "difficulty": difficulty,
        "weight": weight,
        "extracted": model_answer[:200],
        "detail": detail,
    }


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["correct"] is not None]
    n = len(scored)
    acc = (sum(r["correct"] for r in scored) / n) if n else 0.0
    w_total = sum(r["meta"]["weight"] for r in scored)
    w_acc = (sum(r["correct"] * r["meta"]["weight"] for r in scored) / w_total) if w_total else 0.0

    per_type: dict[str, list[float]] = {}
    per_diff: dict[str, list[float]] = {}
    for r in scored:
        per_type.setdefault(r["meta"]["type"], []).append(r["correct"])
        per_diff.setdefault(r["meta"]["difficulty"], []).append(r["correct"])
    return {
        "acc": acc,
        "weighted_acc": w_acc,
        "n": len(rows),
        "n_scored": n,
        "per_type": {k: sum(v) / len(v) for k, v in per_type.items()},
        "per_difficulty": {k: sum(v) / len(v) for k, v in per_diff.items()},
    }
