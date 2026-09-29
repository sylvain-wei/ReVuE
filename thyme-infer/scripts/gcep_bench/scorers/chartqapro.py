# Modified for anonymous review: release paths, configuration, and documentation.
"""ChartQAPro scorer. OFFICIAL_DETERMINISTIC.

Faithful port of upstream evaluate_predictions.py (pinned:
github.com/vis-nlp/ChartQAPro main), INCLUDING the known bug: the per-split
``always_use_exact_match`` (Fact Checking / Multi Choice) is computed but never
passed into ``relaxed_correctness_chartqapro``. The headline score therefore
replicates upstream exactly. A corrected variant is emitted separately as
``chartqapro_patched_exact`` (never mixed into the headline), with a per-sample
diff list.
"""

from __future__ import annotations

import ast
import re
from typing import Any, List, Optional

from ..common import extract_answer_tag, is_model_no_answer
from .anls_lite import anls_score

PROTOCOL_NAME = "chartqapro_upstream_relaxed_v1"
SCORING_LAYER = "OFFICIAL_DETERMINISTIC"
REQUIRES_JUDGE = False

MAX_RELATIVE_CHANGE = 0.05


# ---------------------------------------------------------------- upstream helpers
def fix_list_format(item: str) -> Any:
    if not isinstance(item, str):
        return item
    match = re.match(r"^\[(.*)\]$", item.strip())
    if not match:
        return item
    content = match.group(1)
    corrected = re.sub(r"(?<!['\w])(\w[^,]*?)(?!['\w])", r"'\1'", content)
    try:
        return ast.literal_eval(f"[{corrected}]")
    except (SyntaxError, ValueError):
        return item


def parse_to_list(text: str) -> Optional[List[str]]:
    if not isinstance(text, str):
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, list):
        return [str(x).strip(" '") for x in parsed]
    return None


def to_float(text: str) -> Optional[float]:
    try:
        return float(text.strip().strip('%'))
    except ValueError:
        return None


def evaluate_single_answer(target: str, prediction: str,
                           max_relative_change: float = MAX_RELATIVE_CHANGE) -> float:
    t = target.strip().strip('%').strip()
    p = prediction.strip().strip('%').strip()
    t_f, p_f = to_float(t), to_float(p)
    if t_f is not None and p_f is not None:
        if t_f == 0.0:
            return 1.0 if p_f == 0.0 else 0.0
        change = abs(p_f - t_f) / abs(t_f)
        return 1.0 if change <= max_relative_change else 0.0
    return anls_score(prediction=p.lower(), gold_labels=[t.lower()], threshold=0.5)


def relaxed_correctness_chartqapro(target: str, prediction: str,
                                   max_relative_change: float = MAX_RELATIVE_CHANGE,
                                   year_flags: Optional[List[bool]] = None,
                                   always_use_exact_match: bool = False) -> float:
    fixed_t = fix_list_format(target)
    t_list = parse_to_list(str(fixed_t)) or [str(target)]
    p_list = parse_to_list(str(prediction)) or [str(prediction)]
    n = len(t_list)
    if year_flags is not None and len(year_flags) < n:
        year_flags = year_flags * n

    scores: List[float] = []
    for idx in range(max(len(t_list), len(p_list))):
        if idx >= len(t_list) or idx >= len(p_list):
            scores.append(0.0)
            continue
        t_item, p_item, flag = t_list[idx], p_list[idx], year_flags[idx]
        flag_cond = True if str(flag).upper() == 'YES' else False
        if flag_cond or always_use_exact_match:
            try:
                scores.append(1.0 if t_item.strip().lower() == p_item.strip().lower() else 0.0)
            except ValueError:
                scores.append(0.0)
        else:
            scores.append(evaluate_single_answer(t_item, p_item, max_relative_change))
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------- adapter layer
def _prediction_view(prediction: str) -> str:
    """Upstream feeds the model's answer string; we strip <answer> tags first."""
    ans = extract_answer_tag(prediction)
    return ans if ans else (prediction or "").strip()


def score_sample(sample: dict, prediction: str, **ctx) -> tuple[Optional[float], dict]:
    gt = str(sample.get("answer", "")).strip().strip(".").strip("\n")
    qtype = str(sample.get("question_type", ""))
    if is_model_no_answer(prediction):
        # Model-side format failure -> 0 on both headline and patched tracks.
        return 0.0, {"status": "SCORED", "score_method": "model_no_answer",
                     "score_continuous": 0.0, "score_patched_exact": 0.0,
                     "patch_differs": False, "question_type": qtype,
                     "gt": gt, "extracted_prediction": ""}
    pred = _prediction_view(prediction).strip(".").strip("\n")
    year_flags = sample.get("year_flags", [])
    if isinstance(year_flags, str):
        year_flags = [f.strip() for f in year_flags.split("|") if f.strip()]
    if qtype == "Conversational":
        year_flags = list(year_flags)[-1:]

    # BUG REPLICATION: upstream computes always_use_exact_match for
    # ['Fact Checking', 'Multi Choice'] but never passes it. Headline = as-upstream.
    score_headline = relaxed_correctness_chartqapro(gt, pred, year_flags=year_flags)
    always_exact = qtype in ("Fact Checking", "Multi Choice")
    score_patched = relaxed_correctness_chartqapro(
        gt, pred, year_flags=year_flags, always_use_exact_match=always_exact)

    if pred == "":
        return None, {"status": "UNSCORED", "error": "empty prediction", "gt": gt}
    return score_headline, {
        "status": "SCORED",
        "score_method": "chartqapro_relaxed_upstream_bugreplica",
        "score_continuous": score_headline,  # relaxed correctness in [0,1]
        "score_patched_exact": score_patched,
        "patch_differs": score_patched != score_headline,
        "question_type": qtype,
        "gt": gt,
        "extracted_prediction": pred[:500],
    }


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["correct"] is not None]
    n = len(scored)
    per_split: dict[str, list[float]] = {}
    patched_all: list[float] = []
    diffs = []
    for r in scored:
        split = r["meta"].get("question_type", "") or "UNKNOWN"
        per_split.setdefault(split, []).append(r["correct"])
        patched_all.append(r["meta"]["score_patched_exact"])
        if r["meta"]["patch_differs"]:
            diffs.append(r["sample_id"])
    metrics = {
        "overall": (sum(r["correct"] for r in scored) / n) if n else 0.0,
        "n": len(rows),
        "n_scored": n,
        "per_question_type": {k: sum(v) / len(v) for k, v in per_split.items()},
        "chartqapro_patched_exact": {
            "overall": (sum(patched_all) / n) if n else 0.0,
            "n_diff_samples": len(diffs),
            "diff_sample_ids": diffs[:200],
        },
    }
    return metrics
