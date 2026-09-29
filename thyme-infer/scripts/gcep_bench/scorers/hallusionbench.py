# Modified for anonymous review: release paths, configuration, and documentation.
"""HallusionBench scorer. ADAPTED_LLM_JUDGE.

Official protocol (tianyi-lab/HallusionBench @744007c2, utils.py):
- GPT judge prompt (utils.py:33-40) over question + gt_answer_details + prediction,
  verdict in {correct, incorrect, unclear};
- correctness (utils.py:404-411 assign_correctness): correct = (verdict==correct)
  OR (verdict==unclear AND category=="VS" AND figure_id=="0") — the ONLY condition
  where unclear counts as correct;
- metrics: aAcc per-question (1129), fAcc per-figure (groups keyed
  category/subcategory/set_id/figure_id, VS figure_id==0 excluded -> 346, all rows
  in group correct), qAcc per question-control-group (keyed
  category/subcategory/set_id/question_id -> 455, all rows correct),
  overall = (aAcc + fAcc + qAcc) / 3.

Deliberate deviations (recorded as adapted protocol):
- judge model: qwen3.5-397b (text-only, temperature=0, no_thinking, max_tokens=512),
  judge_repeats=1 single deterministic verdict;
  upstream used GPT-4 with paper protocol of three judgments; we do NOT claim
  paper-equivalence, and never majority-vote;
- verdict parser is word-boundary based and checks ``incorrect`` BEFORE ``correct``
  (substring 'correct' also matches 'incorrect'); upstream substring semantics are
  preserved but unparseable output -> bounded retry -> UNSCORED instead of
  silently mapping to unclear;
- the optional LH/VI/Mix diagnostic (check_same_by_chatgpt, utils.py:75-128) is NOT
  implemented: it needs a second judge prompt and is not part of aAcc/fAcc/qAcc.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from ..common import extract_answer_tag, is_model_no_answer
from ..judge import GcepTextJudge

PROTOCOL_NAME = "hallusionbench_official_prompt_qwen397b_t0_repeats1"
SCORING_LAYER = "ADAPTED_LLM_JUDGE"
REQUIRES_JUDGE = True
# Judge is mandatory for every sample: in-loop progress defers to post-hoc.
DEFER_WITHOUT_JUDGE = True
JUDGE_REPEATS = 1  # single deterministic verdict

_ASSET = Path(__file__).resolve().parent.parent / "assets" / "hallusionbench_judge_prompt.txt"
JUDGE_TEMPLATE = _ASSET.read_text(encoding="utf-8")

DATASET_REVISION = "744007c232c292942c7f80eb61edb2465482da31"

_INCORRECT_RE = re.compile(r"\bincorrect\b", re.IGNORECASE)
_CORRECT_RE = re.compile(r"\bcorrect\b", re.IGNORECASE)
_UNCLEAR_RE = re.compile(r"\bunclear\b", re.IGNORECASE)


def build_judge_prompt(question: str, gt_answer_details: str, prediction: str) -> str:
    return (JUDGE_TEMPLATE
            .replace("{question}", question)
            .replace("{gt_answer_details}", gt_answer_details)
            .replace("{prediction}", prediction))


def verdict_parser(raw_text: str) -> Optional[str]:
    """Official enum semantics: incorrect checked BEFORE correct (substring trap)."""
    t = raw_text.strip().lower()
    if _INCORRECT_RE.search(t):
        return "incorrect"
    if _CORRECT_RE.search(t):
        return "correct"
    if _UNCLEAR_RE.search(t):
        return "unclear"
    return None


def prompt_hash() -> str:
    return hashlib.sha256(JUDGE_TEMPLATE.encode("utf-8")).hexdigest()[:16]


def _unclear_counts_correct(category: str, figure_id: str) -> bool:
    # utils.py:407 — the only official condition where unclear counts as correct.
    return category == "VS" and str(figure_id) == "0"


def score_sample(sample: dict, prediction: str, *, judge: Optional[GcepTextJudge] = None,
                 benchmark: str = "HallusionBench_GCEP", **ctx) -> tuple[Optional[float], dict]:
    if judge is None:
        raise ValueError("hallusionbench scorer requires judge=GcepTextJudge")
    question = str(sample.get("question", ""))
    details = str(sample.get("gt_answer_details", ""))
    cat = str(sample.get("category", ""))
    sub = str(sample.get("subcategory", ""))
    set_id = str(sample.get("set_id", ""))
    figure_id = str(sample.get("figure_id", ""))
    question_id = str(sample.get("question_id", ""))
    vi = str(sample.get("visual_input", ""))
    base_meta = {"category": cat, "subcategory": sub, "set_id": set_id,
                 "figure_id": figure_id, "question_id": question_id,
                 "visual_input": vi, "gt_answer": str(sample.get("answer", ""))}
    if is_model_no_answer(prediction):
        return 0.0, {**base_meta, "status": "SCORED", "score_method": "model_no_answer",
                     "verdict": "incorrect", "extracted_prediction": ""}
    pred_answer = extract_answer_tag(prediction) or (prediction or "")
    prompt = build_judge_prompt(question, details, pred_answer)
    outcome = judge.judge(
        prompt,
        benchmark_id=benchmark,
        dataset_revision=DATASET_REVISION,
        sample_id=str(sample.get("index", "")),
        question=question, gold=details, prediction=pred_answer,
        prompt_hash=prompt_hash(),
        verdict_parser=verdict_parser,
        verdict_schema="hb_correct_incorrect_unclear",
    )
    meta = {**base_meta,
            "status": outcome.status,
            "score_method": "qwen397b_official_hb_prompt_repeats1",
            "extracted_prediction": pred_answer[:500],
            "verdict": outcome.verdict,
            "judge_raw": {
                "raw_text": outcome.raw_text[:500],
                "attempts": outcome.attempts,
                "cache_hit": outcome.cache_hit,
                "returned_model": outcome.returned_model,
                "error": outcome.error,
            }}
    if outcome.status != "SCORED" or outcome.verdict is None:
        return None, meta
    v = outcome.verdict
    correct = (v == "correct") or (v == "unclear" and _unclear_counts_correct(cat, figure_id))
    return (1.0 if correct else 0.0), meta


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["correct"] is not None]
    n = len(scored)
    a_acc = (sum(r["correct"] for r in scored) / n) if n else 0.0

    def _group_acc(key_fn, filt=None):
        groups: dict[tuple, list[float]] = {}
        for r in scored:
            m = r["meta"]
            if filt and not filt(m):
                continue
            groups.setdefault(key_fn(m), []).append(r["correct"])
        allc = sum(1 for v in groups.values() if sum(v) == len(v))
        return (allc / len(groups)) if groups else 0.0, len(groups)

    f_acc, n_fig = _group_acc(
        lambda m: (m["category"], m["subcategory"], m["set_id"], m["figure_id"]),
        filt=lambda m: not (m["category"] == "VS" and m["figure_id"] == "0"))
    q_acc, n_q = _group_acc(
        lambda m: (m["category"], m["subcategory"], m["set_id"], m["question_id"]))

    def _a(filt):
        v = [r["correct"] for r in scored if filt(r["meta"])]
        return (sum(v) / len(v)) if v else 0.0

    overall = (a_acc + f_acc + q_acc) / 3.0
    return {
        "aAcc": a_acc,
        "fAcc": f_acc,
        "qAcc": q_acc,
        "overall": overall,
        "n": len(rows),
        "n_scored": n,
        "n_figure_groups": n_fig,
        "n_question_groups": n_q,
        "aAcc_VD": _a(lambda m: m["category"] == "VD"),
        "aAcc_VS": _a(lambda m: m["category"] == "VS"),
        "aAcc_easy": _a(lambda m: m["visual_input"] != "2"),   # Easy = vi != 2
        "aAcc_hard": _a(lambda m: m["visual_input"] == "2"),   # Hard = vi == 2
        "verdict_counts": {
            v: sum(1 for r in scored if r["meta"].get("verdict") == v)
            for v in ("correct", "incorrect", "unclear")
        },
    }
