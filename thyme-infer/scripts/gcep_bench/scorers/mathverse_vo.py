# Modified for anonymous review: release paths, configuration, and documentation.
"""MathVerse Vision-Only scorer. ADAPTED_LLM_JUDGE.

Official w/o CoT-E protocol (ZrrSkywalker/MathVerse @937b0905,
evaluation/score_answer_s2.py + prompts.py): extract the final answer from the
rollout, then an LLM judges semantic equivalence vs the GT with the verbatim
``demo_prompt_score`` few-shot template, output 0/1.

Deliberate deviations (recorded as adapted protocol):
- extraction: our uniform answer-extraction protocol (<answer> tag, then full
  response tail) instead of upstream's GPT-4 extraction stage s1 — the upstream s1
  LLM-extractor is unnecessary because our models emit explicit <answer> tags;
- judge model: qwen3.5-397b (text-only, temperature=0, no_thinking, max_tokens=512)
  instead of upstream GPT-4;
- {question} slot := question_for_eval (for Vision Only rows the ``question`` field
  is EMPTY — the problem statement lives only in the image; question_for_eval is
  its textual form and gives the judge the problem context). The model itself never
  sees question_for_eval (model input = image + query_wo only);
- strict 0/1 parser + bounded retries -> UNSCORED instead of upstream's unbounded
  retry loop (score_answer_s2.py:90-96);
- upstream `category` vs `metadata` field inconsistency: we use metadata.subject /
  metadata.subfield captured at build time.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from ..common import extract_answer_tag, extract_model_answer, is_model_no_answer
from ..judge import GcepTextJudge

PROTOCOL_NAME = "mathverse_vo_wo_cote_qwen397b_t0"
SCORING_LAYER = "ADAPTED_LLM_JUDGE"
REQUIRES_JUDGE = True
# Judge is mandatory for every sample: in-loop progress defers to post-hoc.
DEFER_WITHOUT_JUDGE = True

_ASSET = Path(__file__).resolve().parent.parent / "assets" / "mathverse_demo_prompt_score.txt"
DEMO_PROMPT_SCORE = _ASSET.read_text(encoding="utf-8")

DATASET_REVISION = "3bc86196678bad115a923d2851c6821dbe235939"

_JUDGEMENT_RE = re.compile(r"^\s*(?:Judgement:)?\s*([01])\s*$", re.IGNORECASE)


def build_judge_prompt(question_for_eval: str, gt: str, extraction: str) -> str:
    # Upstream create_test_prompt: demo_prompt.strip().format(question=, gt=, extraction=)
    return DEMO_PROMPT_SCORE.strip().format(
        question=question_for_eval, gt=gt, extraction=extraction)


def verdict_parser(raw_text: str) -> Optional[str]:
    m = _JUDGEMENT_RE.match(raw_text.strip())
    return m.group(1) if m else None


def prompt_hash() -> str:
    return hashlib.sha256(DEMO_PROMPT_SCORE.encode("utf-8")).hexdigest()[:16]


def score_sample(sample: dict, prediction: str, *, judge: Optional[GcepTextJudge] = None,
                 benchmark: str = "MathVerseVO", **ctx) -> tuple[Optional[float], dict]:
    if judge is None:
        raise ValueError("mathverse_vo scorer requires judge=GcepTextJudge")
    gt = str(sample.get("answer", "")).strip()
    q_eval = str(sample.get("question_for_eval", ""))
    qtype = str(sample.get("question_type", ""))
    base_meta = {"gt": gt, "question_type": qtype,
                 "subject": str(sample.get("subject", "")),
                 "subfield": str(sample.get("subfield", ""))}
    if is_model_no_answer(prediction):
        return 0.0, {**base_meta, "status": "SCORED", "score_method": "model_no_answer",
                     "extracted_prediction": ""}
    extraction = extract_answer_tag(prediction) or extract_model_answer(prediction)
    prompt = build_judge_prompt(q_eval, gt, extraction)
    outcome = judge.judge(
        prompt,
        benchmark_id=benchmark,
        dataset_revision=DATASET_REVISION,
        sample_id=str(sample.get("index", "")),
        question=q_eval, gold=gt, prediction=extraction,
        prompt_hash=prompt_hash(),
        verdict_parser=verdict_parser,
        verdict_schema="mv_judgement01",
    )
    meta = {**base_meta,
            "status": outcome.status,
            "score_method": "qwen397b_demo_prompt_score_strict01",
            "extracted_prediction": extraction[:500],
            "judge_raw": {
                "raw_text": outcome.raw_text[:500],
                "attempts": outcome.attempts,
                "cache_hit": outcome.cache_hit,
                "returned_model": outcome.returned_model,
                "error": outcome.error,
            }}
    if outcome.status != "SCORED" or outcome.score is None:
        return None, meta
    return float(outcome.score), meta


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["correct"] is not None]
    n = len(scored)
    acc = (sum(r["correct"] for r in scored) / n) if n else 0.0

    def _by(key):
        groups: dict[str, list[float]] = {}
        for r in scored:
            groups.setdefault(str(r["meta"].get(key, "")), []).append(r["correct"])
        return {k: (sum(v) / len(v)) for k, v in sorted(groups.items()) if v}

    per_qtype = _by("question_type")
    return {
        "accuracy": acc,
        "n": len(rows),
        "n_scored": n,
        "multi_choice_accuracy": per_qtype.get("multi-choice", 0.0),
        "free_form_accuracy": per_qtype.get("free-form", 0.0),
        "per_subject": _by("subject"),
        "per_subfield": _by("subfield"),
        "judge_failure_rate": (len(rows) - n) / len(rows) if rows else 0.0,
    }
