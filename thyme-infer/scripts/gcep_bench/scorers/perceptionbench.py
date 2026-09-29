# Modified for anonymous review: release paths, configuration, and documentation.
"""PerceptionBench scorer. ADAPTED_LLM_JUDGE.

Official protocol (MoonshotAI/PerceptionBench @ba032c06, eval/eval.py +
eval/judge_prompt.txt): judge gets question (raw problem text with <|image_N|>
placeholders, text-only), the student's full answer, and the reference answer;
verdict regex ``\\[judge\\]\\s*(True|False)\\s*$``.

Deliberate deviations (recorded as adapted protocol):
- judge model: qwen3.5-397b (text-only, temperature=0, no_thinking, max_tokens=512)
  instead of upstream gpt-oss-120b @ temperature=0.3;
- parse failure or judge API failure -> JUDGE_INFRA_ERROR (UNSCORED), NEVER scored
  0 (upstream decode_judge defaults False on missing [reason]/[judge] markers);
- bounded retries (upstream chat_with_retry defaults MAX_RETRIES=3; we keep our
  standard transport x5 / parse x2, fail-closed).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from ..common import is_model_no_answer
from ..judge import GcepTextJudge

PROTOCOL_NAME = "perceptionbench_official_prompt_qwen397b_t0"
SCORING_LAYER = "ADAPTED_LLM_JUDGE"
REQUIRES_JUDGE = True
# Judge is mandatory for every sample: in-loop progress defers to post-hoc.
DEFER_WITHOUT_JUDGE = True

_ASSET = Path(__file__).resolve().parent.parent / "assets" / "perceptionbench_judge_prompt.txt"
JUDGE_TEMPLATE = _ASSET.read_text(encoding="utf-8")
# sha256 of upstream third_party/PerceptionBench/eval/judge_prompt.txt @ba032c06
JUDGE_TEMPLATE_SHA256 = "3736ab84d90d96a0344a25bf96a1d9ac2a0672656bbc1eeb01176f20cd3d6397"

VERDICT_RE = re.compile(r"\[judge\]\s*(True|False)\s*$")
DATASET_REVISION = "6ba8c3135c7675ad6a5c141536a86b9460c70960"

# This scorer uses a 1536-token judge budget to accommodate the reasoning
# requested by the upstream prompt. Other scorer budgets remain unchanged.
JUDGE_MAX_TOKENS = 1536


def normalize_escape(s: str) -> str:
    """Verbatim port of upstream eval.py:65-69."""
    wl = " 0123456789-"
    s = re.sub(rf"\\+[{wl}]", r"\\\\" + " ", s)
    s = re.sub(rf"\\+(?![{wl}])", r"\\", s)
    return re.sub(r"\\+n", "\n", s)


def build_judge_prompt(question: str, prediction: str, reference: str) -> str:
    return (JUDGE_TEMPLATE
            .replace("{problem}", normalize_escape(str(question)).strip())
            .replace("{reference_answer}", normalize_escape(str(reference)).strip())
            .replace("{assistant_answer}", normalize_escape(str(prediction)).strip()))


def verdict_parser(raw_text: str) -> Optional[str]:
    m = VERDICT_RE.search(raw_text.strip())
    if not m:
        return None
    return "1" if m.group(1) == "True" else "0"


def prompt_hash() -> str:
    return hashlib.sha256(JUDGE_TEMPLATE.encode("utf-8")).hexdigest()[:16]


def score_sample(sample: dict, prediction: str, *, judge: Optional[GcepTextJudge] = None,
                 benchmark: str = "PerceptionBench", **ctx) -> tuple[Optional[float], dict]:
    if judge is None:
        raise ValueError("perceptionbench scorer requires judge=GcepTextJudge")
    question = str(sample.get("question", ""))
    gt = str(sample.get("answer", ""))
    cat = str(sample.get("error_category", ""))
    n_images = int(sample.get("n_images", 1) or 1)
    source_bmk = str(sample.get("source_bmk", ""))
    if is_model_no_answer(prediction):
        return 0.0, {"status": "SCORED", "score_method": "model_no_answer",
                     "extracted_prediction": "", "gt": gt[:500],
                     "category": cat, "n_images": n_images, "source_bmk": source_bmk}
    # Upstream passes the model's full response as the student answer.
    prompt = build_judge_prompt(question, prediction, gt)
    outcome = judge.judge(
        prompt,
        benchmark_id=benchmark,
        dataset_revision=DATASET_REVISION,
        sample_id=str(sample.get("index", "")),
        question=question, gold=gt, prediction=prediction,
        prompt_hash=prompt_hash(),
        verdict_parser=verdict_parser,
        verdict_schema="pb_judge_truefalse",
        max_tokens=JUDGE_MAX_TOKENS,
    )
    meta = {
        "status": outcome.status,
        "score_method": "qwen397b_official_pb_prompt_strict_judge_regex_mt1536",
        "gt": gt[:500],
        "category": cat,
        "n_images": n_images,
        "source_bmk": source_bmk,
        "judge_raw": {
            "raw_text": outcome.raw_text[:500],
            "attempts": outcome.attempts,
            "cache_hit": outcome.cache_hit,
            "returned_model": outcome.returned_model,
            "error": outcome.error,
            "verdict": outcome.verdict,
            "judge_max_tokens": JUDGE_MAX_TOKENS,
        },
    }
    if outcome.status != "SCORED" or outcome.score is None:
        return None, meta
    return float(outcome.score), meta


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["correct"] is not None]
    n = len(scored)
    acc = (sum(r["correct"] for r in scored) / n) if n else 0.0
    per_cat: dict[str, list[float]] = {}
    single: list[float] = []
    multi: list[float] = []
    src_na: list[float] = []
    src_derived: list[float] = []
    for r in scored:
        m = r["meta"]
        per_cat.setdefault(str(m.get("category", "")), []).append(r["correct"])
        (multi if int(m.get("n_images", 1) or 1) >= 2 else single).append(r["correct"])
        (src_na if m.get("source_bmk") == "NA" else src_derived).append(r["correct"])
    _mean = lambda v: (sum(v) / len(v)) if v else 0.0
    return {
        "accuracy": acc,
        "n": len(rows),
        "n_scored": n,
        "per_category": {k: _mean(v) for k, v in sorted(per_cat.items())},
        "single_image_accuracy": _mean(single),
        "multi_image_accuracy": _mean(multi),
        "n_single_image": len(single),
        "n_multi_image": len(multi),
        # source_bmk=NA is NOT an official split: custom diagnostic only.
        "diag_source_na_accuracy": _mean(src_na),
        "diag_source_derived_accuracy": _mean(src_derived),
    }
