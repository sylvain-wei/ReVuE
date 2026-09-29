# Modified for anonymous review: release paths, configuration, and documentation.
"""ZoomBench scorer. OFFICIAL_DETERMINISTIC (+ tiered judge track).

Two tracks, reconciled on all 845 before any final table:

  release_deterministic_v1 (headline, no judge):
    mcq   -> letter A-D exact (extracted from <answer>/response)
    blank -> numeric normalization (GT int 1-18) + math equivalence fallback

  upstream_tiered (parity with pinned upstream judge_qwenlm.py):
    extraction: <answer>...</answer> -> 'Answer:' suffix -> last 3 lines
    mathruler.grade_answer -> unresolved -> official-prompt text-only 397B judge
    ('Yes'/'yes' substring counts as correct, matching upstream in-pipeline
    counting; cal_acc.py's exact 'Yes' mismatch is a known upstream bug)
    NOTE: 'zoom-bench' is NOT in the upstream mcq_benchmarks whitelist, so the
    upstream track applies NO letter matching — replicated here deliberately.

Global-View accuracy = this scorer on full images (the GCEP main track).
Regional-View accuracy requires the separate oracle-crop control run and is
NOT computed here.
"""

from __future__ import annotations

import re
from typing import Optional

from ..common import extract_answer_tag, is_model_no_answer, math_equal, normalize_number
from ..judge import GcepTextJudge

PROTOCOL_NAME = "zoombench_release_deterministic_v1"
SCORING_LAYER = "OFFICIAL_DETERMINISTIC"
REQUIRES_JUDGE = True  # only the upstream_tiered track calls the judge

# Verbatim from upstream judge_qwenlm.py (pinned fdc0ba1).
UPSTREAM_JUDGE_PROMPT = (
    "Your task is to judge whether the response expresses the same meaning as the "
    "answer of a question.\nThe question is: {question}\nThe answer is: {gt}\n"
    "The response is: {response}\nPlease check and compare them and then judge. "
    "If the response is correct, your output should be Yes. Otherwise, your output "
    "should be No. Directly give me your output."
)

_DATASET_REVISION = "b788097e57d30510c6877824833234a73bf80d25"

_FIRST_OPTION_PAREN_RE = re.compile(r"\(([A-Z])\)")
_FIRST_OPTION_LOOSE_RE = re.compile(r"([A-Z])[\.\)\s]")
_FIRST_OPTION_FALLBACK_RE = re.compile(r"([A-Z])")


def extract_first_option(text: str) -> str:
    """Upstream extract_first_option (used on the MODEL answer)."""
    if not text:
        return ""
    m = _FIRST_OPTION_PAREN_RE.search(text)
    if m:
        return m.group(1)
    m = _FIRST_OPTION_LOOSE_RE.search(text)
    if m:
        return m.group(1)
    m = _FIRST_OPTION_FALLBACK_RE.search(text)
    if m:
        return m.group(1)
    return ""


def upstream_extract_answer(model_answer_raw: str) -> str:
    """Upstream answer-extraction cascade (judge_qwenlm.py)."""
    raw = model_answer_raw or ""
    if "<answer>" in raw:
        seg = raw[raw.find("<answer>"):raw.find("</answer>")]
        return seg.replace("<answer>", "").replace("</answer>", "").strip()
    if "Answer:" in raw:
        return raw[raw.find("Answer:"):].strip()
    return "\n".join(raw.split("\n")[-3:]).strip()


def _deterministic_verdict(qtype: str, gt: str, prediction: str) -> tuple[Optional[bool], str]:
    """release_deterministic_v1. Returns (correct, method); None => unresolved."""
    ans = extract_answer_tag(prediction) or prediction or ""
    if qtype == "mcq":
        letter = extract_first_option(ans)
        if not letter:
            return None, "mcq_no_letter"
        return (letter == gt.strip().upper()), "mcq_letter_exact"
    # blank: GT is an integer 1-18
    p, g = normalize_number(ans), normalize_number(gt)
    if p is not None and g is not None:
        return (p == g), "blank_numeric_exact"
    eq = math_equal(ans, gt)
    if eq is not None:
        return eq, "blank_math_equal"
    return None, "blank_unresolved"


def score_sample(sample: dict, prediction: str, *, judge: Optional[GcepTextJudge] = None,
                 benchmark: str = "ZoomBench", **ctx) -> tuple[Optional[float], dict]:
    qtype = str(sample.get("question_type", "")).strip()
    gt = str(sample.get("answer", "")).strip()
    question = str(sample.get("question", ""))

    if is_model_no_answer(prediction):
        # Model-side format failure -> wrong on both tracks, no judge call.
        return 0.0, {
            "status": "SCORED",
            "question_type": qtype,
            "gt": gt,
            "release_deterministic_v1": {"correct": False, "final": False, "method": "model_no_answer"},
            "upstream_tiered": {"correct": False, "final": False, "source": "model_no_answer"},
            "tracks_agree": True,
            "score_method": "model_no_answer",
        }

    det_correct, det_method = _deterministic_verdict(qtype, gt, prediction)

    # ---------------- upstream_tiered track
    extracted = upstream_extract_answer(prediction)
    tier_correct: Optional[bool] = None
    tier_source = ""
    eq = math_equal(gt, extracted)
    if eq:
        tier_correct, tier_source = True, "mathruler"
    if tier_correct is None:
        if judge is None:
            tier_correct, tier_source = None, "judge_unavailable"
        else:
            prompt = UPSTREAM_JUDGE_PROMPT.format(
                question=question.replace("<image>", ""), gt=gt, response=extracted)
            outcome = judge.judge(
                prompt, benchmark_id=benchmark, dataset_revision=_DATASET_REVISION,
                sample_id=str(sample.get("index", "")), question=question, gold=gt,
                prediction=extracted, prompt_hash="zoombench_upstream_judge_v1",
            )
            if outcome.status == "SCORED":
                # upstream counts 'Yes'/'yes' substring; our strict parser already
                # yields 0/1, map 1->Yes-equivalent. Recorded as adapted parser.
                tier_correct, tier_source = bool(outcome.score), "llm"
                judge_meta = {"raw_text": outcome.raw_text[:300], "cache_hit": outcome.cache_hit,
                              "attempts": outcome.attempts}
            else:
                tier_source = "unscored"
                judge_meta = {"error": outcome.error, "attempts": outcome.attempts}
    if tier_source not in ("llm", "unscored"):
        judge_meta = {}

    # headline = deterministic track; if deterministic unresolved, fall back to
    # the tiered verdict (recorded), else UNSCORED.
    if det_correct is not None:
        final, status = det_correct, "SCORED"
    elif tier_correct is not None:
        final, status = tier_correct, "SCORED"
        det_method = f"{det_method}->tiered:{tier_source}"
    else:
        final, status = None, "UNSCORED"

    # Per-track FINAL verdict (each falls back to the other track's verdict when
    # itself is unresolved) so parity covers ALL samples, not just the
    # mathruler-resolved subset (which is trivially correct by construction).
    release_final = det_correct if det_correct is not None else tier_correct
    tier_final = tier_correct if tier_correct is not None else det_correct
    meta = {
        "status": status,
        "question_type": qtype,
        "gt": gt,
        "release_deterministic_v1": {
            "correct": det_correct, "final": release_final, "method": det_method},
        "upstream_tiered": {"correct": tier_correct, "final": tier_final,
                            "source": tier_source, **judge_meta},
        "tracks_agree": (release_final is None or tier_final is None
                         or release_final == tier_final),
        "score_method": det_method,
    }
    if final is None:
        return None, meta
    return (1.0 if final else 0.0), meta


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["correct"] is not None]
    n = len(scored)
    acc = (sum(r["correct"] for r in scored) / n) if n else 0.0

    per_type: dict[str, list[float]] = {}
    det_right = tier_right = disagree = n_judge = 0
    diff_ids = []
    for r in scored:
        qt = r["meta"].get("question_type", "")
        per_type.setdefault(qt, []).append(r["correct"])
        d = r["meta"]["release_deterministic_v1"]["final"]
        t = r["meta"]["upstream_tiered"]["final"]
        if d is not None:
            det_right += bool(d)
        if t is not None:
            tier_right += bool(t)
        if d is not None and t is not None and d != t:
            disagree += 1
            diff_ids.append(r["sample_id"])
        if r["meta"].get("upstream_tiered", {}).get("source") == "llm":
            n_judge += 1
    return {
        "global_view_accuracy": acc,
        "regional_view_accuracy": "NOT_COMPUTED_HERE (oracle-crop control run only)",
        "n": len(rows),
        "n_scored": n,
        "per_question_type": {k: sum(v) / len(v) for k, v in per_type.items()},
        "track_parity": {
            "n_all_scored": n,
            "release_deterministic_v1_acc": (det_right / n) if n else 0.0,
            "upstream_tiered_acc": (tier_right / n) if n else 0.0,
            "n_judge_decided": n_judge,
            "n_disagree": disagree,
            "diff_sample_ids": diff_ids[:200],
        },
    }
