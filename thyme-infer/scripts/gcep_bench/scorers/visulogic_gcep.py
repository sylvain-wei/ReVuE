# Modified for anonymous review: release paths, configuration, and documentation.
"""VisuLogic_GCEP scorer. OFFICIAL_DETERMINISTIC + Qwen fallback extractor.

Intended protocol (VisuLogic-Benchmark/VisuLogic-Eval @00fba6dd,
evaluation/eval_model.py extract_answer_v1):
1. last valid ``\\boxed{A|B|C|D}`` (extract_last_boxed_content) -> "box";
2. language patterns (extract_lang_content: ' X.' / '(X).' / 'answer:' /
   'the final answer is' ...) -> "lang";
3. fixed LLM extractor (extract_gpt prompt) -> "llm";
4. exact match vs ``label``.

Deliberate fixes vs upstream (recorded; upstream evaluator is NOT run as-is):
- upstream extract_gpt retries FOREVER (eval_model.py:106-120 while True); we use
  bounded transport x5 / parse x2, fail-closed;
- upstream judge client is mis-wired (eval_model.py:24 self.judege_api_key typo) and
  its label/answer field usage is inconsistent across versions; our scorer reads
  the pinned ``label`` column only;
- LLM extractor returning None/'None' => extraction outcome 'N' => score 0 and
  counted as parse failure (model produced no extractable option); LLM output that
  is not A-D/None after bounded retries => JUDGE_INFRA_ERROR (UNSCORED), never
  silently 0;
- Qwen here is an answer EXTRACTOR, not a correctness judge.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from ..common import is_model_no_answer
from ..judge import GcepTextJudge

PROTOCOL_NAME = "visulogic_official_cascade_qwen_extractor"
SCORING_LAYER = "OFFICIAL_DETERMINISTIC"  # LLM only as bounded fallback extractor
REQUIRES_JUDGE = True  # fallback extractor needs the endpoint (healthcheck required)

DATASET_REVISION = "3e483f8cca0bc766ecc5f70299f2b8a34b5035ca"

OPTION_LIST = ["A", "B", "C", "D"]

# Verbatim upstream extractor messages (eval_model.py:111-112).
EXTRACTOR_SYSTEM = "You are a helpful and precise assistant for extract the final option in the answer."
EXTRACTOR_USER = ("Extract the option in the solution. Just answer only capital letters. "
                  "If solution have multi options or has no clear option just answer None.\n"
                  "Solution: {ans}\n")

_ANSWER_LETTER_RE = re.compile(r"^\s*([ABCD])\s*$")
# Bare capital letter(s), optionally joined ("E", "A and D", "B, C"): the model's
# answer contains no clear VALID A-D option. The upstream extractor treats such
# outputs verbatim (exact-match fails -> 0); we map them to the contract's 'N'
# ("no clear option") with identical final scoring (0) but honest parse stats.
_NEAR_OPTION_RE = re.compile(r"^\s*[A-Z]\s*((and|or|,|&)\s*[A-Z]\s*)*$", re.IGNORECASE)


# ---------------------------------------------------------------- upstream ports
def extract_last_boxed_content(text: str) -> str:
    """Verbatim port of upstream eval_model.py:36-68."""
    stack = []
    last_boxed_content = None
    text = str(text)
    if len(text) < 3:
        return text
    pattern = re.finditer(r'\\boxed\{|[^\\]\}', text)
    try:
        for match in pattern:
            if match.group().endswith(r'\boxed{'):
                stack.append(match.end())
            elif match.group().endswith('}') and stack:
                start = stack.pop()
                if not stack:
                    last_boxed_content = text[start:match.start() + 1]
        if last_boxed_content:
            latex_commands = [r'\text{', r'\rm{', r'\mathbf{', '$']
            for cmd in latex_commands:
                last_boxed_content = last_boxed_content.replace(cmd, '')
            last_boxed_content = last_boxed_content.replace('}', '')
            if ("LETTER".lower() in last_boxed_content.lower() or
                    'or' in last_boxed_content or
                    len(last_boxed_content) > 2):
                last_boxed_content = text
    except Exception:
        last_boxed_content = text
    return 'N' if last_boxed_content is None else last_boxed_content


def extract_lang_content(ans: str) -> str:
    """Verbatim port of upstream eval_model.py:84-103."""
    ans = str(ans)
    ans = ans.replace("<|endoftext|>", "")
    for c in OPTION_LIST:
        if (ans.endswith(f" {c}.") or ans.endswith(f" ({c}).") or ans.startswith(f"{c}\n")
                or ans.startswith(f"({c})\n") or ans.startswith(f"({c}) {c}\n")):
            return c
    lower_ans = ans.lower()
    for flag in ["answer:", 'the final answer is:', 'the answer is option:', 'the answer is:',
                 'the correct answer is option:', 'the correct answer is:', 'the answer should be:',
                 'the final answer is', 'the answer is option', 'the answer is',
                 'the correct answer is option', 'the correct answer is', 'the answer should be']:
        if flag in lower_ans:
            lower_ans = lower_ans.split(flag)[-1].strip()
            lower_ans = lower_ans.split('\n')[0].split('.')[0]
            upper_ans = lower_ans.upper()
            if upper_ans in OPTION_LIST:
                return upper_ans
    return ans


def _llm_verdict_parser(raw_text: str) -> Optional[str]:
    t = raw_text.strip()
    if not t:
        return None
    t = t.split('\n')[0].split('.')[0].strip()
    if not t:
        return None
    if "none" in t.lower():
        return "N"
    m = _ANSWER_LETTER_RE.match(t)
    if m:
        return m.group(1)
    if _NEAR_OPTION_RE.match(t):
        # Out-of-range letter ("E") or multi-letter ("A and D"): no valid single
        # A-D option -> 'N' (score 0 model parse failure, NOT infra). Only
        # contract-violating prose remains a judge-infra parse failure.
        return "N"
    return None


def prompt_hash() -> str:
    return hashlib.sha256((EXTRACTOR_SYSTEM + "\n" + EXTRACTOR_USER).encode("utf-8")).hexdigest()[:16]


def score_sample(sample: dict, prediction: str, *, judge=None,
                 benchmark: str = "VisuLogic_GCEP", **ctx) -> tuple[Optional[float], dict]:
    gt = str(sample.get("answer", sample.get("label", ""))).strip().upper()
    cat = str(sample.get("category", ""))
    if is_model_no_answer(prediction):
        return 0.0, {"status": "SCORED", "score_method": "model_no_answer",
                     "extracted": "", "gt": gt, "category": cat, "parse_stage": "model_no_answer"}

    text = prediction or ""
    # stage 1: boxed
    boxed = extract_last_boxed_content(text).strip()
    if boxed in OPTION_LIST:
        return _finish(boxed, "box", gt, cat)
    # stage 2: language patterns
    lang = extract_lang_content(text)
    if lang in OPTION_LIST:
        return _finish(lang, "lang", gt, cat)
    # stage 3: bounded Qwen fallback extractor (NOT a correctness judge)
    if judge is None:
        # judge-free contexts (in-loop progress): deterministic cascade exhausted,
        # defer the rest to post-hoc.
        return None, {"status": "UNSCORED", "score_method": "gcep_deferred_to_posthoc",
                      "extracted": "", "gt": gt, "category": cat,
                      "parse_stage": "deferred_llm_fallback"}
    prompt = EXTRACTOR_SYSTEM + "\n\n" + EXTRACTOR_USER.format(ans=text[-4000:])
    outcome = judge.judge(
        prompt,
        benchmark_id=benchmark,
        dataset_revision=DATASET_REVISION,
        sample_id=str(sample.get("index", "")),
        question="", gold=gt, prediction=text[-1000:],
        prompt_hash=prompt_hash(),
        verdict_parser=_llm_verdict_parser,
        verdict_schema="vl_extract_abcd_none",
    )
    if outcome.status != "SCORED" or outcome.verdict is None:
        return None, {"status": "UNSCORED",
                      "score_method": "judge_infra_error",
                      "extracted": "", "gt": gt, "category": cat,
                      "parse_stage": "llm_infra",
                      "judge_raw": {"raw_text": outcome.raw_text[:500],
                                    "attempts": outcome.attempts,
                                    "cache_hit": outcome.cache_hit,
                                    "returned_model": outcome.returned_model,
                                    "error": outcome.error}}
    ext = outcome.verdict  # "A".."D" or "N"
    stage = "llm" if ext in OPTION_LIST else "llm_none"
    return _finish(ext, stage, gt, cat,
                   judge_raw={"attempts": outcome.attempts, "cache_hit": outcome.cache_hit})


def _finish(extracted: str, stage: str, gt: str, cat: str, judge_raw=None) -> tuple[float, dict]:
    correct = 1.0 if extracted == gt else 0.0
    meta = {"status": "SCORED",
            "score_method": f"visulogic_cascade_{stage}",
            "extracted": extracted,
            "gt": gt,
            "category": cat,
            "parse_stage": stage}
    if judge_raw:
        meta["judge_raw"] = judge_raw
    return correct, meta


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["correct"] is not None]
    n = len(scored)
    acc = (sum(r["correct"] for r in scored) / n) if n else 0.0
    per_cat: dict[str, list[float]] = {}
    stage_counts: dict[str, int] = {}
    det: list[float] = []
    for r in scored:
        m = r["meta"]
        per_cat.setdefault(str(m.get("category", "")), []).append(r["correct"])
        stage = str(m.get("parse_stage", ""))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if stage in ("box", "lang"):
            det.append(r["correct"])
    n_scored_and_unscored = len(rows)
    n_parse_fail = stage_counts.get("llm_none", 0) + stage_counts.get("model_no_answer", 0)
    return {
        "accuracy": acc,
        "n": len(rows),
        "n_scored": n,
        "per_category": {k: (sum(v) / len(v)) for k, v in sorted(per_cat.items())},
        "boxed_parser_count": stage_counts.get("box", 0),
        "language_parser_count": stage_counts.get("lang", 0),
        "llm_fallback_count": stage_counts.get("llm", 0),
        "parse_failure_count": n_parse_fail,
        "parse_failure_rate": (n_parse_fail / n_scored_and_unscored) if n_scored_and_unscored else 0.0,
        # deterministic-only accuracy: restricted to samples resolved without the LLM
        "deterministic_only_accuracy": (sum(det) / len(det)) if det else 0.0,
        "deterministic_only_n": len(det),
    }
