# Modified for anonymous review: release paths, configuration, and documentation.
"""VisualProbe scorer. ADAPTED_LLM_JUDGE.

397B text-only judge with official Mini-o3 prompt semantics (pinned commit
2c5a0dedb5279eff2c0e6049aac05de97bf7a2b3, verl/utils/reward_score/general_qa_tool.py).
Deliberate deviations from upstream (recorded as adapted protocol):
  - judge model: qwen3.5-397b (text-only, temperature=0) instead of gpt-4o-2024-11-20;
  - strict parser ``Score:\\s*[01]`` instead of upstream ``"1" in response``;
  - exhausted retries -> UNSCORED (run INCOMPLETE) instead of upstream is_filter.
"""

from __future__ import annotations

from typing import Optional

from ..common import extract_answer_tag, extract_model_answer, is_model_no_answer
from ..judge import GcepTextJudge

PROTOCOL_NAME = "visualprobe_qwen3_5_vl_395b_official_semantics_t0"
SCORING_LAYER = "ADAPTED_LLM_JUDGE"
REQUIRES_JUDGE = True

# Verbatim from Mini-o3 @2c5a0ded (upstream gpt-4o judge).
SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of "
    "generative outputs for question-answer pairs.\nYour task is to compare the "
    "predicted answer with the correct answer and determine if they match "
    "meaningfully. Here's how you can accomplish the task:\n------\n##INSTRUCTIONS:\n"
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)

QUERY_PROMPT = (
    "I will give you a question related to an image and the following text as inputs:\n\n"
    "1. **Question Related to the Image**: {question}\n"
    "2. **Ground Truth Answer**: {ground_truth}\n"
    "3. **Model Predicted Answer**: {prediction}\n\n"
    "Your task is to evaluate the model's predicted answer against the ground truth "
    "answer, based on the context provided by the question related to the image. "
    "Consider the following criteria for evaluation:\n"
    "- **Relevance**: Does the predicted answer directly address the question posed, "
    "considering the information provided by the given question?\n"
    "- **Accuracy**: Compare the predicted answer to the ground truth answer. You need "
    "to evaluate from the following two perspectives:\n"
    "(1) If the ground truth answer is open-ended, consider whether the prediction "
    "accurately reflects the information given in the ground truth without introducing "
    "factual inaccuracies. If it does, the prediction should be considered correct.\n"
    "(2) If the ground truth answer is a definitive answer, strictly compare the model's "
    "prediction to the actual answer. Pay attention to unit conversions such as length "
    "and angle, etc. As long as the results are consistent, the model's prediction "
    "should be deemed correct.\n"
    "**Output Format**:\nYour response should include an integer score indicating the "
    "correctness of the prediction: 1 for correct and 0 for incorrect. Note that 1 means "
    "the model's prediction strictly aligns with the ground truth, while 0 means it does "
    "not.\nThe format should be \"Score: 0 or 1\""
)

PROMPT_HASH_SEED = SYSTEM_PROMPT + "\n" + QUERY_PROMPT

_DATASET_REVISIONS = {
    "VisualProbe_Easy": "0e665a57946d4473deb9975a13f8ddba973b8026",
    "VisualProbe_Medium": "25d5ff5bce48147be34fd2655617ab17dc55b3fa",
    "VisualProbe_Hard": "bbe9b7a6c41d49844f45e51916ec0a60d1f0e378",
}


def build_judge_prompt(question: str, ground_truth: str, prediction: str) -> str:
    return QUERY_PROMPT.format(question=question, ground_truth=ground_truth,
                               prediction=prediction)


def prompt_hash() -> str:
    import hashlib
    return hashlib.sha256(PROMPT_HASH_SEED.encode("utf-8")).hexdigest()[:16]


def score_sample(sample: dict, prediction: str, *, judge: Optional[GcepTextJudge] = None,
                 benchmark: str = "VisualProbe_Easy", **ctx) -> tuple[Optional[float], dict]:
    if judge is None:
        raise ValueError("visualprobe scorer requires judge=GcepTextJudge")
    question = str(sample.get("question", ""))
    gt = str(sample.get("answer", "")).strip()
    if is_model_no_answer(prediction):
        # Model-side format failure -> wrong, no judge call needed.
        return 0.0, {"status": "SCORED", "score_method": "model_no_answer",
                     "extracted_prediction": "", "gt": gt}
    # Judge sees the final answer, not the reasoning trace (upstream passes the
    # model's answer string; images=[] upstream, and we keep it text-only).
    pred_answer = extract_answer_tag(prediction) or extract_model_answer(prediction)
    prompt = build_judge_prompt(question, gt, pred_answer)
    # System prompt semantics are prepended text-only (our client has no system-role
    # helper; content stays strictly textual, images==[] by construction).
    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt
    outcome = judge.judge(
        full_prompt,
        benchmark_id=benchmark,
        dataset_revision=_DATASET_REVISIONS.get(benchmark, ""),
        sample_id=str(sample.get("index", "")),
        question=question, gold=gt, prediction=pred_answer,
        prompt_hash=prompt_hash(),
    )
    meta = {
        "status": outcome.status,
        "score_method": "qwen397b_text_only_strict_parser",
        "extracted_prediction": pred_answer[:500],
        "gt": gt,
        "judge_raw": {
            "raw_text": outcome.raw_text[:500],
            "attempts": outcome.attempts,
            "cache_hit": outcome.cache_hit,
            "returned_model": outcome.returned_model,
            "error": outcome.error,
        },
    }
    if outcome.status != "SCORED":
        return None, meta
    return float(outcome.score), meta


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["correct"] is not None]
    n = len(scored)
    return {
        "avg_at_1": (sum(r["correct"] for r in scored) / n) if n else 0.0,
        "n": len(rows),
        "n_scored": n,
        "note": "Main evaluation Avg@1 (near-greedy). Adapted protocol; NOT upstream Avg@32.",
    }
