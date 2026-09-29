"""Task-level helper functions for remote judge execution."""

from __future__ import annotations

from typing import Optional

from .client import JudgeClient
from .types import JudgeRequest, JudgeTask, ScoreResult


def _score(
    judge_client: JudgeClient,
    request: JudgeRequest,
    *,
    temperature: float,
    max_tokens: int,
    extra_body: Optional[dict],
) -> ScoreResult:
    kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    return judge_client.score(request, **kwargs)


def score_vqa_outcome(
    judge_client: JudgeClient,
    *,
    question: str,
    candidate_answer: str,
    reference_answer: str,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    extra_body: Optional[dict] = None,
) -> ScoreResult:
    """Score candidate answer correctness against a reference answer."""
    return _score(
        judge_client,
        JudgeRequest(
            task=JudgeTask.VQA_OUTCOME,
            question=question,
            candidate_answer=candidate_answer,
            reference_answer=reference_answer,
            image_path=image_path,
            image_url=image_url,
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )


def score_consistency(
    judge_client: JudgeClient,
    *,
    candidate_reasoning: str,
    candidate_answer: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    extra_body: Optional[dict] = None,
) -> ScoreResult:
    """Score whether reasoning and final answer are mutually consistent."""
    return _score(
        judge_client,
        JudgeRequest(
            task=JudgeTask.CONSISTENCY,
            candidate_reasoning=candidate_reasoning,
            candidate_answer=candidate_answer,
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )


def score_vqa_process(
    judge_client: JudgeClient,
    *,
    question: str,
    candidate_reasoning: str,
    reference_answer: str,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    extra_body: Optional[dict] = None,
) -> ScoreResult:
    """Score candidate reasoning quality for a visual question."""
    return _score(
        judge_client,
        JudgeRequest(
            task=JudgeTask.VQA_PROCESS,
            question=question,
            candidate_reasoning=candidate_reasoning,
            reference_answer=reference_answer,
            image_path=image_path,
            image_url=image_url,
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )


def score_code_validity(
    judge_client: JudgeClient,
    *,
    candidate_code: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    extra_body: Optional[dict] = None,
) -> ScoreResult:
    """Score whether candidate code is meaningful and usable."""
    return _score(
        judge_client,
        JudgeRequest(
            task=JudgeTask.CODE_VALIDITY,
            candidate_reasoning=candidate_code,
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
