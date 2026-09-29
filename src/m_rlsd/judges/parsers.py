"""Structured response parsers for judge outputs."""

from __future__ import annotations

import json
import math
from typing import Any

from .prompts import get_prompt_spec
from .types import JudgeTask, ScoreResult


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidate = _strip_code_fences(text)
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no JSON object found in judge output")


def _parse_numeric_score(value: Any, *, field_name: str, allowed_values: set[int] | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")
    numeric_value = float(value)
    if allowed_values is not None and numeric_value not in allowed_values:
        raise ValueError(f"{field_name} must be one of {sorted(allowed_values)}")
    return numeric_value


def _parse_integer_range(value: Any, *, field_name: str, minimum: int, maximum: int) -> float:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return float(value)


def _normalize_score(parsed: dict[str, Any], task: JudgeTask) -> tuple[float | None, dict[str, float]]:
    if task == JudgeTask.VQA_PROCESS:
        scores = {
            "process_correctness": _parse_integer_range(
                parsed["process_correctness"],
                field_name="process_correctness",
                minimum=1,
                maximum=10,
            ),
            "text_quality": _parse_integer_range(
                parsed["text_quality"],
                field_name="text_quality",
                minimum=1,
                maximum=10,
            ),
        }
        return sum(scores.values()) / len(scores), scores

    score = _parse_numeric_score(parsed["score"], field_name="score", allowed_values={0, 1})
    return score, {}


def parse_judge_response(task: JudgeTask, raw_text: str, raw_response: dict[str, Any]) -> ScoreResult:
    """Parse a remote judge response into a normalized result."""
    spec = get_prompt_spec(task)
    try:
        parsed = _extract_first_json_object(raw_text)
        missing_keys = [key for key in spec.expected_keys if key not in parsed]
        if missing_keys:
            raise ValueError(f"missing expected keys: {missing_keys}")
        score, scores = _normalize_score(parsed, task)
        return ScoreResult(
            task=task,
            score=score,
            scores=scores,
            raw_text=raw_text,
            raw_response=raw_response,
            parsed=parsed,
            valid=True,
        )
    except Exception as exc:
        return ScoreResult(
            task=task,
            raw_text=raw_text,
            raw_response=raw_response,
            valid=False,
            error=str(exc),
        )
