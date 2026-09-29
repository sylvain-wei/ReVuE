"""Type definitions for remote judge and RM tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class JudgeTask(str, Enum):
    """Supported judge task families."""

    VQA_OUTCOME = "vqa_outcome"
    VQA_PROCESS = "vqa_process"
    CONSISTENCY = "consistency"
    CODE_VALIDITY = "code_validity"


class ParseMode(str, Enum):
    """Supported output parsing modes for judge prompts."""

    JSON_OBJECT = "json_object"
    TEXT = "text"


@dataclass
class JudgeRequest:
    """Task-specific input payload for remote judge calls."""

    task: JudgeTask
    question: str = ""
    candidate_answer: str = ""
    reference_answer: str = ""
    candidate_reasoning: str = ""
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    sub_image_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    """Normalized response object for judge task outputs."""

    task: JudgeTask
    score: Optional[float] = None
    scores: dict[str, float] = field(default_factory=dict)
    raw_text: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)
    parsed: dict[str, Any] = field(default_factory=dict)
    valid: bool = False
    error: Optional[str] = None
