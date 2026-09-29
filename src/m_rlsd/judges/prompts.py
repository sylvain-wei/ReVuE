"""Prompt registry for remote judge and RM tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from m_rlsd.inference.remote_vlm_client import RemoteVLMClient

from .types import JudgeRequest, JudgeTask, ParseMode


DEFAULT_SYSTEM_PROMPT = (
    "You are a precise multimodal judge. Follow the requested output schema exactly "
    "and do not add any extra commentary outside the JSON object."
)


@dataclass(frozen=True)
class JudgePromptSpec:
    """Prompt definition and rendering rules for one judge task."""

    task: JudgeTask
    parse_mode: ParseMode
    uses_image: bool
    expected_keys: tuple[str, ...] = field(default_factory=tuple)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def render_messages(self, request: JudgeRequest) -> list[dict]:
        """Build OpenAI-compatible messages for this task."""
        if self.uses_image and not (request.image_path or request.image_url):
            raise ValueError(f"{self.task.value} requires image_path or image_url")
        user_text = _render_user_prompt(self.task, request)
        user_message = _build_user_message(
            user_text,
            image_path=request.image_path if self.uses_image else None,
            image_url=request.image_url if self.uses_image else None,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            user_message,
        ]


def _build_user_message(text: str, *, image_path: str | None, image_url: str | None) -> dict:
    if image_path or image_url:
        return RemoteVLMClient.build_multimodal_user_message(
            text,
            image_path=image_path,
            image_url=image_url,
        )
    return RemoteVLMClient.build_text_user_message(text)


def _render_user_prompt(task: JudgeTask, request: JudgeRequest) -> str:
    if task == JudgeTask.VQA_OUTCOME:
        return f"""You are an expert about question answering. I will provide a question, a generated answer to the question, and the reference answer. Please evaluate the correctness of the generated answer. If the generated answer is consistent with the reference answer, score it as 1. Otherwise, score it as 0.

Question: {request.question}
Generated Answer: {request.candidate_answer}
Reference Answer: {request.reference_answer}

Return exactly one JSON object with this schema:
{{"score": 0 or 1}}
"""
    if task == JudgeTask.VQA_PROCESS:
        return f"""Evaluate the quality of the candidate reasoning process for the visual question.

Question: {request.question}
Candidate Reasoning: {request.candidate_reasoning}
Reference Answer: {request.reference_answer}

Score the reasoning on these dimensions:
- process_correctness: integer from 1 to 10
- text_quality: integer from 1 to 10

Return exactly one JSON object with this schema:
{{"process_correctness": 1-10, "text_quality": 1-10}}
"""
    if task == JudgeTask.CONSISTENCY:
        return f"""Evaluate whether the candidate reasoning and final answer are mutually consistent.

Candidate Reasoning: {request.candidate_reasoning}
Candidate Final Answer: {request.candidate_answer}

Return exactly one JSON object with this schema:
{{"score": 0 or 1}}
"""
    if task == JudgeTask.CODE_VALIDITY:
        return f"""Determine whether the candidate code is meaningful and useful.

Candidate Code:
{request.candidate_reasoning}

Return exactly one JSON object with this schema:
{{"score": 0 or 1}}
"""
    raise ValueError(f"unsupported judge task: {task}")


PROMPT_REGISTRY: dict[JudgeTask, JudgePromptSpec] = {
    JudgeTask.VQA_OUTCOME: JudgePromptSpec(
        task=JudgeTask.VQA_OUTCOME,
        parse_mode=ParseMode.JSON_OBJECT,
        uses_image=True,
        expected_keys=("score",),
    ),
    JudgeTask.VQA_PROCESS: JudgePromptSpec(
        task=JudgeTask.VQA_PROCESS,
        parse_mode=ParseMode.JSON_OBJECT,
        uses_image=True,
        expected_keys=("process_correctness", "text_quality"),
    ),
    JudgeTask.CONSISTENCY: JudgePromptSpec(
        task=JudgeTask.CONSISTENCY,
        parse_mode=ParseMode.JSON_OBJECT,
        uses_image=False,
        expected_keys=("score",),
    ),
    JudgeTask.CODE_VALIDITY: JudgePromptSpec(
        task=JudgeTask.CODE_VALIDITY,
        parse_mode=ParseMode.JSON_OBJECT,
        uses_image=False,
        expected_keys=("score",),
    ),
}


def get_prompt_spec(task: JudgeTask) -> JudgePromptSpec:
    """Return the prompt specification for a judge task."""
    return PROMPT_REGISTRY[task]
