"""High-level client for remote judge and RM task execution."""

from __future__ import annotations

from typing import Optional

from m_rlsd.inference.remote_vlm_client import RemoteVLMClient, RemoteVLMConfig

from .parsers import parse_judge_response
from .prompts import get_prompt_spec
from .types import JudgeRequest, ScoreResult


class JudgeClient:
    """Task-oriented facade over the remote OpenAI-compatible VLM client."""

    def __init__(
        self,
        config: RemoteVLMConfig,
        *,
        remote_client: Optional[RemoteVLMClient] = None,
    ) -> None:
        self.config = config
        self.remote_client = remote_client or RemoteVLMClient(config)

    def healthcheck(self) -> dict:
        """Return the remote models listing."""
        return self.remote_client.healthcheck()

    def score(
        self,
        request: JudgeRequest,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        extra_body: Optional[dict] = None,
    ) -> ScoreResult:
        """Execute one judge task and parse the response."""
        prompt_spec = get_prompt_spec(request.task)
        messages = prompt_spec.render_messages(request)
        raw_text, raw_response = self.remote_client.generate_text(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        return parse_judge_response(request.task, raw_text, raw_response)
