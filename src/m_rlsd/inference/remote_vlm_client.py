# Modified for this anonymized research release: portable paths and release cleanup.
"""OpenAI-compatible multimodal client for remote VLM services."""

from __future__ import annotations

import base64
import json
import mimetypes
import socket
import time
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional
from urllib import error, request


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class RemoteVLMConfig:
    """Configuration for a remote OpenAI-compatible multimodal endpoint."""

    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout_sec: float = 120.0
    max_retries: int = 2
    retry_backoff_sec: float = 2.0

    def normalized_base_url(self) -> str:
        """Return a normalized base URL that ends in `/v1`."""
        base = self.base_url.strip().rstrip("/")
        if not base:
            raise ValueError("remote_api.base_url is required")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    def validate(self) -> None:
        """Fail fast on incomplete remote API configuration."""
        if not self.model.strip():
            raise ValueError("remote_api.model is required")
        if self.timeout_sec <= 0:
            raise ValueError("remote_api.timeout_sec must be > 0")
        if self.max_retries < 0:
            raise ValueError("remote_api.max_retries must be >= 0")
        if self.retry_backoff_sec < 0:
            raise ValueError("remote_api.retry_backoff_sec must be >= 0")
        self.normalized_base_url()

    @staticmethod
    def expand_value(value: str) -> str:
        """Expand environment placeholders and treat unresolved placeholders as empty."""
        expanded = os.path.expandvars(value).strip()
        if expanded.startswith("${") and expanded.endswith("}"):
            return ""
        return expanded


class RemoteVLMClient:
    """Minimal OpenAI-compatible client with multimodal helpers."""

    def __init__(self, config: RemoteVLMConfig):
        self.config = config
        self.config.validate()
        self.base_url = self.config.normalized_base_url()

    def healthcheck(self) -> dict[str, Any]:
        """Return `/v1/models` response."""
        return self._request_json("GET", "/models")

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stream: bool = False,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Call `/v1/chat/completions`."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if extra_body:
            payload.update(extra_body)
        return self._request_json("POST", "/chat/completions", payload)

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Return the assistant text content and raw response body."""
        response = self.chat_completions(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("remote VLM response did not include any choices")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "\n".join(part for part in text_parts if part)
        return str(content).strip(), response

    @staticmethod
    def build_text_user_message(text: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        }

    @classmethod
    def build_multimodal_user_message(
        cls,
        text: str,
        *,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build a user message containing text plus one image."""
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        resolved_image_url = cls._resolve_image_url(image_path=image_path, image_url=image_url)
        if resolved_image_url:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": resolved_image_url},
                }
            )
        return {"role": "user", "content": content}

    @staticmethod
    def _resolve_image_url(
        *,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
    ) -> Optional[str]:
        if image_url:
            return image_url
        if not image_path:
            return None

        # Release: thyme_rl_train 的 parquet 把原图以 base64 字符串存进 `path` 字段
        # （dataset preprocessor 包成 {'bytes':None,'path':<base64>}）。这种情况下
        # image_path 其实是 base64 内容而非文件路径，直接包成 data URL 返回；
        # 否则会 Path.exists()==False -> FileNotFoundError -> judge 快速失败。
        if isinstance(image_path, str):
            s = image_path.strip()
            if s.startswith("data:"):
                return s
            # 裸 base64（很长、非 http、且不是真实存在的文件路径）-> 直接包成 data URL。
            # 注意：base64 字母表含 '/'，不能用 '含不含斜杠' 判断；也不能对超长串
            # 直接 Path().exists()（会报 file name too long）。
            if len(s) > 255 and not s.startswith(("http://", "https://")):
                import base64 as _b64
                try:
                    _b64.b64decode(s[:64] + "=" * (-len(s[:64]) % 4))  # 粗验是 base64
                    return f"data:image/png;base64,{s}"
                except Exception:
                    pass  # 不是合法 base64，按文件路径走

        path = Path(image_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"image file not found: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        body = None if payload is None else json.dumps(payload).encode("utf-8")

        for attempt in range(self.config.max_retries + 1):
            req = request.Request(url, data=body, headers=headers, method=method)
            try:
                with request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
            except error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                should_retry = exc.code in RETRYABLE_STATUS_CODES and attempt < self.config.max_retries
                if should_retry:
                    time.sleep(self.config.retry_backoff_sec * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"remote VLM request failed: {exc.code} {exc.reason}; body={details}"
                ) from exc
            except (error.URLError, TimeoutError, socket.timeout) as exc:
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_sec * (attempt + 1))
                    continue
                raise RuntimeError(f"remote VLM request failed: {exc}") from exc
